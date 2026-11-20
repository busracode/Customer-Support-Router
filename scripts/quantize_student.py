"""Öğrenci FP32/INT8 ölçümleri. INT8 eşiği yalnızca doğrulamada seçilir."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import zipfile

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from threadpoolctl import threadpool_limits
from sklearn.metrics import classification_report
from src.models.quantized import quantize_model, load_quantized
from src.preprocessing import normalize_text
from scripts.train_baseline import read_split, reject, metrics


def main():
    source = ROOT / "colab/student_model"
    archive_path = ROOT / "colab/student_model.zip"
    if not source.exists() and archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                target = (ROOT / "colab" / item.filename).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError("ZIP yalnızca student_model/ klasörünü içermeli.")
            archive.extractall(ROOT / "colab")
    if not (source / "model.safetensors").exists():
        raise FileNotFoundError("colab/student_model.zip veya açılmış student_model/ bulunamadı.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    routing = json.loads((source / "routing_config.json").read_text())
    training = json.loads((source / "training_report.json").read_text())
    baseline = json.loads((ROOT / "results/baseline_metrics.json").read_text())
    hashes = {name: hashlib.sha256((ROOT / f"data/processed/{name}.jsonl").read_bytes()).hexdigest()
              for name in ("train", "validation", "test")}
    assert hashes == routing["data_sha256"] == baseline["data_sha256"]
    assert hashlib.sha256((ROOT / "src/preprocessing.py").read_bytes()).hexdigest() == routing["normalization_sha256"]
    assert training["teacher_sha256"] == json.loads((ROOT / "results/bert_metrics.json").read_text())["model_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    fp32 = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True, use_safetensors=True).cpu().eval()
    classes = np.array(routing["labels"])
    assert [fp32.config.id2label[i] for i in range(len(classes))] == list(classes)

    def predict(model, texts, padding=True):
        batch = tokenizer([normalize_text(t) for t in texts], padding=padding, truncation=True,
                          max_length=routing["max_length"], return_tensors="pt")
        with torch.inference_mode():
            probs = model(**batch).logits.softmax(dim=-1).numpy()
        assert np.isfinite(probs).all()
        return probs

    def all_probs(model, texts, label):
        outputs = []
        for start in range(0, len(texts), 16):
            outputs.append(predict(model, texts[start:start + 16]))
            if start % 800 == 0:
                print(f"{label}: {min(start + 16, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(outputs)

    destination = ROOT / "artifacts/student_int8"
    destination.mkdir(parents=True, exist_ok=True)
    print("Ogrenci yuklendi. INT8 donusumu basliyor...", flush=True)
    quantized = quantize_model(fp32)
    count = sum(isinstance(layer, torch.ao.nn.quantized.dynamic.Linear) for layer in quantized.modules())
    assert count > 0
    quantized.config.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    torch.save(quantized.state_dict(), destination / "model_int8.pt")
    metadata = {"method": "PyTorch dynamic quantization, Linear weights INT8; embeddings and LayerNorm remain FP32",
                "quantized_linear_layers": count, "engine": torch.backends.quantized.engine,
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "source_sha256": hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()}
    (destination / "quantization.json").write_text(json.dumps(metadata, indent=2))
    sample = ["Where is my order?", "zzqvzz qqqvzz"]
    reference = predict(quantized, sample)
    quantized = load_quantized(destination)
    np.testing.assert_allclose(predict(quantized, sample), reference, atol=1e-6, rtol=1e-5)
    print("INT8 kaydetme/yukleme kontrolu basarili.", flush=True)
    val_x, val_y = read_split("validation")
    val_probs = all_probs(quantized, val_x, "INT8 validation")
    trials = []
    for threshold in np.linspace(0, 0.95, 96):
        metric = metrics(val_y, reject(val_probs, classes, threshold))
        trials.append({"threshold": float(threshold),
                       "objective": (metric["in_scope_accuracy"] + metric["oos_recall"]) / 2, **metric})
    selected = max(trials, key=lambda row: row["objective"])
    (destination / "routing_config.json").write_text(json.dumps({**routing, "threshold": selected["threshold"]}, indent=2))
    output = ROOT / "results"
    (output / "student_int8_thresholds.json").write_text(json.dumps(trials, indent=2))
    test_x, test_y = read_split("test")
    for name, model, folder, chosen in (
        ("student_fp32", fp32, source, training["validation_selected"]),
        ("student_int8", quantized, destination, selected),
    ):
        torch.set_num_threads(4)
        probs = all_probs(model, test_x, name)
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        predicted = reject(probs, classes, chosen["threshold"])
        times = []
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            for i, index in enumerate(np.random.default_rng(42).integers(0, len(test_x), size=220)):
                started = perf_counter()
                reject(predict(model, [test_x[index]], padding="max_length"), classes, chosen["threshold"])
                if i >= 20:
                    times.append((perf_counter() - started) * 1000)
        excluded = {"training_args.bin", "training_report.json", "reload_check.json"}
        size = sum(path.stat().st_size for path in folder.iterdir() if path.is_file() and path.name not in excluded)
        result = {"model": name, "test": metrics(test_y, predicted),
                  "test_without_threshold": metrics(test_y, classes[probs.argmax(axis=1)]),
                  "validation_selected": chosen, "model_size_mb": size / 1e6,
                  "size_definition": "Uncompressed inference weights, tokenizer and config, decimal MB",
                  "latency_mean_ms": float(np.mean(times)), "latency_p95_ms": float(np.percentile(times, 95)),
                  "latency_runs": 200, "warmup_runs": 20,
                  "latency_scope": "Single CPU thread, normalization/tokenization, fixed padding max_length=64, model, softmax, rejection. No load/UI.",
                  "accuracy_evaluation": "Four CPU threads, batches of 16, dynamic padding; max_length=64",
                  "hardware": {"processor": platform.processor(), "platform": platform.platform(), "logical_cpus": os.cpu_count(), "numeric_threads": 1},
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                  "data_sha256": hashes, "source_sha256": metadata["source_sha256"],
                  "quantization": metadata if name.endswith("int8") else None}
        (output / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
        (output / f"{name}_classification_report.json").write_text(json.dumps(classification_report(test_y, predicted, output_dict=True, zero_division=0), indent=2))
        print(json.dumps(result["test"]), "Size MB:", result["model_size_mb"], "Latency ms:", result["latency_mean_ms"], flush=True)
    summaries = [json.loads((output / f"{name}_metrics.json").read_text())
                 for name in ("baseline", "neural", "bert", "student_fp32", "student_int8")]
    lines = ["# Model comparison", "", "Four required models plus FP32 student diagnostic; same held-out test set.", "",
             "| Model | Accuracy | OOS recall | False rejection | Size MB | CPU mean ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        m = item["test"]
        lines.append(f"| {item['model']} | {m['accuracy']:.2%} | {m['oos_recall']:.2%} | {m['in_scope_false_rejection_rate']:.2%} | {item['model_size_mb']:.2f} | {item['latency_mean_ms']:.3f} |")
    lines += ["", "All thresholds selected on validation only. Latency: 200 runs after 20 warmups, one CPU thread; loading/UI excluded.",
              "Size includes inference assets. First two models use compressed joblib; Transformer assets are uncompressed.",
              "INT8 converts Linear layers only; embeddings and LayerNorm remain FP32. FP32 student is an extra diagnostic row, not a fifth required model."]
    (output / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
