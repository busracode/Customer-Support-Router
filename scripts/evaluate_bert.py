"""İndirilen BERT'i CPU üzerinde sabit test kümesiyle değerlendirir."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

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
from scripts.train_baseline import read_split, reject, metrics
from src.preprocessing import normalize_text


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = ROOT / "colab/bert_model"
    routing = json.loads((directory / "routing_config.json").read_text())
    training = json.loads((directory / "training_report.json").read_text())
    hashes = {name: hashlib.sha256((ROOT / f"data/processed/{name}.jsonl").read_bytes()).hexdigest()
              for name in ("train", "validation", "test")}
    baseline = json.loads((ROOT / "results/baseline_metrics.json").read_text())
    assert hashes == routing["data_sha256"] == baseline["data_sha256"]
    assert hashlib.sha256((ROOT / "src/preprocessing.py").read_bytes()).hexdigest() == routing["normalization_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True,
                                                              use_safetensors=True).cpu().eval()
    classes = np.array(routing["labels"])
    assert [model.config.id2label[i] for i in range(len(classes))] == list(classes)
    threshold = routing["threshold"]

    def probabilities(texts, padding="max_length"):
        inputs = tokenizer([normalize_text(text) for text in texts], padding=padding,
                           truncation=True, max_length=routing["max_length"], return_tensors="pt")
        with torch.inference_mode():
            return model(**inputs).logits.softmax(dim=-1).numpy()

    with threadpool_limits(limits=1):
        sample = probabilities(["Where is my order?"])
        print("Model yuklendi. Ornek:", reject(sample, classes, threshold)[0], flush=True)
        texts, truth = read_split("test")
        batches = []
        torch.set_num_threads(4)
        for start in range(0, len(texts), 16):
            batches.append(probabilities(texts[start:start + 16], padding=True))
            if start % 320 == 0:
                print(f"Test: {min(start + 16, len(texts))}/{len(texts)}", flush=True)
        probs = np.concatenate(batches)
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        predicted = reject(probs, classes, threshold)
        timings = []
        torch.set_num_threads(1)
        print("200 tek mesajlik CPU sure olcumu...", flush=True)
        for i, index in enumerate(np.random.default_rng(42).integers(0, len(texts), size=220)):
            start = perf_counter()
            reject(probabilities([texts[index]]), classes, threshold)
            if i >= 20:
                timings.append((perf_counter() - start) * 1000)
    inference_files = [p for p in directory.iterdir() if p.is_file() and p.name not in
                       {"training_args.bin", "training_report.json"}]
    result = {
        "model": "Fine-tuned BERT FP32", "test": metrics(truth, predicted),
        "test_without_threshold": metrics(truth, classes[probs.argmax(axis=1)]),
        "validation_selected": training["validation_selected"],
        "training_seconds": training["training_seconds"],
        "model_size_mb": sum(p.stat().st_size for p in inference_files) / 1e6,
        "size_definition": "Uncompressed safetensors, tokenizer and inference config; decimal MB. Baselines use compressed joblib.",
        "latency_mean_ms": float(np.mean(timings)), "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_runs": 200, "warmup_runs": 20,
        "latency_scope": "CPU FP32; normalization, tokenization (fixed padding 64), forward, softmax and rejection; no loading/UI.",
        "accuracy_evaluation": "CPU FP32, four threads, batches of 16, dynamic padding; max_length remains 64. Latency measured separately at one thread.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "numeric_threads": 1},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "data_sha256": hashes,
        "model_sha256": hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest(),
    }
    output = ROOT / "results"
    for name, payload in (("bert_metrics.json", result),
                          ("bert_classification_report.json", classification_report(truth, predicted, output_dict=True, zero_division=0))):
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    errors = [{"text": text, "expected": actual, "predicted": guess, "confidence": float(prob.max())}
              for text, actual, guess, prob in zip(texts, truth, predicted, probs) if actual != guess]
    (output / "bert_errors.json").write_text(json.dumps(errors[:30], indent=2), encoding="utf-8")
    results = [baseline, json.loads((output / "neural_metrics.json").read_text()), result]
    lines = ["# Model comparison", "", "Same held-out test data; local CPU, one numeric thread, 200 timed calls.", "",
             "| Model | Accuracy | OOS recall | False rejection | Size MB | Mean ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        m = row["test"]
        lines.append(f"| {row['model']} | {m['accuracy']:.2%} | {m['oos_recall']:.2%} | {m['in_scope_false_rejection_rate']:.2%} | {row['model_size_mb']:.2f} | {row['latency_mean_ms']:.3f} |")
    lines += ["", "Size includes inference assets: compressed joblib for first two models, uncompressed safetensors/tokenizer for BERT.",
              "Latency includes preprocessing and rejection; excludes loading/UI. Each model was measured in a separate run.",
              "Thresholds selected on validation only. Distilled and quantized student is still pending."]
    (output / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
