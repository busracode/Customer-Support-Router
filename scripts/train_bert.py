"""Colab GPU üzerinde BERT eğitimi. Test verisini okumaz."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import normalize_text


def validate_data(root):
    manifest = json.loads((root / "colab_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Dosya ozeti uyusmuyor: {name}")
    data_dir = root / "data/processed"
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    if len(labels) != len(set(labels)) or "oos" not in labels:
        raise ValueError("Etiket listesi gecersiz.")
    data = {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["data_sha256"][split]:
            raise ValueError("Baseline veri ayrimi ile uyusmuyor.")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any(not row["text"].strip() or row["label"] not in labels for row in rows):
            raise ValueError("Bos metin veya bilinmeyen etiket.")
        if set(row["label"] for row in rows) != set(labels):
            raise ValueError("Veride eksik sinif var.")
        data[split] = rows
    keys = [{" ".join(row["text"].casefold().split()) for row in data[split]}
            for split in ("train", "validation")]
    if keys[0] & keys[1]:
        raise ValueError("Egitim ve dogrulama metinleri cakisiiyor.")
    return manifest, labels, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/bert_run")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    manifest, labels, data = validate_data(ROOT)
    print({name: len(rows) for name, rows in data.items()}, flush=True)
    if args.check_data:
        print("Veri ve dosya ozetleri dogrulandi. Test verisi okunmadi.")
        return
    if args.epochs < 1 or args.batch_size < 1 or not 8 <= args.max_length <= 512:
        parser.error("Egitim ayarlari gecersiz.")

    import numpy as np
    import torch
    import transformers
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU yok. Colab: Runtime > Change runtime type > T4 GPU.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Cikti klasoru dolu. Yeni bir run klasoru secin; onceki modeli ezmeyin.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    model_name = "google-bert/bert-base-uncased"
    label2id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            texts = [normalize_text(row["text"]) for row in rows]
            lengths = tokenizer(texts, truncation=False, padding=False)["input_ids"]
            self.truncated = sum(len(ids) > args.max_length for ids in lengths)
            self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                       max_length=args.max_length, return_tensors="pt")
            self.targets = torch.tensor([label2id[row["label"]] for row in rows], dtype=torch.long)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encodings.items()},
                    "labels": self.targets[index]}

    train_data = IntentDataset(data["train"])
    validation_data = IntentDataset(data["validation"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), label2id=label2id,
        id2label={index: label for label, index in label2id.items()}, use_safetensors=True,
    )
    base_revision = getattr(model.config, "_commit_hash", None)

    def compute_metrics(prediction):
        predicted = prediction.predictions.argmax(axis=1)
        return {"accuracy": accuracy_score(prediction.label_ids, predicted),
                "macro_f1": f1_score(prediction.label_ids, predicted, average="macro", zero_division=0)}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs, learning_rate=2e-5, weight_decay=0.01,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=32,
        warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, save_only_model=True, save_safetensors=True,
        fp16=True, optim="adamw_torch", report_to="none", logging_steps=100,
        seed=42, data_seed=42, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=validation_data, processing_class=tokenizer,
                      compute_metrics=compute_metrics)
    started = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - started
    # En iyi doğrulama kaybına sahip checkpoint otomatik yüklenmiştir.
    prediction = trainer.predict(validation_data)
    probs = torch.softmax(torch.tensor(prediction.predictions), dim=1).numpy()
    truth = np.array([row["label"] for row in data["validation"]])
    classes = np.array(labels)
    in_scope = truth != "oos"
    trials = []
    for threshold in np.linspace(0, 0.95, 96):
        predicted = classes[probs.argmax(axis=1)].copy()
        predicted[probs.max(axis=1) < threshold] = "oos"
        inside = float(np.mean(predicted[in_scope] == truth[in_scope]))
        outside = float(np.mean(predicted[~in_scope] == "oos"))
        trials.append({"threshold": float(threshold), "objective": (inside + outside) / 2,
                       "in_scope_accuracy": inside, "oos_recall": outside,
                       "accuracy": float(np.mean(predicted == truth)),
                       "in_scope_false_rejection_rate": float(np.mean(predicted[in_scope] == "oos"))})
    selected = max(trials, key=lambda row: row["objective"])
    export = args.output_dir / "bert_model"
    trainer.save_model(str(export))
    tokenizer.save_pretrained(export)
    routing = {"threshold": selected["threshold"], "max_length": args.max_length,
               "labels": labels, "normalization": "src.preprocessing.normalize_text",
               "data_sha256": manifest["data_sha256"],
               "normalization_sha256": manifest["files"]["src/preprocessing.py"]}
    (export / "routing_config.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    report = {
        "model": model_name, "base_revision": base_revision,
        "status": "trained; local held-out test and CPU benchmark still required",
        "validation_selected": selected, "threshold_trials": trials,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "training_seconds": training_seconds, "training_args": training_args.to_dict(),
        "training_history": trainer.state.log_history,
        "truncated_messages": {"train": train_data.truncated, "validation": validation_data.truncated},
        "hardware": {"gpu": torch.cuda.get_device_name(0)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
        "data_sha256": manifest["data_sha256"],
        "note": "Validation numbers are not final test metrics. Do not compare GPU timing to local CPU timings.",
    }
    (export / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # İndirilecek modelin tekrar yüklenebildiğini, aynı girdide doğrula.
    trained_model = trainer.model
    trained_model.eval()
    sample = {key: value[:2].to("cuda") for key, value in validation_data.encodings.items()}
    with torch.inference_mode():
        reference = trained_model(**sample).logits.float().cpu()
    trained_model.to("cpu")
    restored = AutoModelForSequenceClassification.from_pretrained(export, local_files_only=True).to("cuda").eval()
    with torch.inference_mode():
        actual = restored(**sample).logits.float().cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    print("EGITIM TAMAMLANDI. Model tekrar yukleme kontrolu basarili.", flush=True)
    print("Dogrulama esigi:", selected["threshold"])
    print("Model:", export)


if __name__ == "__main__":
    main()
