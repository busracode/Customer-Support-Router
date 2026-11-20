"""BERT öğretmeninden BERT-Mini'ye gerçek logit damıtma; test verisi kullanılmaz."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_bert import validate_data
from src.preprocessing import normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, labels, rows = validate_data(ROOT)
    teacher_routing = json.loads((args.teacher / "routing_config.json").read_text())
    assert manifest["data_sha256"] == teacher_routing["data_sha256"]
    assert manifest["files"]["src/preprocessing.py"] == teacher_routing["normalization_sha256"]
    assert labels == teacher_routing["labels"]
    expected_teacher = json.loads((ROOT / "teacher_manifest.json").read_text())
    for name, expected_hash in expected_teacher.items():
        assert hashlib.sha256((args.teacher / name).read_bytes()).hexdigest() == expected_hash, name

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
    from src.models.distillation import distillation_loss

    if not torch.cuda.is_available():
        raise RuntimeError("T4 GPU secili degil.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Cikti klasoru dolu; yeni tarihli klasor kullanin.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    started = perf_counter()
    student_name = "google/bert_uncased_L-4_H-256_A-4"
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    assert tokenizer.get_vocab() == student_tokenizer.get_vocab(), "Ogretmen/ogrenci sozlukleri farkli."
    max_length = teacher_routing["max_length"]
    label2id = {label: index for index, label in enumerate(labels)}

    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.inputs = tokenizer([normalize_text(row["text"]) for row in records],
                                    padding="max_length", truncation=True,
                                    max_length=max_length, return_tensors="pt")
            self.labels = torch.tensor([label2id[row["label"]] for row in records])
            self.teacher_logits = None

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.inputs.items()}
            item["labels"] = self.labels[index]
            if self.teacher_logits is not None:
                item["teacher_logits"] = self.teacher_logits[index]
            return item

    train = IntentDataset(rows["train"])
    validation = IntentDataset(rows["validation"])
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher, local_files_only=True,
                                                                use_safetensors=True).to("cuda").eval()
    assert [teacher.config.id2label[i] for i in range(len(labels))] == labels
    teacher.requires_grad_(False)
    cached = []
    print("Ogretmen yalnizca egitim mesajlarina puan veriyor...", flush=True)
    with torch.inference_mode():
        for offset in range(0, len(train), 32):
            batch = {key: value[offset:offset + 32].to("cuda") for key, value in train.inputs.items()}
            cached.append(teacher(**batch).logits.float().cpu())
            if offset % 1024 == 0:
                print(f"Ogretmen: {min(offset + 32, len(train))}/{len(train)}", flush=True)
    train.teacher_logits = torch.cat(cached).clone()
    del teacher, cached
    torch.cuda.empty_cache()
    student = AutoModelForSequenceClassification.from_pretrained(
        student_name, num_labels=len(labels), label2id=label2id,
        id2label={i: label for i, label in enumerate(labels)},
    )
    base_revision = getattr(student.config, "_commit_hash", None)

    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            soft_targets = inputs.pop("teacher_logits", None)
            outputs = model(**inputs)
            loss = outputs.loss if soft_targets is None else distillation_loss(
                outputs.logits, inputs["labels"], soft_targets, temperature=2.0, alpha=0.5)
            return (loss, outputs) if return_outputs else loss

    settings = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"), num_train_epochs=6,
        learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.1,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        save_only_model=True, save_safetensors=True, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, fp16=True, optim="adamw_torch",
        report_to="none", logging_steps=100, seed=42, data_seed=42,
    )
    trainer = DistillationTrainer(model=student, args=settings, train_dataset=train,
                                 eval_dataset=validation, processing_class=tokenizer)
    trainer.train()
    probs = torch.softmax(torch.tensor(trainer.predict(validation).predictions), dim=1).numpy()
    classes = np.array(labels)
    truth = np.array([row["label"] for row in rows["validation"]])
    inside = truth != "oos"
    trials = []
    for threshold in np.linspace(0, 0.95, 96):
        predicted = classes[probs.argmax(axis=1)].copy()
        predicted[probs.max(axis=1) < threshold] = "oos"
        known = float(np.mean(predicted[inside] == truth[inside]))
        oos = float(np.mean(predicted[~inside] == "oos"))
        trials.append({"threshold": float(threshold), "objective": (known + oos) / 2,
                       "accuracy": float(np.mean(predicted == truth)),
                       "in_scope_accuracy": known, "oos_recall": oos,
                       "in_scope_false_rejection_rate": float(np.mean(predicted[inside] == "oos"))})
    best = max(trials, key=lambda row: row["objective"])
    export = args.output_dir / "student_model"
    trainer.save_model(str(export))
    tokenizer.save_pretrained(export)
    routing = {**teacher_routing, "threshold": best["threshold"]}
    (export / "routing_config.json").write_text(json.dumps(routing, indent=2))
    report = {"status": "distilled FP32; quantization and held-out test pending",
              "model": student_name, "base_revision": base_revision,
              "teacher_sha256": expected_teacher["model.safetensors"],
              "loss": "0.5 * cross_entropy + 0.5 * T^2 * KL(teacher || student)",
              "temperature": 2.0, "alpha": 0.5,
              "training_seconds": perf_counter() - started, "validation_selected": best,
              "threshold_trials": trials, "history": trainer.state.log_history,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "training_args": settings.to_dict(), "data_sha256": manifest["data_sha256"],
              "parameters": sum(p.numel() for p in trainer.model.parameters()),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
              "gpu": torch.cuda.get_device_name(0)}
    (export / "training_report.json").write_text(json.dumps(report, indent=2))
    # Yeniden yüklemeyi CPU'da yaparak ikinci GPU model kopyasından kaçın.
    reloaded = AutoModelForSequenceClassification.from_pretrained(export, local_files_only=True).cpu().eval()
    sample = {key: value[:1] for key, value in validation.inputs.items()}
    with torch.inference_mode():
        logits = reloaded(**sample).logits
    assert logits.shape == (1, len(labels)) and torch.isfinite(logits).all()
    (export / "reload_check.json").write_text(json.dumps({"passed": True, "device": "cpu"}))
    print("DAMITMA TAMAMLANDI. CPU yukleme kontrolu basarili.", flush=True)
    print(best, flush=True)


if __name__ == "__main__":
    main()
