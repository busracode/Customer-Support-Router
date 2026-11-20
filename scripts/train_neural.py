"""Word2Vec + SGD ile eğitilen feedforward ağ; test sadece son ölçümde kullanılır."""

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gensim
from gensim.models import Word2Vec
import joblib
import numpy as np
import sklearn
from sklearn.metrics import classification_report, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_baseline import read_split, reject, metrics
from src.models.neural import AveragedWord2VecNetwork, stable_hash, tokenize


def main():
    started = perf_counter()
    train_x, train_y = read_split("train")
    val_x, val_y = read_split("validation")
    labels = np.unique(train_y)
    config = {
        "word2vec": {"vector_size": 100, "window": 5, "min_count": 1,
                     "sg": 1, "negative": 5, "epochs": 40, "seed": 42, "workers": 1},
        "network": {"hidden_layer_sizes": (256, 128), "activation": "relu", "solver": "sgd",
                    "batch_size": 128, "learning_rate_init": 0.02, "momentum": 0.9,
                    "alpha": 0.001, "random_state": 42},
        "max_epochs": 100, "patience": 12,
        "selection": "Minimum validation log loss; threshold selected on validation after checkpoint selection.",
    }
    print("Word2Vec yalnizca egitim mesajlarindan ogreniyor...", flush=True)
    with threadpool_limits(limits=1):
        embeddings = Word2Vec(sentences=[tokenize(text) for text in train_x],
                              hashfxn=stable_hash, **config["word2vec"])
        word2vec_seconds = perf_counter() - started
        model = AveragedWord2VecNetwork(dict(embeddings.wv.key_to_index),
                                       embeddings.wv.vectors.copy(), StandardScaler(),
                                       MLPClassifier(**config["network"]))
        train_features = model.scaler.fit_transform(model.features(train_x))
        val_features = model.scaler.transform(model.features(val_x))
        best_loss = float("inf")
        best_classifier = None
        best_epoch = 0
        history = []
        print("Sinir agi SGD ve geri yayilim ile egitiliyor...", flush=True)
        for epoch in range(1, config["max_epochs"] + 1):
            model.classifier.partial_fit(train_features, train_y, classes=labels)
            probabilities = model.classifier.predict_proba(val_features)
            loss = float(log_loss(val_y, probabilities, labels=labels))
            accuracy = float(np.mean(labels[probabilities.argmax(axis=1)] == val_y))
            history.append({"epoch": epoch, "train_loss": float(model.classifier.loss_),
                            "validation_loss": loss, "validation_accuracy": accuracy})
            if loss < best_loss - 1e-4:
                best_loss, best_epoch = loss, epoch
                best_classifier = copy.deepcopy(model.classifier)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch}: validation loss={loss:.4f}, accuracy={accuracy:.4f}", flush=True)
            if epoch - best_epoch >= config["patience"]:
                break
        model.classifier = best_classifier
        training_seconds = perf_counter() - started
        candidates = []
        val_probs = model.predict_proba(val_x)
        for threshold in np.linspace(0, 0.95, 96):
            result = metrics(val_y, reject(val_probs, model.classes_, threshold))
            objective = (result["in_scope_accuracy"] + result["oos_recall"]) / 2
            candidates.append({"threshold": float(threshold), "objective": objective, **result})
        selected = max(candidates, key=lambda item: item["objective"])
        threshold = selected["threshold"]
        path = ROOT / "artifacts/neural.joblib"
        path.parent.mkdir(exist_ok=True)
        # Çıkarım için yalnızca vektörler, ölçekleyici ve ağ gerekir.
        reference_probs = model.predict_proba(val_x[:10])
        joblib.dump({"pipeline": model, "threshold": threshold}, path, compress=3)
        saved = joblib.load(path)
        # Aynı batch boyutu kullanılır; float32 matris hesabı farklı batch'lerde
        # küçük yuvarlama farkları üretebilir.
        np.testing.assert_allclose(saved["pipeline"].predict_proba(val_x[:10]), reference_probs,
                                   rtol=1e-5, atol=1e-7)
        model = saved["pipeline"]
        test_x, test_y = read_split("test")
        baseline = json.loads((ROOT / "results/baseline_metrics.json").read_text(encoding="utf-8"))
        hashes = {name: hashlib.sha256((ROOT / f"data/processed/{name}.jsonl").read_bytes()).hexdigest()
                  for name in ("train", "validation", "test")}
        if hashes != baseline["data_sha256"]:
            raise ValueError("Baseline ile veri ayrimi ayni degil.")
        probs = model.predict_proba(test_x)
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
        predictions = reject(probs, model.classes_, threshold)
        indices = np.random.default_rng(42).integers(0, len(test_x), size=220)
        times = []
        for i, index in enumerate(indices):
            start = perf_counter()
            reject(model.predict_proba([test_x[index]]), model.classes_, threshold)
            elapsed = (perf_counter() - start) * 1000
            if i >= 20:
                times.append(elapsed)
    results = {
        "model": "Mean Word2Vec + feedforward neural network (SGD)",
        "test": metrics(test_y, predictions),
        "test_without_threshold": metrics(test_y, model.classes_[probs.argmax(axis=1)]),
        "validation_selected": selected, "best_epoch": best_epoch,
        "epochs_run": len(history), "config": config,
        "vocabulary_size": len(model.vocabulary),
        "training_seconds": training_seconds, "word2vec_training_seconds": word2vec_seconds,
        "model_size_mb": path.stat().st_size / 1_000_000,
        "size_definition": "Compressed joblib including word vectors, vocabulary, scaler, network and threshold; decimal MB.",
        "latency_mean_ms": float(np.mean(times)), "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_runs": 200, "warmup_runs": 20,
        "latency_scope": "Single-message normalization, tokenization, averaged vectors, scaling, network and rejection; excludes loading/UI.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "numeric_threads": 1},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "gensim": gensim.__version__},
        "data_sha256": hashes,
        "confidence_note": "Softmax probability is not calibrated. No known tokens forces OOS as a rule, not model confidence.",
    }
    output = ROOT / "results"
    for name, payload in (("neural_metrics.json", results), ("neural_history.json", history),
                          ("neural_thresholds.json", candidates),
                          ("neural_classification_report.json", classification_report(test_y, predictions, output_dict=True, zero_division=0))):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [{"text": text, "expected": actual, "predicted": predicted, "confidence": float(prob.max())}
              for text, actual, predicted, prob in zip(test_x, test_y, predictions, probs) if actual != predicted]
    (output / "neural_errors.json").write_text(json.dumps(errors[:30], ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Model comparison", "", "Same held-out CLINC150 test set; CPU, one numeric thread, 200 timed single-message calls.", "",
             "| Model | Accuracy | OOS recall | False rejection (in-scope) | Size MB | Mean ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in (baseline, results):
        metric = result["test"]
        lines.append(f"| {result['model']} | {metric['accuracy']:.2%} | {metric['oos_recall']:.2%} | "
                     f"{metric['in_scope_false_rejection_rate']:.2%} | {result['model_size_mb']:.2f} | {result['latency_mean_ms']:.3f} |")
    lines += ["", "Size includes all inference assets in compressed joblib. Latency excludes loading and UI.",
              "Thresholds and neural checkpoint were selected on validation only. Transformer models are not trained yet."]
    (output / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
