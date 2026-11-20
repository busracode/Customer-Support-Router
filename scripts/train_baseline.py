"""TF-IDF baseline eğitimi ve sabit test kümesinde ölçüm."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.preprocessing import normalize_text


def read_split(name):
    path = ROOT / f"data/processed/{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["text"] for row in rows], np.array([row["label"] for row in rows])


def reject(probs, classes, threshold):
    predictions = classes[probs.argmax(axis=1)].copy()
    predictions[probs.max(axis=1) < threshold] = "oos"
    return predictions


def metrics(truth, predictions):
    inside = truth != "oos"
    outside = ~inside
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "in_scope_accuracy": float(accuracy_score(truth[inside], predictions[inside])),
        "oos_recall": float(np.mean(predictions[outside] == "oos")),
        "in_scope_false_rejection_rate": float(np.mean(predictions[inside] == "oos")),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }


def main():
    train_x, train_y = read_split("train")
    val_x, val_y = read_split("validation")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
    ])
    started = perf_counter()
    print("Baseline egitiliyor...", flush=True)
    # Küçük matrislerde fazla CPU iş parçacığının ek yükünü sınırlıyoruz.
    with threadpool_limits(limits=1):
        model.fit(train_x, train_y)
        training_seconds = perf_counter() - started
        classes = model.classes_
        val_probs = model.predict_proba(val_x)
        candidates = []
        for threshold in np.linspace(0, 0.95, 96):
            result = metrics(val_y, reject(val_probs, classes, threshold))
            # OOS ve kapsam içi başarısına eşit ağırlık; sadece validation.
            objective = (result["in_scope_accuracy"] + result["oos_recall"]) / 2
            candidates.append({"threshold": float(threshold), "objective": objective, **result})
        best = max(candidates, key=lambda item: item["objective"])
        threshold = best["threshold"]
        print(f"Dogrulama verisinde secilen esik: {threshold:.2f}", flush=True)
        model_dir = ROOT / "artifacts"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "baseline.joblib"
        joblib.dump({"pipeline": model, "threshold": threshold}, model_path, compress=3)
        # Kaydedilen model yüklenerek test edilir; test ayar seçimine katılmaz.
        saved = joblib.load(model_path)
        model = saved["pipeline"]
        test_x, test_y = read_split("test")
        probs = model.predict_proba(test_x)
        predictions = reject(probs, classes, saved["threshold"])
        raw_predictions = classes[probs.argmax(axis=1)]
        # Sabit seed ile 200 tek-mesaj çağrısı; 20 ısınma çağrısı hariç.
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(test_x), size=220)
        timings = []
        for i, index in enumerate(indices):
            start = perf_counter()
            reject(model.predict_proba([test_x[index]]), classes, threshold)
            elapsed = (perf_counter() - start) * 1000
            if i >= 20:
                timings.append(elapsed)
    results = {
        "model": "TF-IDF (word 1-2 grams) + Logistic Regression",
        "test": metrics(test_y, predictions),
        "test_without_threshold": metrics(test_y, raw_predictions),
        "validation_selected": best,
        "threshold_selection": "Maximize mean of in-scope accuracy and OOS recall on validation only; ties choose smaller threshold.",
        "confidence_note": "Maximum predicted probability; not calibrated reliability.",
        "training_seconds": training_seconds,
        "model_size_mb": model_path.stat().st_size / 1_000_000,
        "size_definition": "Compressed joblib, including vectorizer, classifier and threshold; decimal MB.",
        "latency_mean_ms": float(np.mean(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_runs": len(timings), "warmup_runs": 20,
        "latency_scope": "Single-message normalization, TF-IDF, predict_proba and OOS rejection; excludes loading and UI.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "numeric_threads": 1},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "data_sha256": {name: hashlib.sha256((ROOT / f"data/processed/{name}.jsonl").read_bytes()).hexdigest()
                        for name in ("train", "validation", "test")},
    }
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    for name, payload in (("baseline_metrics.json", results), ("baseline_thresholds.json", candidates),
                          ("baseline_classification_report.json", classification_report(test_y, predictions, output_dict=True, zero_division=0))):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [{"text": text, "expected": actual, "predicted": predicted, "confidence": float(prob.max())}
              for text, actual, predicted, prob in zip(test_x, test_y, predictions, probs) if actual != predicted]
    (output / "baseline_errors.json").write_text(json.dumps(errors[:30], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
