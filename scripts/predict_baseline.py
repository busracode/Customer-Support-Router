"""Örnek: python scripts/predict_baseline.py "Where is my order?" """

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib


def main():
    parser = argparse.ArgumentParser(description="Egitilen baseline ile mesaj siniflandir.")
    parser.add_argument("message")
    args = parser.parse_args()
    if not args.message.strip():
        parser.error("Mesaj bos olamaz.")
    path = ROOT / "artifacts/baseline.joblib"
    if not path.exists():
        parser.error("Once python scripts/train_baseline.py komutunu calistirin.")
    saved = joblib.load(path)
    model = saved["pipeline"]
    probs = model.predict_proba([args.message])[0]
    best = int(probs.argmax())
    label = model.classes_[best]
    print(f"Ham tahmin: {label}; skor: {probs[best]:.3f}")
    if label == "oos" or probs[best] < saved["threshold"]:
        print("Sonuc: oos -> Human Agent")
    else:
        print(f"Sonuc: {label}")
        print("Destek ekibi eslemesi sonraki asamada eklenecek.")


if __name__ == "__main__":
    main()
