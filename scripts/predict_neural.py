"""Word2Vec + sinir ağıyla yeni bir mesajın niyetini tahmin eder."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from threadpoolctl import threadpool_limits

from src.models.neural import tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()
    if not args.message.strip():
        parser.error("Mesaj bos olamaz.")
    path = ROOT / "artifacts/neural.joblib"
    if not path.exists():
        parser.error("Once python scripts/train_neural.py calistirin.")
    saved = joblib.load(path)
    model = saved["pipeline"]
    if not any(token in model.vocabulary for token in tokenize(args.message)):
        print("Bilinen kelime yok: oos -> Human Agent (kural)")
        return
    with threadpool_limits(limits=1):
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
