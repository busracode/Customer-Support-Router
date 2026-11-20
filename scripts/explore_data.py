"""CLINC150 verisini indirir, özgün bölümlerini değiştirmeden inceler."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
RAW = ROOT / "data/raw/clinc150_full.json"


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print("CLINC150 indiriliyor...", flush=True)
        with urlopen(SOURCE, timeout=90) as response:
            content = response.read()
        json.loads(content)  # Geçersiz indirme dosyaya kaydedilmez.
        RAW.write_bytes(content)
    content = RAW.read_bytes()
    data = json.loads(content)
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if not required.issubset(data):
        raise ValueError("Beklenen veri bölümleri eksik.")
    report = {
        "source": SOURCE,
        "sha256": hashlib.sha256(content).hexdigest(),
        "language": "English",
        "note": "Exploration only; original splits preserved. No model trained.",
        "splits": {},
        "cross_split_text_overlap": {},
    }
    text_sets = {}
    for split in sorted(required):
        rows = data[split]
        if any(not isinstance(row, list) or len(row) != 2
               or not all(isinstance(value, str) and value.strip() for value in row)
               for row in rows):
            raise ValueError(f"Geçersiz kayıt: {split}")
        counts = Counter(label for _, label in rows)
        normalized = [" ".join(text.casefold().split()) for text, _ in rows]
        text_sets[split] = set(normalized)
        report["splits"][split] = {
            "rows": len(rows), "classes": len(counts),
            "class_counts": dict(sorted(counts.items())),
            "duplicate_texts": len(rows) - len(set(normalized)),
        }
        print(f"{split}: {len(rows)} mesaj, {len(counts)} sinif")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = text_sets[left] | text_sets[f"oos_{left}"]
        b = text_sets[right] | text_sets[f"oos_{right}"]
        report["cross_split_text_overlap"][f"{left}/{right}"] = len(a & b)
    # Test mesajlarını örnek seçimi için kullanmıyoruz.
    examples = {}
    for text, label in data["train"]:
        examples.setdefault(label, text)
    report["training_examples_by_intent"] = dict(sorted(examples.items()))
    output = ROOT / "results/data_exploration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Bolumler arasi ayni metin sayilari:", report["cross_split_text_overlap"])
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()
