"""CLINC150 bölümlerini JSONL biçiminde hazırlar, metin sızıntısını giderir."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(text):
    return " ".join(text.casefold().split())


def main():
    source = ROOT / "data/raw/clinc150_full.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    report = {}
    labels = sorted({label for _, label in data["train"]} | {"oos"})
    # Test önceliklidir: test metinleri eğitim ve doğrulamadan çıkarılır.
    # Aynı bölümdeki yinelenen metinler de tekilleştirilir.
    for original, target in (("test", "test"), ("val", "validation"), ("train", "train")):
        rows = data[original] + data[f"oos_{original}"]
        kept = []
        for index, (text, label) in enumerate(rows):
            normalized = key(text)
            if normalized in seen:
                continue
            seen.add(normalized)
            kept.append({"id": f"{original}-{index}", "text": text, "label": label})
        with (output / f"{target}.jsonl").open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report[target] = {"original": len(rows), "kept": len(kept), "removed": len(rows) - len(kept)}
        print(target, report[target])
    (output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (ROOT / "results/data_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Eğitimde herhangi bir özellik/vocabulary henüz öğrenilmez.
    sets = []
    for split in ("train", "validation", "test"):
        records = [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        sets.append({key(row["text"]) for row in records})
        assert {row["label"] for row in records} == set(labels), f"Eksik sınıf: {split}"
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    print("Kontrol basarili: 151 etiket; bolumler arasinda ayni metin yok.")


if __name__ == "__main__":
    main()
