"""Öğretmen ağırlıklarını tekrar yükletmeden öğrenci Colab defteri oluşturur."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = ["data/processed/train.jsonl", "data/processed/validation.jsonl", "data/processed/labels.json",
             "src/__init__.py", "src/preprocessing.py", "src/models/__init__.py", "src/models/distillation.py",
             "scripts/train_bert.py", "scripts/train_student.py", "requirements-colab.txt"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    baseline = json.loads((ROOT / "results/baseline_metrics.json").read_text())
    for split in ("train", "validation"):
        assert hashes[f"data/processed/{split}.jsonl"] == baseline["data_sha256"][split]
    teacher = ROOT / "colab/bert_model"
    expected_teacher = {name: hashlib.sha256((teacher / name).read_bytes()).hexdigest()
                        for name in ("model.safetensors", "config.json", "routing_config.json", "tokenizer.json")}
    teacher_report = json.loads((teacher / "training_report.json").read_text())
    teacher_path = str(Path(teacher_report["best_checkpoint"]).parent.parent / "bert_model").replace("\\", "/")
    manifest = {"files": hashes, "data_sha256": baseline["data_sha256"], "test_data_included": False}
    with zipfile.ZipFile(ROOT / "colab/student_training_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, name)
        archive.writestr("colab_manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("teacher_manifest.json", json.dumps(expected_teacher, indent=2))
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        ast.parse(text)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(True)})

    md("""# SWE015 — Küçük modele bilgi damıtma
Önceki BERT öğretmenimiz hazır. Bu defter BERT-Mini öğrencisini gerçek etiketler
ve öğretmenin yumuşatılmış sınıf tahminleriyle eğitir. Test verisi kullanılmaz.
**Runtime → Change runtime type → T4 GPU** seç; hücreleri sırayla çalıştır.
Bu aşama sonunda FP32 öğrenci indirilir. INT8 quantization ve son test bilgisayarında ayrıca yapılacak.
""")
    md("## 1. GPU kontrolü")
    code('''import torch
assert torch.cuda.is_available(), "T4 GPU sec ve tekrar baglan."
print("GPU:", torch.cuda.get_device_name(0))
''')
    md("## 2. Yeni eğitim paketini yükle\n`colab/student_training_bundle.zip` dosyasını seç. Eski BERT paketini seçme.")
    code('''from google.colab import files
from pathlib import Path
import io, zipfile, json, hashlib
uploaded = files.upload()
assert len(uploaded) == 1, "Tek ZIP sec."
ROOT = Path("/content/support_router_student")
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    for entry in archive.infolist():
        assert (ROOT / entry.filename).resolve().is_relative_to(ROOT.resolve())
    archive.extractall(ROOT)
manifest = json.loads((ROOT / "colab_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
print("Paket dogrulandi.")
''')
    md("## 3. Kütüphaneler\nGPU destekli mevcut PyTorch kullanılır.")
    code('''import subprocess, sys
def run_visible(command, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    if status:
        raise RuntimeError(f"Islem durdu ({status}). Yukaridaki hata mesajini gonder. Log: {log_path}")
run_visible([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-colab.txt")], "/content/student_install.log")
run_visible([sys.executable, str(ROOT / "scripts/train_bert.py"), "--check-data"], "/content/student_data.log")
''')
    md("## 4. Drive'daki öğretmeni bağla\nÖnceki eğitimin model klasörü otomatik seçilmiştir. Dosyalar aynı BERT olduğundan emin olmak için doğrulanır.")
    code('''from google.colab import drive
from datetime import datetime, timezone
drive.mount("/content/drive")
TEACHER = Path(''' + repr(teacher_path) + ''')
assert (TEACHER / "model.safetensors").exists(), "Ogretmen klasoru bulunamadi. Yolu kontrol et."
RUN = Path("/content/drive/MyDrive/SWE015_Customer_Support_Router") / ("student_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
print("Ogretmen:", TEACHER)
print("Ogrenci kayit klasoru:", RUN)
''')
    md("""## 5. Öğrenciyi eğit
Önce öğretmen yalnızca eğitim mesajlarını puanlar; sonra 6 epoch öğrenci eğitilir.
Kayıp = %50 gerçek etiket kaybı + %50 yumuşak öğretmen dağılımı kaybı; sıcaklık 2.
En iyi model doğrulama kaybıyla seçilir. Güven eşiği de yalnızca doğrulamayla belirlenir.
Çalışırken yeniden başlatma. Hata olursa buradaki son mesajı sohbete gönder.
Drive'a model checkpoint'leri kaydedilir; optimizer saklanmadığından otomatik adım devamı yoktur.
""")
    code('''run_visible([sys.executable, "-u", str(ROOT / "scripts/train_student.py"),
             "--teacher", str(TEACHER), "--output-dir", str(RUN)], "/content/student_training.log")
''')
    md("## 6. Sonuç ve indirme\nBunlar doğrulama ölçümleridir; test sonucu değildir. İnen `student_model.zip` dosyasını projenin `colab/` klasörüne koy.")
    code('''import shutil
assert (RUN / "student_model/reload_check.json").exists(), "Yukleme kontrolu tamamlanmadi; once hata mesajini inceleyelim."
report = json.loads((RUN / "student_model/training_report.json").read_text())
print("Dogrulama:", report["validation_selected"])
print("Parametre sayisi:", report["parameters"])
archive = shutil.make_archive("/content/student_model", "zip", root_dir=RUN, base_dir="student_model")
shutil.copy2(archive, RUN / "student_model.zip")
shutil.copy2("/content/student_training.log", RUN / "student_training.log")
files.download(archive)
''')
    md("""Kaynak: [Google BERT-Mini](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4).
Öğretmen bu projede eğitilen BERT'tir. Kod hazırlığında Codex kullanıldı.
Sonraki aşama INT8 quantization ve tüm modellerin aynı testte karşılaştırılmasıdır.
""")
    for index, cell in enumerate(cells):
        cell["id"] = f"student-cell-{index:02d}"
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/04_student_colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(ROOT / "colab/student_training_bundle.zip")
    print("Teacher:", teacher_path)


if __name__ == "__main__":
    main()
