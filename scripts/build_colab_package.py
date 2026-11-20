"""Colab veri/kod paketini ve kullanıcıya yönelik eğitim defterini oluşturur."""

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = ["data/processed/train.jsonl", "data/processed/validation.jsonl",
             "data/processed/labels.json", "src/__init__.py", "src/preprocessing.py",
             "scripts/train_bert.py", "requirements-colab.txt"]
    baseline = json.loads((ROOT / "results/baseline_metrics.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    for split in ("train", "validation", "test"):
        digest = hashlib.sha256((ROOT / f"data/processed/{split}.jsonl").read_bytes()).hexdigest()
        if digest != baseline["data_sha256"][split]:
            raise ValueError("Baseline veri ayrimi degismis.")
    manifest = {"files": hashes, "data_sha256": baseline["data_sha256"],
                "source": "https://github.com/clinc/oos-eval", "test_data_included": False}
    output = ROOT / "colab"
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(output / "bert_training_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            archive.write(ROOT / name, name)
        archive.writestr("colab_manifest.json", json.dumps(manifest, indent=2))
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        ast.parse(text)
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": text.splitlines(True)})

    md("""# SWE015 — İngilizce BERT eğitimi
Bu defter üçüncü modeli eğitir: **bert-base-uncased + 151 sınıflı sınıflandırma başlığı**.
İlk iki modelle aynı eğitim/doğrulama verisi kullanılır. Test verisi bu pakette yoktur.

**Başlamadan:** Runtime / Çalışma zamanı → Change runtime type / Çalışma zamanı türünü değiştir
→ **T4 GPU** seç. Ardından hücreleri yukarıdan aşağıya çalıştır.
GPU erişimi ücretsiz Colab kotasına bağlıdır. Model indirme ve eğitim internet gerektirir.
Eğitim süresi GPU'ya göre değişir; ilerleme çubuğundaki tahmini takip et.

Bu defter eğitim ve doğrulama sonuçları üretir. Son test ve CPU hız ölçümü daha sonra bilgisayarında yapılacak.
""")
    md("## 1. GPU kontrolü\nGPU bulunamazsa bu hücre durur; CPU üzerinde uzun bir eğitimi yanlışlıkla başlatmaz.")
    code('''import torch
assert torch.cuda.is_available(), "GPU bulunamadi. Runtime > Change runtime type > T4 GPU sec."
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
''')
    md("## 2. Eğitim paketini yükle\nBilgisayarındaki `colab/bert_training_bundle.zip` dosyasını seç. `.venv` veya model dosyalarını yüklemene gerek yok.")
    code('''from google.colab import files
from pathlib import Path
import json, hashlib, zipfile, io

uploaded = files.upload()
assert len(uploaded) == 1, "Yalnizca bert_training_bundle.zip sec."
payload = next(iter(uploaded.values()))
ROOT = Path("/content/support_router")
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        destination = (ROOT / entry.filename).resolve()
        assert destination.is_relative_to(ROOT.resolve()), "Gecersiz ZIP yolu."
    archive.extractall(ROOT)
manifest = json.loads((ROOT / "colab_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
assert manifest["test_data_included"] is False
print("Paket dogrulandi. Test verisi egitime dahil degil.")
''')
    md("## 3. Kütüphaneleri hazırla\nColab'ın GPU destekli PyTorch kurulumu kullanılır. Transformers sürümü sabittir.")
    code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-colab.txt")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/train_bert.py"), "--check-data"], check=True)
''')
    md("""## 4. Google Drive'a yedekleme
Bu hücre Google Drive bağlantısı ister. Colab'ın gösterdiği erişim adımlarını tamamla.
Model ve en fazla iki epoch checkpoint'i `MyDrive/SWE015_Customer_Support_Router` altında saklanır.
Birkaç GB boş alan bırak. Her çalıştırmada ayrı tarihli klasör oluşturulur.
Bağlantı koparsa tamamlanan epoch'ların model ağırlıkları korunur; optimizer durumu saklanmadığından
bu defter kaldığı adımda otomatik devam etmez. Yeniden başlatmadan önce kaydedilen durumu kontrol et.
""")
    code('''from google.colab import drive
from datetime import datetime, timezone
drive.mount("/content/drive")
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN = Path("/content/drive/MyDrive/SWE015_Customer_Support_Router") / ("bert_" + stamp)
print("Kayit klasoru:", RUN)
''')
    md("""## 5. BERT'i eğit
4 epoch, 16 mesajlık batch, en fazla 64 token. BERT'in tüm katmanları fine-tune edilir.
En düşük doğrulama kaybındaki model seçilir; ardından insan temsilci eşiği doğrulama verisiyle belirlenir.
Başlangıçtaki `classifier ... newly initialized` uyarısı normaldir: yeni sınıflandırma başlığı öğrenilecek.
GPU belleği hatası olursa batch değerini 8 yapıp yeni kayıt klasörüyle yeniden başlat.
Bu hücreyi başarıyla tamamlamadan indirme hücresine geçme.
""")
    code('''command = [sys.executable, "-u", str(ROOT / "scripts/train_bert.py"),
           "--output-dir", str(RUN), "--epochs", "4", "--batch-size", "16", "--max-length", "64"]
subprocess.run(command, check=True)
''')
    md("## 6. Doğrulama sonucunu görüntüle\nBunlar test sonuçları değildir. Test başarı tablosuna bu sayıları yazma.")
    code('''report = json.loads((RUN / "bert_model/training_report.json").read_text())
print("GPU:", report["hardware"]["gpu"])
print("Egitim suresi (dakika):", round(report["training_seconds"] / 60, 1))
print("Dogrulama sonuclari:", report["validation_selected"])
''')
    md("""## 7. Eğitilmiş modeli indir
`bert_model.zip` dosyası indirilecek (yaklaşık birkaç yüz MB).
Drive'da bir kopyası da kalır. İndirme başlamazsa Drive'daki kayıt klasöründen alabilirsin.
İndirilen dosyayı bilgisayarındaki proje klasörünün `colab/` dizinine koy ve sohbete tamamlandığını yaz.
Sonraki aşama: yerel test/CPU ölçümü, ardından bu BERT öğretmeninden küçük modele bilgi damıtma.
""")
    code('''import shutil
archive_path = shutil.make_archive("/content/bert_model", "zip", root_dir=RUN, base_dir="bert_model")
shutil.copy2(archive_path, RUN / "bert_model.zip")
files.download(archive_path)
print("Drive yedegi:", RUN / "bert_model.zip")
''')
    md("""## Kaynaklar ve notlar
- Veri: [CLINC150, Larson et al.](https://github.com/clinc/oos-eval)
- Model: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- Eğitim araçları: [Hugging Face Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert)
- Kod hazırlığında Codex kullanıldı; AI kullanım slaydında belirt.
- Bu veri genel intent sınıfları içerir; beyaz eşya arıza başarısı olarak sunma.
""")
    notebook = {"cells": cells, "metadata": {"colab": {"name": "03_bert_colab.ipynb"},
                 "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"bert-cell-{index:02d}"
    (ROOT / "notebooks/03_bert_colab.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Hazir:", output / "bert_training_bundle.zip")
    print("Hazir:", ROOT / "notebooks/03_bert_colab.ipynb")


if __name__ == "__main__":
    main()
