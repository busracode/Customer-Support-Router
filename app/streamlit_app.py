"""Bilingual comparison UI; model inputs remain English."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
from src.inference import MODELS, load_model, predict
from src.routing import prepare_message, small_talk

st.set_page_config(page_title="Support Router", page_icon="📨", layout="wide")
heading, language = st.columns([4, 1])
with language:
    lang = st.radio("Language / Dil", ["English", "Türkçe"], horizontal=True, key="language")

def tr(en, tur):
    return tur if lang == "Türkçe" else en

TRANSLATIONS = {
    "Orders & Shopping": "Sipariş ve Alışveriş", "Banking & Cards": "Bankacılık ve Kartlar",
    "Billing & Insurance": "Fatura ve Sigorta", "Travel & Transport": "Seyahat ve Ulaşım",
    "Reservations": "Rezervasyonlar", "Vehicle Service": "Araç Servisi",
    "HR & Payroll": "İnsan Kaynakları ve Maaş", "Food & Recipes": "Yemek ve Tarifler",
    "Device & Assistant Support": "Cihaz ve Asistan Desteği", "General Assistance": "Genel Destek",
    "Human Agent": "İnsan Temsilci", "Out-of-scope prediction": "Kapsam dışı tahmin",
    "Below confidence threshold": "Güven eşiğinin altında", "Accepted prediction": "Tahmin kabul edildi",
    "Unmapped intent": "Ekiple eşleştirilmemiş kategori", "No known words (rule)": "Bilinen kelime yok (kural)",
    "Hello! How can I help you today?": "Merhaba! Bugün size nasıl yardımcı olabilirim?",
    "You're welcome!": "Rica ederim!",
    "Goodbye! Have a nice day.": "Hoşça kalın! İyi günler.",
    "I'm a support routing assistant. Tell me what you need help with.": "Ben bir destek yönlendirme asistanıyım. Yardım istediğiniz konuyu yazın.",
    "I'm sorry you feel that way. Can you tell me what happened?": "Böyle hissetmenize üzüldüm. Ne olduğunu anlatabilir misiniz?",
    "TF-IDF + Logistic Regression": "TF-IDF + Lojistik Regresyon",
    "Word2Vec + Neural Network": "Word2Vec + Sinir Ağı",
    "Distilled Student · INT8": "Damıtılmış Öğrenci · INT8",
}
def localized(value):
    return TRANSLATIONS.get(value, value) if lang == "Türkçe" else value

with heading:
    st.title(tr("📨 Customer Support Router", "📨 Müşteri Destek Yönlendiricisi"))
st.write(tr("Enter an English message to compare intent and team predictions across four models.",
            "Dört modelin kategori ve ekip kararlarını karşılaştırmak için İngilizce bir mesaj yazın."))
st.caption(tr("Interface language only: models accept English messages. CLINC150 general-intent demo; no appliance diagnosis or real ticket submission.",
              "Dil seçimi yalnızca arayüz içindir; modeller İngilizce mesajlarla çalışır. CLINC150 genel niyet demosudur; cihaz teşhisi yapmaz veya gerçek destek talebi göndermez."))

@st.cache_resource(show_spinner=False)
def cached_model(name):
    return load_model(name)

with st.form("router"):
    message = st.text_area(tr("Customer message", "Müşteri mesajı"), value="Where is my order?",
                           max_chars=2000, height=110, key="message")
    correct = st.checkbox(tr("Correct product-name spelling (optional)", "Ürün adlarındaki yazım hatalarını düzelt (isteğe bağlı)"),
                          value=False, key="correct")
    submitted = st.form_submit_button(tr("Route message", "Talebi yönlendir"), type="primary")
st.caption(tr("Examples", "Örnekler") + ': “Where is my order?” · “I lost my credit card” · “Hello”')

if submitted:
    if not message.strip():
        st.session_state.pop("last_result", None)
        st.warning(tr("Please enter a message.", "Lütfen bir mesaj yazın."))
    else:
        text, codes, corrections = prepare_message(message, correct)
        results = {}
        with st.spinner(tr("Preparing models and predictions…", "Modeller ve tahminler hazırlanıyor…")):
            for name in MODELS:
                try:
                    results[name] = predict(name, cached_model(name), text)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    results[name] = {"error": str(error)}
        st.session_state["last_result"] = dict(text=text, codes=codes, corrections=corrections,
                                               correct=correct, results=results, reply=small_talk(text))

if "last_result" in st.session_state:
    saved = st.session_state["last_result"]
    if saved["reply"]:
        st.success(tr("Rule-based reply → ", "Sohbet kuralı → ") + localized(saved["reply"]))
        st.caption(tr("A conversation rule handled this message. Model predictions below are for comparison only.",
                      "Bu mesaj sohbet kuralıyla yanıtlandı. Aşağıdaki model kararları yalnızca karşılaştırma içindir."))
    with st.expander(tr("Processed message", "İşlenen mesaj"), expanded=bool(saved["codes"] or saved["corrections"])):
        st.text(saved["text"])
        st.write(tr("Model codes: ", "Model kodları: ") + (", ".join(saved["codes"]) or tr("None found", "Bulunamadı")))
        for old, new in saved["corrections"]:
            st.write(tr("Spelling correction: ", "Yazım düzeltmesi: ") + f"{old} → {new}")
        if saved["correct"]:
            st.caption(tr("Product correction is an application rule, excluded from the recorded test metrics.",
                          "Ürün düzeltme uygulama kuralıdır; kayıtlı test ölçümlerinde kullanılmadı."))
    for column, (name, title) in zip(st.columns(4), MODELS.items()):
        with column, st.container(border=True):
            st.subheader(localized(title))
            result = saved["results"][name]
            if "error" in result:
                st.error(tr("Model unavailable: ", "Model kullanılamıyor: ") + result["error"])
                continue
            st.metric(tr("Confidence score", "Güven skoru"),
                      "—" if result["score"] is None else f'{result["score"]:.1%}')
            st.write(tr("**Raw intent:**", "**Ham kategori:**"), result["label"])
            st.write(tr("**Suggested team:**", "**Önerilen ekip:**"), localized(result["team"]))
            st.caption(localized(result["reason"]))
            st.caption(tr("Threshold", "Eşik") + f': {result["threshold"]:.0%} · ' +
                       tr("Time", "Süre") + f': {result["milliseconds"]:.1f} ms')
    st.caption(tr("Confidence is not an accuracy guarantee. Times are per request and exclude loading. Human Agent means human review is needed.",
                  "Güven skoru doğruluk garantisi değildir. Süre tek isteğe aittir; yükleme hariçtir. İnsan Temsilci sonuçları insan incelemesi gerektirir."))

with st.expander(tr("Held-out test results", "Sabit test kümesindeki sonuçlar")):
    st.subheader(tr("Model comparison", "Model karşılaştırması"))
    rows = []
    for name, title in MODELS.items():
        path = ROOT / "results" / f"{name}_metrics.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                tr("Model", "Model"): localized(title),
                tr("Accuracy", "Doğruluk"): f'{report["test"]["accuracy"]:.2%}',
                tr("OOS recall", "Kapsam dışı yakalama"): f'{report["test"]["oos_recall"]:.2%}',
                tr("False rejection", "Yanlış ret"): f'{report["test"]["in_scope_false_rejection_rate"]:.2%}',
                tr("Size (MB)", "Boyut (MB)"): round(report["model_size_mb"], 2),
                tr("CPU mean (ms)", "CPU ortalaması (ms)"): round(report["latency_mean_ms"], 3),
            })
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(tr("Same held-out test set; thresholds chosen on validation data. CPU: one thread, 200 timed runs after 20 warmups; no loading or UI.",
                      "Aynı test kümesi; eşikler doğrulama verisiyle seçildi. CPU: tek iş parçacığı, 20 ısınma sonrası 200 ölçüm; yükleme ve arayüz hariç."))
        st.caption(tr("Size includes inference assets. Joblib models are compressed; Transformer files are uncompressed. INT8 applies to Linear layers; embeddings and LayerNorm remain FP32.",
                      "Boyut çıkarım dosyalarını kapsar. Joblib modelleri sıkıştırılmış, Transformer dosyaları sıkıştırılmamıştır. INT8, Linear katmanlarına uygulanır; gömmeler ve LayerNorm FP32 kalır."))
    else:
        st.info(tr("Test reports are not available yet.", "Test raporları henüz bulunamadı."))
