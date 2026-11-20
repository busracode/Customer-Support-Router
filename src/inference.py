"""Local CPU inference for the four-model UI."""
import json
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import joblib
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.models.neural import tokenize
from src.models.quantized import load_quantized
from src.routing import route

ROOT = Path(__file__).resolve().parents[1]
LOCK = RLock()
MODELS = {"baseline": "TF-IDF + Logistic Regression", "neural": "Word2Vec + Neural Network",
          "bert": "BERT", "student_int8": "Distilled Student · INT8"}

def load_model(name):
    with LOCK:
        torch.set_num_threads(1)
        if name in ("baseline", "neural"):
            saved = joblib.load(ROOT / "artifacts" / f"{name}.joblib")
            return saved["pipeline"], None, {"threshold": saved["threshold"]}
        directory = ROOT / ("colab/bert_model" if name == "bert" else "artifacts/student_int8")
        config = json.loads((directory / "routing_config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = (AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True,
                  use_safetensors=True).cpu().eval() if name == "bert" else load_quantized(directory))
        if [model.config.id2label[i] for i in range(len(config["labels"]))] != config["labels"]:
            raise ValueError("Model label order differs from routing configuration")
        return model, tokenizer, config

def predict(name, loaded, text):
    model, tokenizer, config = loaded
    with LOCK, threadpool_limits(limits=1), torch.inference_mode():
        start = perf_counter()
        unknown = name == "neural" and not any(t in model.vocabulary for t in tokenize(text))
        if unknown:
            label, score = "oos", None
        else:
            if tokenizer is None:
                probs = model.predict_proba([text])[0]
                labels = model.classes_
            else:
                inputs = tokenizer(text, padding="max_length", truncation=True,
                                   max_length=config["max_length"], return_tensors="pt")
                probs = model(**inputs).logits.softmax(dim=-1)[0].numpy()
                labels = config["labels"]
            best = int(probs.argmax())
            label, score = str(labels[best]), float(probs[best])
        team, reason = route(label, score, config["threshold"])
        if unknown:
            reason = "No known words (rule)"
        elapsed = (perf_counter() - start) * 1000
    return dict(label=label, score=score, team=team, reason=reason,
                threshold=config["threshold"], milliseconds=elapsed)
