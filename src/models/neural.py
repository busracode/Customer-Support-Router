"""Eğitimden öğrenilmiş Word2Vec vektörlerinin ortalaması ve feedforward ağ."""

import hashlib
import re

import numpy as np

from src.preprocessing import normalize_text


def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", normalize_text(text))


def stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


class AveragedWord2VecNetwork:
    def __init__(self, vocabulary, vectors, scaler, classifier):
        self.vocabulary = vocabulary
        self.vectors = vectors
        self.scaler = scaler
        self.classifier = classifier

    @property
    def classes_(self):
        return self.classifier.classes_

    def features(self, texts):
        features = np.zeros((len(texts), self.vectors.shape[1]), dtype=np.float32)
        for index, text in enumerate(texts):
            ids = [self.vocabulary[token] for token in tokenize(text) if token in self.vocabulary]
            if ids:
                features[index] = self.vectors[ids].mean(axis=0)
        return features

    def predict_proba(self, texts):
        features = self.features(texts)
        probs = self.classifier.predict_proba(self.scaler.transform(features))
        unknown = ~np.any(features, axis=1)
        # Bilinen hiçbir kelime yoksa modelin rastgele sınıfına güvenme.
        probs[unknown] = 0
        probs[unknown, list(self.classes_).index("oos")] = 1
        return probs
