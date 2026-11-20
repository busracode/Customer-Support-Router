"""Temel bağımlılıkların kullanılabilir olduğunu kontrol eder."""

import importlib
import sys

PACKAGES = ("pandas", "numpy", "sklearn", "datasets", "matplotlib", "streamlit", "joblib", "gensim", "torch", "transformers", "safetensors")


def main():
    print(f"Python: {sys.version.split()[0]}")
    missing = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            print(f"OK: {package} {getattr(module, '__version__', '')}")
        except ImportError as error:
            missing.append(package)
            print(f"HATA: {package}: {error}")
    if missing:
        print("Follow the CPU PyTorch and requirements installation commands in README.md.")
        return 1
    print("Kurulum başarılı!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
