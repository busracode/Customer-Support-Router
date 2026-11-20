# Setup validation

Validated on 2026-11-20 in a newly created `.venv-verify` environment using Python 3.11.9 on the development Windows x86-64 computer.
This environment does not inherit the working `.venv` packages.

Results:
- CPU PyTorch 2.15.0+cpu installation: passed.
- `pip install -r requirements.txt`: passed.
- `pip check`: no broken requirements.
- `scripts/verify_model_bundle.py`: all 17 model files matched the manifest.
- `python -m unittest discover -s tests`: 9 tests passed.
- Streamlit AppTest: four real models loaded and returned results without app errors.
- Switching to Turkish retained all four result cards.

The initial dependency installation was interrupted by insufficient C: disk space. It completed after retrying with TEMP/TMP on D: and pip caching disabled.
Model loading emits PyTorch quantization deprecation warnings, but inference succeeds with the pinned version.

Limits: this verifies a separate environment on the same computer, not a second computer or operating system.
Model download from an anonymous public link remains unverified until the bundle is published.
No training or test metrics were regenerated during this installation check.
