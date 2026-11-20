# SWE015 — Customer Support Ticket Router

An interactive Streamlit application comparing four trained models on English support messages.
The interface supports English and Turkish; changing the interface language does not translate model inputs.

## Current status
All four models are trained, evaluated and integrated: TF-IDF + Logistic Regression,
averaged Word2Vec + a feedforward network, fine-tuned BERT, and a distilled student with dynamic INT8 Linear layers.
Low-confidence and out-of-scope predictions go to Human Agent. This demo displays routing decisions; it does not create real tickets.

## Requirements
- Python 3.11 (development version: 3.11.9), 64-bit Windows with an x86 CPU.
- Internet for installation and downloading the model bundle. Inference uses local files only.
- The quantized artifact uses the PyTorch `onednn` engine. Other platforms are not yet verified.
- Four model artifacts, provided separately from source code. Retraining is not required to run the app.

## Model download
**Release link pending:** the verified model bundle must be uploaded to Google Drive or Hugging Face before submission.
There is currently no published download URL. This README alone is not yet sufficient for an external user to obtain the weights.

The author builds `release/support-router-models.zip` with:
```powershell
python scripts/build_model_bundle.py
```
After downloading the bundle, extract its contents into the project root. The final paths must be:
```text
artifacts/baseline.joblib
artifacts/neural.joblib
artifacts/student_int8/model_int8.pt
artifacts/student_int8/config.json
artifacts/student_int8/routing_config.json
artifacts/student_int8/quantization.json
colab/bert_model/model.safetensors
colab/bert_model/config.json
colab/bert_model/routing_config.json
```
Both Transformer directories also need their tokenizer files included in the bundle.
Do not place the whole bundle inside an extra nested folder.
Verify the extracted files with `python scripts/verify_model_bundle.py`.
The manifest in `docs/model_manifest.json` records SHA-256 hashes.
Load only this project's trusted model bundle: joblib files deserialize Python objects.

## Setup
Open PowerShell in the project root:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py
python scripts/verify_model_bundle.py
```
If activation is unavailable, use `.venv\Scripts\python.exe` instead of `python`.
The pinned dependencies match the environment that produced the serialized models.

## Single run command
```powershell
python -m streamlit run app/streamlit_app.py
```
Open the local URL shown in the terminal. The first submission loads the four models into memory.

## Using the application
1. Select English or Türkçe in the upper-right corner.
2. Enter an **English** message and press **Route message / Talebi yönlendir**.
3. Compare raw intent, confidence, suggested team, validation threshold and request time.
4. Expand the processed-message section to inspect normalization and extracted model codes.

Examples:
- `Where is my order?`: order-status routing.
- `I lost my credit card`: card-support routing.
- `Hello!`: a rule-based greeting; model cards remain visible for comparison.
- `My dishwahser DW-6534 is broken`: enable optional spelling correction to inspect the correction and code extraction. Appliance fault recognition is not guaranteed.
- Try an unrelated question as a difficult case. OOS detection can make mistakes.

Product correction uses minimum edit distance and a small product lexicon. It is optional and excluded from recorded classifier metrics.
Short, anchored ELIZA-style rules handle greetings, thanks, introductions and feelings.
A greeting followed by a support question is not swallowed by the greeting rule.
The Word2Vec pipeline rejects messages without known words using a rule and displays no fabricated confidence.
Confidence scores are not calibrated accuracy guarantees.

## Data and scope
Source: [CLINC150 / Larson et al.](https://github.com/clinc/oos-eval).
150 in-scope intents plus an out-of-scope class. After duplicate removal:
15,095 training, 3,100 validation and 5,500 test messages.
The original splits are preserved, with duplicate text removed in test-first order.
Source metadata and preparation counts are in `results/data_exploration.json` and `results/data_preparation.json`.

This is a general intent corpus, not an appliance-repair corpus. Team mappings are application rules, not ground-truth team annotations.
English and corpus changes are permitted by the homework; appliance-support accuracy is not claimed.

## Models and training
| Model | Training / representation |
|---|---|
| TF-IDF + Logistic Regression | Word unigrams/bigrams, training-only vocabulary and IDF, C=5 |
| Word2Vec + neural network | Training-only skip-gram embeddings, 100 dimensions, 40 epochs; mean pooling; StandardScaler; 256/128 ReLU layers; mini-batch SGD and backpropagation |
| BERT FP32 | google-bert/bert-base-uncased, 4 epochs, AdamW 2e-5, batch 16, max length 64, Colab T4 |
| Student INT8 | google/bert_uncased_L-4_H-256_A-4, 6 epochs of distillation; 0.5 CE + 0.5 T² KL, T=2; local dynamic quantization of Linear layers |

Student embeddings and LayerNorm remain FP32. BERT/student checkpoints use validation loss.
Each rejection threshold maximizes the mean of validation in-scope accuracy and OOS recall.
No test-set tuning is used. The FP32 student is an additional diagnostic for isolating quantization effects.

## Measured results
| Model | Accuracy | OOS recall | Size MB | Mean CPU ms |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 82.64% | 87.80% | 34.11 | 0.776 |
| Word2Vec + neural network | 75.07% | 85.90% | 2.90 | 0.953 |
| BERT FP32 | 89.69% | 93.10% | 439.37 | 284.418 |
| Student FP32 (diagnostic) | 75.91% | 82.70% | 45.80 | 19.555 |
| Student INT8 | 75.85% | 80.30% | 36.10 | 20.113 |

All four required models share the same held-out test hash.
Hardware recorded by the runs: Windows x86-64, Intel64 Family 6 Model 186 Stepping 2,
20 logical CPUs; one numeric thread for timing.
Latency averages 200 calls after 20 warmups, includes preprocessing/inference/rejection and excludes loading/UI.
The UI shows individual request timings, not this benchmark average.
Joblib sizes are compressed; Transformer assets are uncompressed. All sizes include inference assets.
INT8 reduced the student size but did **not** improve its measured CPU latency.
Full reports, thresholds and failure examples are under `results/`.

## Reproducing training and evaluation
Running the application needs neither training data nor retraining.
For reproduction, prepare data first:
```powershell
python scripts/explore_data.py
python scripts/prepare_data.py
python scripts/train_baseline.py
python scripts/train_neural.py
```
These training commands overwrite their respective local artifacts and result reports.

For BERT, run `python scripts/build_colab_package.py`, open `notebooks/03_bert_colab.ipynb` in Colab with a T4,
upload the generated training bundle, and follow the notebook. Extract the downloaded BERT into `colab/bert_model/`.
Run `python scripts/evaluate_bert.py` for local evaluation.

For distillation, run `python scripts/build_student_package.py`, then use `notebooks/04_student_colab.ipynb`.
The notebook requires the trained teacher in Google Drive and verifies its hash.
Extract the student into `colab/student_model/`, then run `python scripts/quantize_student.py`.
Colab dependencies are separate in `requirements-colab.txt`. Test data is excluded from training bundles.
Quantization regenerates both student reports and the complete comparison table.
The BERT-only evaluation script writes an intermediate comparison table; rerun student evaluation to rebuild the complete table.

## Checks
```powershell
python scripts/check_environment.py
python scripts/verify_model_bundle.py
python -m unittest discover -s tests
```
The full unit suite includes training-package checks and requires prepared data.
The application itself does not require the dataset.
Fresh-environment validation status is recorded in `docs/setup_validation.md`; same-machine verification is not proof of compatibility with another operating system.

## Architecture and technique map
See `docs/architecture.md` and the two-slide architecture/technique-map supplement in `deliverables/` once generated.
Runtime: local CPU and Streamlit. Transformer training: Colab GPU. Quantization and evaluation: local CPU.

## Repository and model sharing
Large weights, local environments, datasets and release ZIPs are excluded by `.gitignore`.
Publish source to GitHub/GitLab and add the repository URL to the presentation.
Publish the separate model bundle, replace the pending download notice above with its real URL,
and verify anonymous access before submission.

## AI assistance and remaining deliverables
Codex/ChatGPT assisted with implementation, debugging, documentation and presentation preparation.
The student is responsible for understanding all components and must include an AI Usage slide.
The final 8–12-slide PDF, narrated video (at most 2 minutes), and accessible repository/video/model links remain submission requirements.

## Sources
- Dataset: https://github.com/clinc/oos-eval
- Teacher: https://huggingface.co/google-bert/bert-base-uncased
- Student initialization: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
- Gensim: https://radimrehurek.com/gensim/
- scikit-learn: https://scikit-learn.org/
- Transformers: https://huggingface.co/docs/transformers/
- PyTorch: https://pytorch.org/
