# Model comparison

Four required models plus FP32 student diagnostic; same held-out test set.

| Model | Accuracy | OOS recall | False rejection | Size MB | CPU mean ms |
|---|---:|---:|---:|---:|---:|
| TF-IDF (word 1-2 grams) + Logistic Regression | 82.64% | 87.80% | 15.20% | 34.11 | 0.776 |
| Mean Word2Vec + feedforward neural network (SGD) | 75.07% | 85.90% | 24.11% | 2.90 | 0.953 |
| Fine-tuned BERT FP32 | 89.69% | 93.10% | 10.27% | 439.37 | 284.418 |
| student_fp32 | 75.91% | 82.70% | 15.20% | 45.80 | 19.555 |
| student_int8 | 75.85% | 80.30% | 12.33% | 36.10 | 20.113 |

All thresholds selected on validation only. Latency: 200 runs after 20 warmups, one CPU thread; loading/UI excluded.
Size includes inference assets. First two models use compressed joblib; Transformer assets are uncompressed.
INT8 converts Linear layers only; embeddings and LayerNorm remain FP32. FP32 student is an extra diagnostic row, not a fifth required model.