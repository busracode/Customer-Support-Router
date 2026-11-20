"""CPU için dinamik INT8 Linear katmanları; embedding/LayerNorm FP32 kalır."""
import json
from pathlib import Path

import torch
from torch.ao.quantization import quantize_dynamic
from transformers import AutoConfig, AutoModelForSequenceClassification


def quantize_model(model):
    return quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8, inplace=False)


def load_quantized(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "quantization.json").read_text())
    if metadata["engine"] not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Bu sistem kayıtlı quantization motorunu desteklemiyor.")
    torch.backends.quantized.engine = metadata["engine"]
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    model = quantize_model(AutoModelForSequenceClassification.from_config(config))
    state = torch.load(directory / "model_int8.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
