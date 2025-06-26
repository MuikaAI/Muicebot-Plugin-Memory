"""
Source: https://github.com/ryuichi-sumida/LUFY/blob/main/Code/roberta/inference_large.py#
Thanks ryuichi-sumida(Ryuichi Sumida) and Graduate School of Informatics, Kyoto University
Paper: https://arxiv.org/html/2409.12524v1#S1
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from nonebot import logger
from transformers import RobertaForSequenceClassification, RobertaTokenizer

ROBERTA_BASE_MODEL_PATH = Path("./models") / "roberta-large"
MODEL_FILE = Path("./models") / "best_roberta_large.pth"

# Set the logging level to ERROR to avoid unnecessary outputs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set the environment variable to use only GPU 4
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

_device: Optional[torch.device] = None
_model: Optional[RobertaForSequenceClassification] = None
_tokenizer: Optional[RobertaTokenizer] = None


def load_model():
    global _device, _model, _tokenizer
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "best_roberta_large.pth 模型不存在！\n"
            "请先到 https://github.com/ryuichi-sumida/LUFY/blob/main/Code/roberta/best_roberta_large.pth 下载"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    _model = RobertaForSequenceClassification.from_pretrained(
        ROBERTA_BASE_MODEL_PATH, num_labels=2
    )

    # Set device to GPU if available, otherwise fallback to CPU
    logger.debug(f"device: {_device}")
    _model = _model.to(_device)  # type:ignore

    _model.load_state_dict(torch.load(MODEL_FILE, map_location=_device))
    _model.eval()

    # Load the tokenizer
    _tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_BASE_MODEL_PATH)


def emotion_prediction(text_to_predict: str):
    """
    计算用户发言的情绪强度
    """
    assert _device
    assert _model
    assert _tokenizer

    # Tokenize the input text
    encoding = _tokenizer(
        text_to_predict,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Perform a forward pass and get the prediction
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(_device)  # type:ignore
        attention_mask = encoding["attention_mask"].to(_device)  # type:ignore
        outputs = _model(input_ids, attention_mask=attention_mask)
        prediction = outputs.logits.squeeze()

    return prediction
