import torch

from safetensors.torch import load_file
from transformers import AutoTokenizer

from .config import (
    BIAS_MODEL_PATH,
    BIAS_CHECKPOINT_PATH,
    SYCO_MODEL_PATH,
    DEVICE,
    validate_project_paths,
)

from .model_architecture import (
    KcELECTRAMultiTask,
    KcELECTRASycophancyModel,
)


def load_bias_model():
    # 편향·심각도 토크나이저 로드
    bias_tokenizer = AutoTokenizer.from_pretrained(
        str(BIAS_MODEL_PATH)
    )

    # 모델 구조 생성
    bias_model = KcELECTRAMultiTask().to(DEVICE)

    # 학습된 가중치 로드
    bias_state_dict = load_file(
        str(BIAS_CHECKPOINT_PATH)
    )

    bias_model.load_state_dict(
        bias_state_dict,
        strict=True,
    )

    bias_model.eval()

    return bias_model, bias_tokenizer


def load_sycophancy_model():
    # 동조 모델 토크나이저 로드
    syco_tokenizer = AutoTokenizer.from_pretrained(
        str(SYCO_MODEL_PATH)
    )

    # Hugging Face 형식으로 저장된 모델 로드
    syco_model = (
        KcELECTRASycophancyModel
        .from_pretrained(
            str(SYCO_MODEL_PATH)
        )
        .to(DEVICE)
    )

    syco_model.eval()

    return syco_model, syco_tokenizer


def load_models():
    # 모델 및 필수 파일 확인
    validate_project_paths()

    bias_model, bias_tokenizer = load_bias_model()
    syco_model, syco_tokenizer = load_sycophancy_model()

    return (
        bias_model,
        bias_tokenizer,
        syco_model,
        syco_tokenizer,
        DEVICE,
    )