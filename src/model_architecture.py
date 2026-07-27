from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModel,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput


MODEL_NAME = "beomi/KcELECTRA-base"

NUM_BIAS_CLASSES = 4
NUM_SEVERITY_CLASSES = 3
NUM_SYCOPHANCY_CLASSES = 2


BIAS_LABELS = {
    0: "편향 없음",
    1: "약한 또는 잠재적 편향",
    2: "명확한 집단 일반화·편향",
    3: "심각한 편향",
}

SEVERITY_MODEL_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

SYCOPHANCY_LABELS = {
    0: "비동조",
    1: "동조",
}


@dataclass
class MultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    bias_logits: Optional[torch.FloatTensor] = None
    severity_logits: Optional[torch.FloatTensor] = None
    sycophancy_logits: Optional[torch.FloatTensor] = None


class KcELECTRAMultiTask(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # KcELECTRA 기본 설정과 인코더
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_config(self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # 분류 헤드
        self.bias_head = nn.Linear(
            hidden_size,
            NUM_BIAS_CLASSES,
        )

        self.severity_head = nn.Linear(
            hidden_size,
            NUM_SEVERITY_CLASSES,
        )

        self.sycophancy_head = nn.Linear(
            hidden_size,
            NUM_SYCOPHANCY_CLASSES,
        )

        # 기존 체크포인트 구조 유지
        self.register_buffer(
            "bias_weight_tensor",
            torch.ones(NUM_BIAS_CLASSES),
        )

        self.register_buffer(
            "severity_weight_tensor",
            torch.ones(NUM_SEVERITY_CLASSES),
        )

        self.bias_loss_fn = nn.CrossEntropyLoss(
            weight=self.bias_weight_tensor
        )

        self.severity_loss_fn = nn.CrossEntropyLoss(
            weight=self.severity_weight_tensor
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS 토큰 벡터 사용
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return MultiTaskOutput(
            bias_logits=self.bias_head(pooled),
            severity_logits=self.severity_head(pooled),
            sycophancy_logits=self.sycophancy_head(pooled),
        )


@dataclass
class SycophancyMultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    syc_logits: Optional[torch.FloatTensor] = None
    bias_logits: Optional[torch.FloatTensor] = None
    sev_logits: Optional[torch.FloatTensor] = None


class KcELECTRASycophancyModel(
    ElectraPreTrainedModel
):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

        # 동조, 편향, 심각도 분류 헤드
        self.syc_classifier = nn.Linear(
            config.hidden_size,
            NUM_SYCOPHANCY_CLASSES,
        )

        self.bias_classifier = nn.Linear(
            config.hidden_size,
            NUM_BIAS_CLASSES,
        )

        self.sev_classifier = nn.Linear(
            config.hidden_size,
            NUM_SEVERITY_CLASSES,
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs,
    ):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS 토큰 벡터 사용
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return SycophancyMultiTaskOutput(
            syc_logits=self.syc_classifier(pooled),
            bias_logits=self.bias_classifier(pooled),
            sev_logits=self.sev_classifier(pooled),
        )