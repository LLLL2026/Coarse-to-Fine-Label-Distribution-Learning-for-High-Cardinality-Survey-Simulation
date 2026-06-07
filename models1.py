import dataclasses
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, PreTrainedModel, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, ContextPooler

class ModelMultipleChoice(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else config.classifier_dropout
        )
        self.pooler = ContextPooler(config)
        self.dropout = StableDropout(classifier_dropout)
        # 单分类头：直接对传入的选项做评分
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        mask = kwargs.get("mask", None)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            target_device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            target_device = inputs_embeds.device
        else:
            raise ValueError("必须提供 input_ids 或 inputs_embeds")

        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=target_device)
        elif mask is not None:
            mask = mask.to(target_device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 此时传进来的 num_choices 应该已经是 10
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None else None
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 重塑为 [batch_size, 10]
        reshaped_logits = logits.view(batch_size, num_choices)

        # ===================================================================
        # 软 Mask 策略：过滤掉 <PAD> 的无效选项
        # ===================================================================
        if mask is not None:
            reshaped_logits = reshaped_logits.masked_fill(mask == 0, -100.0)

        loss = None
        if labels is not None:
            labels = labels.to(target_device)

            # ========== 最原始 Baseline 损失计算 ==========
            # 1. 标签平滑与 Mask 保护
            if mask is not None:
                labels = labels.masked_fill(mask == 0, 1e-4)

            # 2. 归一化为标准的 10 维概率分布 (Sums to 1)
            labels_norm = labels / (labels.sum(dim=-1, keepdim=True) + 1e-10)

            # 3. 计算 10 维空间内的竞争概率
            log_probs = F.log_softmax(reshaped_logits, dim=-1)

            # 4. 直接计算单一的 KL 散度损失
            loss = F.kl_div(log_probs, labels_norm, reduction='batchmean')

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:] if isinstance(outputs, tuple) else (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits, # 10 维 Logits 原样返回
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )
