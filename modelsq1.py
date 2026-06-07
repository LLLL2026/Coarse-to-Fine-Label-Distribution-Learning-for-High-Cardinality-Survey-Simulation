from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class ModelMultipleChoice(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = AutoModel.from_config(config)

        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        embed_dim = None
        possible_dim_names = ["hidden_size", "d_model", "n_embd", "dim", "hidden_dim"]

        for name in possible_dim_names:
            if hasattr(config, name):
                embed_dim = getattr(config, name)
                break

        if embed_dim is None and hasattr(config, "text_config"):
            for name in possible_dim_names:
                if hasattr(config.text_config, name):
                    embed_dim = getattr(config.text_config, name)
                    break

        if embed_dim is None:
            raise ValueError(f"无法在 config 中找到隐藏层维度参数！请检查模型配置。")

        self.classifier = nn.Linear(embed_dim, 1)
        self.post_init()

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

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
            raise ValueError("Need input_ids or inputs_embeds")

        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=target_device)
        elif mask is not None:
            mask = mask.to(target_device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None else None
        )

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]

        if attention_mask is not None:
            sequence_lengths = (
                        attention_mask * torch.arange(attention_mask.shape[-1], device=attention_mask.device)).argmax(
                dim=-1)
            batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            pooled_output = last_hidden_state[batch_indices, sequence_lengths]
        else:
            pooled_output = last_hidden_state[:, -1, :]

        pooled_output = self.dropout(pooled_output)

        pooled_output = pooled_output.to(torch.float32)
        self.classifier = self.classifier.to(torch.float32)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(batch_size, num_choices)

        if mask is not None:
            reshaped_logits = reshaped_logits.masked_fill(mask == 0, -100.0)

        loss = None
        if labels is not None:
            labels = labels.to(target_device)

            if mask is not None:
                labels = labels.masked_fill(mask == 0, 1e-4)

            labels_norm = labels / (labels.sum(dim=-1, keepdim=True) + 1e-10)
            log_probs = F.log_softmax(reshaped_logits, dim=-1)

            loss = F.kl_div(log_probs, labels_norm, reduction='batchmean')

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:] if isinstance(outputs, tuple) else (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )