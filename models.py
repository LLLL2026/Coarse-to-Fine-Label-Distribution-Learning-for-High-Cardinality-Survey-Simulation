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
import dataclasses
from typing import Optional, Tuple, List


class ModelMultipleChoice(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else config.classifier_dropout
        )
        self.pooler = ContextPooler(config)
        self.dropout = StableDropout(classifier_dropout)

        #  坚持单分类头：因为实验证明单头在当前架构下泛化效果更好
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.Tensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    #         **kwargs
    # ):
    #     mask = kwargs.get("mask", None)

    #     if input_ids is not None:
    #         batch_size = input_ids.shape[0]
    #         target_device = input_ids.device
    #     elif inputs_embeds is not None:
    #         batch_size = inputs_embeds.shape[0]
    #         target_device = inputs_embeds.device
    #     else:
    #         raise ValueError("必须提供 input_ids 或 inputs_embeds")

    #     if mask is not None and not isinstance(mask, torch.Tensor):
    #         mask = torch.tensor(mask, device=target_device)
    #     elif mask is not None:
    #         mask = mask.to(target_device)

    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    #     # 展平输入 [batch_size * 13, seq_len]
    #     input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    #     attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    #     token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    #     position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    #     inputs_embeds = (
    #         inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
    #         if inputs_embeds is not None else None
    #     )

    #     outputs = self.deberta(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
    #     pooled_output = self.pooler(last_hidden_state)
    #     pooled_output = self.dropout(pooled_output)

    #     # 单分类头打分
    #     logits = self.classifier(pooled_output)
    #     reshaped_logits = logits.view(batch_size, num_choices)

    #     # ===================================================================
    #     #  软 Mask 策略：屏蔽不存在的选项
    #     # ===================================================================
    #     if mask is not None:
    #         reshaped_logits = reshaped_logits.masked_fill(mask == 0, -100.0)

    #     loss = None
    #     if labels is not None:
    #         labels = labels.to(target_device)

    #         # ========== 1. 分割概率空间 ==========
    #         coarse_logits = reshaped_logits[:, :3]
    #         fine_logits = reshaped_logits[:, 3:]
    #         coarse_labels = labels[:, :3]
    #         fine_labels = labels[:, 3:]

    #         if mask is not None:
    #             # 标签平滑保护，防止真实标签概率为 0 导致 KL 散度里出现 log(0)
    #             coarse_labels = coarse_labels.masked_fill(mask[:, :3] == 0, 1e-4)
    #             fine_labels = fine_labels.masked_fill(mask[:, 3:] == 0, 1e-4)

    #         # ========== 2. 粗粒度独立损失 (Sums to 1) ==========
    #         coarse_labels_norm = coarse_labels / (coarse_labels.sum(dim=-1, keepdim=True) + 1e-10)
    #         coarse_log_probs = F.log_softmax(coarse_logits, dim=-1)
    #         loss_coarse = F.kl_div(coarse_log_probs, coarse_labels_norm, reduction='batchmean')

    #         # ========== 3. 细粒度独立损失 (Sums to 1) ==========
    #         fine_labels_norm = fine_labels / (fine_labels.sum(dim=-1, keepdim=True) + 1e-10)
    #         fine_log_probs = F.log_softmax(fine_logits, dim=-1)
    #         loss_fine = F.kl_div(fine_log_probs, fine_labels_norm, reduction='batchmean')

    #         loss = 0.3 * loss_coarse + 0.7 * loss_fine

    #     if not return_dict:
    #         output = (reshaped_logits,) + outputs[1:] if isinstance(outputs, tuple) else (reshaped_logits,)
    #         return ((loss,) + output) if loss is not None else output

    #     return MultipleChoiceModelOutput(
    #         loss=loss,
    #         logits=reshaped_logits,
    #         hidden_states=outputs.hidden_states if return_dict else None,
    #         attentions=outputs.attentions if return_dict else None,
    #     )

    # JSD

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
        reshaped_logits = logits.view(batch_size, num_choices)
        # ===================================================================
        #  软 Mask 策略：保留，防止概率变绝对 0
        # ===================================================================
        if mask is not None:
            reshaped_logits = reshaped_logits.masked_fill(mask == 0, -100.0)

        #  FP16 绝对安全的 JSD 损失函数
        def compute_jsd_loss(p, q):
            eps = 1e-5  # 安全截断，绝不允许 log(0)
            p_safe = torch.clamp(p, min=eps, max=1.0)
            q_safe = torch.clamp(q, min=eps, max=1.0)
            m = 0.5 * (p_safe + q_safe)
            # 手工展开算 KL 散度，防止 PyTorch 原生 kl_div 在部分场景下报错
            kl_pm = torch.sum(p_safe * (torch.log(p_safe) - torch.log(m)), dim=-1)
            kl_qm = torch.sum(q_safe * (torch.log(q_safe) - torch.log(m)), dim=-1)
            return (0.5 * kl_pm + 0.5 * kl_qm).mean()

        loss = None
        if labels is not None:
            labels = labels.to(target_device)
            # ========== 1. 分割概率空间 ==========
            coarse_logits = reshaped_logits[:, :3]
            fine_logits = reshaped_logits[:, 3:]
            coarse_labels = labels[:, :3]
            fine_labels = labels[:, 3:]
            if mask is not None:
                # 标签平滑保护，保留 1e-4
                coarse_labels = coarse_labels.masked_fill(mask[:, :3] == 0, 1e-4)
                fine_labels = fine_labels.masked_fill(mask[:, 3:] == 0, 1e-4)
            # ========== 2. 粗粒度 JSD 损失 ==========
            coarse_labels_norm = coarse_labels / (coarse_labels.sum(dim=-1, keepdim=True) + 1e-10)
            coarse_probs = F.softmax(coarse_logits, dim=-1)  # JSD 需要输入实际概率 P
            loss_coarse = compute_jsd_loss(coarse_probs, coarse_labels_norm)
            # ========== 3. 细粒度 JSD 损失 ==========
            fine_labels_norm = fine_labels / (fine_labels.sum(dim=-1, keepdim=True) + 1e-10)
            fine_probs = F.softmax(fine_logits, dim=-1)  # JSD 需要输入实际概率 P
            loss_fine = compute_jsd_loss(fine_probs, fine_labels_norm)
            # ========== 4. 多任务联合优化 ==========
            loss = 0.0 * loss_coarse + 1.0 * loss_fine
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:] if isinstance(outputs, tuple) else (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

    # CROSS
    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.Tensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    #         **kwargs
    # ):
    #     mask = kwargs.get("mask", None)
    #     if input_ids is not None:
    #         batch_size = input_ids.shape[0]
    #         target_device = input_ids.device
    #     elif inputs_embeds is not None:
    #         batch_size = inputs_embeds.shape[0]
    #         target_device = inputs_embeds.device
    #     else:
    #         raise ValueError("必须提供 input_ids 或 inputs_embeds")
    #     if mask is not None and not isinstance(mask, torch.Tensor):
    #         mask = torch.tensor(mask, device=target_device)
    #     elif mask is not None:
    #         mask = mask.to(target_device)
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    #     input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    #     attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    #     token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    #     position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    #     inputs_embeds = (
    #         inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
    #         if inputs_embeds is not None else None
    #     )
    #     outputs = self.deberta(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
    #     pooled_output = self.pooler(last_hidden_state)
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     reshaped_logits = logits.view(batch_size, num_choices)
    #     # ===================================================================
    #     #  软 Mask 策略：用 -100.0 代替
    #     # 防止无效选项参与 Softmax 概率分配
    #     # ===================================================================
    #     if mask is not None:
    #         reshaped_logits = reshaped_logits.masked_fill(mask == 0, -100.0)
    #     loss = None
    #     if labels is not None:
    #         labels = labels.to(target_device)
    #         # ========== 1. 分割概率空间 ==========
    #         coarse_logits = reshaped_logits[:, :3]
    #         fine_logits = reshaped_logits[:, 3:]
    #         coarse_labels = labels[:, :3]
    #         fine_labels = labels[:, 3:]
    #         if mask is not None:
    #             # 交叉熵允许真实标签为 0，所以无效位置直接清零，不参与损失计算
    #             coarse_labels = coarse_labels.masked_fill(mask[:, :3] == 0, 1e-4)
    #             fine_labels = fine_labels.masked_fill(mask[:, 3:] == 0, 1e-4)
    #         # ========== 2. 粗粒度交叉熵损失 ==========
    #         # 将标签归一化为和为 1 的合法概率分布
    #         coarse_labels_norm = coarse_labels / (coarse_labels.sum(dim=-1, keepdim=True) + 1e-10)
    #         # 直接使用 F.cross_entropy (它内部会自动对 logits 做 log_softmax)
    #         loss_coarse = F.cross_entropy(coarse_logits, coarse_labels_norm, reduction='mean')
    #         # ========== 3. 细粒度交叉熵损失 ==========
    #         fine_labels_norm = fine_labels / (fine_labels.sum(dim=-1, keepdim=True) + 1e-10)
    #         loss_fine = F.cross_entropy(fine_logits, fine_labels_norm, reduction='mean')
    #         # ========== 4. 多任务联合优化 ==========
    #         loss = 0.3 * loss_coarse + 0.7 * loss_fine
    #     if not return_dict:
    #         output = (reshaped_logits,) + outputs[1:] if isinstance(outputs, tuple) else (reshaped_logits,)
    #         return ((loss,) + output) if loss is not None else output
    #     return MultipleChoiceModelOutput(
    #         loss=loss,
    #         logits=reshaped_logits,
    #         hidden_states=outputs.hidden_states if return_dict else None,
    #         attentions=outputs.attentions if return_dict else None,
    #     )