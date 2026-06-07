import logging
import os
import sys
import pickle
from pathlib import Path

import scipy
import scipy.special
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

import datasets
import numpy as np
import pandas as pd
import torch
import transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.pipelines.text_classification import softmax
from transformers.trainer_utils import get_last_checkpoint

from arguments import ModelArguments, DataTrainingArguments
from models import ModelMultipleChoice
from pdt_trainer import Trainer
from utils import configure_dataset, DataCollatorForMultipleChoice

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig


def is_peft_available():
    try:
        import peft
        return True
    except ImportError:
        return False


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    if data_args.dataset_path is not None:
        train_dataset, val_dataset, test_dataset = configure_dataset(
            data_args.dataset_path,
            training_args.do_train,
            training_args.do_eval,
            training_args.do_predict,
            data_args.objective
        )
    else:
        train_dataset, val_dataset, test_dataset = None, None, None

    mname = model_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else mname,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else mname,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # ---------------- 仅保留 Score 模式下的模型加载 ----------------
    if training_args.do_train:
        use_lora = getattr(data_args, 'use_lora', False) or getattr(model_args, 'use_lora', False)

        model = ModelMultipleChoice.from_pretrained(
            mname,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.forward = lambda *args, **kwargs: ModelMultipleChoice.forward(model, *args, **kwargs)

        if use_lora:
            # LoRA模式下，冻结基础参数，稍后添加LoRA适配器
            model.requires_grad_(False)
        else:
            logger.info("正在进入 Linear Probing 模式：冻结 Backbone，仅训练分类头")
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                if "classifier" in name or "pooler" in name:
                    param.requires_grad = True
                    logger.info(f"已解冻用于训练的参数: {name}")

    else:
        # 推理或测试时的加载
        logger.info("初始化推理模式模型...")
        use_lora = getattr(data_args, 'use_lora', False) or getattr(model_args, 'use_lora', False)

        # 1. 首先加载基础 Backbone 模型
        base_model_path = model_args.base_name if getattr(model_args, 'base_name',
                                                          None) else model_args.model_name_or_path
        model = ModelMultipleChoice.from_pretrained(
            base_model_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # 绑定 forward 函数
        model.forward = lambda *args, **kwargs: ModelMultipleChoice.forward(model, *args, **kwargs)

        # 2. 如果训练时使用了 LoRA，这里必须加载 LoRA 权重
        if use_lora:
            if not is_peft_available():
                raise ImportError("推理 LoRA 模型需要安装 peft 库：pip install peft")

            # 这里的 model_args.model_name_or_path 应该是你训练保存的 checkpoint 路径
            logger.info(f"正在加载 LoRA 权重: {model_args.model_name_or_path}")
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path)

            # 推理模式下，推荐合并权重以提升推理速度（可选）
            # model = model.merge_and_unload()

        model.eval()  # 确保进入评估模式
    # -------------------------------------------------------------

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead."
            )
            max_seq_length = 1024
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if val_dataset is None:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = val_dataset
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if test_dataset is None:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.max_eval_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_eval_samples))

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(
            tokenizer=tokenizer,
            max_length=max_seq_length,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    global running_set
    global dataset
    global path
    global do_test
    do_test = training_args.do_predict
    running_set = "validation"
    dataset = data_args.dataset_path.split("/")[1]
    path = training_args.output_dir + "/val_results/"
    Path(path).mkdir(parents=True, exist_ok=True)

    def compute_alpha_jsd(pred, label, m, alpha, standard_jsd, eps=1e-10):
        try:
            m_safe = np.clip(m, eps, 1.0)
            m_alpha = m_safe ** alpha
            m_alpha_norm = m_alpha / np.sum(m_alpha)
            m_alpha_norm = np.clip(m_alpha_norm, eps, 1.0)

            pred_safe = np.clip(pred, eps, 1.0)
            label_safe = np.clip(label, eps, 1.0)

            coef = 1.0 / (2.0 * alpha)
            kl_pred = scipy.stats.entropy(pred_safe, m_alpha_norm)
            kl_label = scipy.stats.entropy(label_safe, m_alpha_norm)
            val = coef * (kl_pred + kl_label)

            if np.isinf(val) or np.isnan(val):
                return standard_jsd
            return float(val)
        except Exception as e:
            return standard_jsd

    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        kl_combined_list, kl_coarse_list, kl_fine_list = [], [], []
        jsd_combined_list, jsd_coarse_list, jsd_fine_list = [], [], []
        jsd_score_combined_list, jsd_score_coarse_list, jsd_score_fine_list = [], [], []
        emd_combined_list, emd_coarse_list, emd_fine_list = [], [], []
        emd_coarse_s_list, emd_fine_s_list = [], []
        eps = 1e-10

        for idx, (pred, label) in enumerate(zip(predictions, label_ids)):
            coarse_mask = pred[:3] > -50.0
            fine_mask = pred[3:] > -50.0

            if np.sum(coarse_mask) == 0:
                kl_combined_list.append(np.nan);
                kl_coarse_list.append(np.nan);
                kl_fine_list.append(np.nan)
                jsd_combined_list.append(np.nan);
                jsd_coarse_list.append(np.nan);
                jsd_fine_list.append(np.nan)
                jsd_score_combined_list.append(np.nan);
                jsd_score_coarse_list.append(np.nan);
                jsd_score_fine_list.append(np.nan)
                emd_combined_list.append(np.nan);
                emd_coarse_list.append(np.nan);
                emd_fine_list.append(np.nan)
                emd_coarse_s_list.append(np.nan);
                emd_fine_s_list.append(np.nan)
                continue

            # ========== 1. 粗粒度部分 (前3个) ==========
            coarse_pred_logits = pred[:3][coarse_mask]
            coarse_pred = scipy.special.softmax(coarse_pred_logits)
            coarse_label_raw = label[:3][coarse_mask]
            coarse_label_sum = np.sum(coarse_label_raw)
            coarse_label = coarse_label_raw / coarse_label_sum if coarse_label_sum > eps else np.ones_like(
                coarse_label_raw) / len(coarse_label_raw)

            coarse_pred_safe = np.clip(coarse_pred, eps, 1.0)
            coarse_label_safe = np.clip(coarse_label, eps, 1.0)

            kl_coarse = scipy.stats.entropy(coarse_pred_safe, coarse_label_safe)
            jsd_coarse = distance.jensenshannon(coarse_pred, coarse_label)

            coarse_absolute_indices = np.where(coarse_mask)[0]
            emd_coarse_s = wasserstein_distance(
                u_values=coarse_absolute_indices,
                v_values=coarse_absolute_indices,
                u_weights=coarse_pred,
                v_weights=coarse_label
            )
            emd_coarse = wasserstein_distance(coarse_pred, coarse_label)

            kl_coarse_list.append(kl_coarse)
            jsd_coarse_list.append(jsd_coarse)
            jsd_score_coarse_list.append(1.0 - jsd_coarse)
            emd_coarse_s_list.append(emd_coarse_s)
            emd_coarse_list.append(emd_coarse)

            # ========== 2. 细粒度部分 (后10个) ==========
            if np.sum(fine_mask) > 0:
                fine_pred_logits = pred[3:][fine_mask]
                fine_pred = scipy.special.softmax(fine_pred_logits)
                fine_label_raw = label[3:][fine_mask]
                fine_label_sum = np.sum(fine_label_raw)
                fine_label = fine_label_raw / fine_label_sum if fine_label_sum > eps else np.ones_like(
                    fine_label_raw) / len(fine_label_raw)

                fine_pred_safe = np.clip(fine_pred, eps, 1.0)
                fine_label_safe = np.clip(fine_label, eps, 1.0)

                kl_fine = scipy.stats.entropy(fine_pred_safe, fine_label_safe)
                jsd_fine = distance.jensenshannon(fine_pred, fine_label)

                fine_absolute_indices = np.where(fine_mask)[0]
                emd_fine_s = wasserstein_distance(
                    u_values=fine_absolute_indices,
                    v_values=fine_absolute_indices,
                    u_weights=fine_pred,
                    v_weights=fine_label
                )
                emd_fine = wasserstein_distance(fine_pred, fine_label)

                kl_fine_list.append(kl_fine)
                jsd_fine_list.append(jsd_fine)
                jsd_score_fine_list.append(1.0 - jsd_fine)
                emd_fine_s_list.append(emd_fine_s)
                emd_fine_list.append(emd_fine)

                kl_combined_list.append((kl_coarse + kl_fine) / 2.0)
                jsd_combined_list.append((jsd_coarse + jsd_fine) / 2.0)
                jsd_score_combined_list.append(1.0 - (jsd_coarse + jsd_fine) / 2.0)
                emd_combined_list.append((emd_coarse + emd_fine) / 2.0)
            else:
                kl_fine_list.append(np.nan);
                jsd_fine_list.append(np.nan);
                jsd_score_fine_list.append(np.nan)
                emd_fine_list.append(np.nan);
                emd_fine_s_list.append(np.nan)

                kl_combined_list.append(kl_coarse)
                jsd_combined_list.append(jsd_coarse)
                jsd_score_combined_list.append(1.0 - jsd_coarse)
                emd_combined_list.append(emd_coarse)

        kl_data = pd.DataFrame({
            "KL-Divergence": kl_combined_list, "KL-Divergence_Coarse": kl_coarse_list,
            "KL-Divergence_Fine": kl_fine_list,
            "JSD": jsd_combined_list, "JSD_Coarse": jsd_coarse_list, "JSD_Fine": jsd_fine_list,
            "JSD_Score": jsd_score_combined_list, "JSD_Score_Coarse": jsd_score_coarse_list,
            "JSD_Score_Fine": jsd_score_fine_list,
            "EMD": emd_combined_list, "EMD_Coarse_S": emd_coarse_s_list, "EMD_Fine_S": emd_fine_s_list,
            "EMD_Coarse": emd_coarse_list, "EMD_Fine": emd_fine_list,
        })

        metrics = {f"mean_{col.lower().replace('-', '_')}": float(np.nanmean(kl_data[col])) for col in kl_data.columns}
        return metrics

    def compute_metrics_acc(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        labels = np.argmax(label_ids, axis=1)
        acc = (preds == labels).astype(np.float32).mean().item()

        if training_args.do_eval:
            with open(path + data_args.dataset_path.split("/")[-1] + "_val.json", "w") as f:
                f.write(f"Mean accuracy: {acc}")
        if training_args.do_predict:
            with open(path + data_args.dataset_path.split("/")[-1] + "_test.json", "w") as f:
                f.write(f"Mean accuracy: {acc}")
        return {"accuracy": acc}

    if getattr(data_args, 'patience', None) is not None:
        earlystop = EarlyStoppingCallback(early_stopping_patience=data_args.patience)

    use_lora = getattr(data_args, 'use_lora', False)

    if use_lora and training_args.do_train:
        if not is_peft_available:
            raise ImportError("LoRA训练需要安装peft库：pip install peft")

        logger.info("=" * 50)
        logger.info("检查模型中的所有线性层模块:")
        linear_layers = set()
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                layer_name = name.split('.')[-1]
                linear_layers.add(layer_name)
                if any(x in name.lower() for x in ['query', 'key', 'value', 'attention', 'q_', 'k_', 'v_']):
                    logger.info(f"  注意力相关模块: {name} -> {layer_name}")
        logger.info(f"所有线性层类型: {sorted(linear_layers)}")
        logger.info("=" * 50)

        if getattr(data_args, 'lora_target_modules', None):
            target_modules = data_args.lora_target_modules.split(",")
            target_modules = [m.strip() for m in target_modules if m.strip()]
        else:
            if "t5" in mname.lower():
                target_modules = ["q", "v", "wi", "wo"]
            else:
                target_modules = ["query_proj", "key_proj", "value_proj", "intermediate_dense", "output_dense"]

        peft_config = LoraConfig(
            r=data_args.lora_r,
            lora_alpha=data_args.lora_alpha,
            lora_dropout=data_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["classifier", "pooler"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        for name, param in model.named_parameters():
            if any(k in name for k in ["classifier", "pooler"]):
                param.requires_grad = True
        logger.info("已手动解冻自定义层 (classifier/pooler)")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA启用: {trainable_params:,} 个可训练参数 / {total_params:,} 总参数")

    training_args.label_names = ["labels"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "auce" in data_args.dataset_path or "wvs" in data_args.dataset_path else compute_metrics_acc,
        shift_answer_epoch=data_args.shift_answer_epoch,
        callbacks=[earlystop] if data_args.patience is not None else None,
        use_lora=use_lora,
        save_full_model=getattr(data_args, 'save_full_model', False),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        running_set = "validation"
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        running_set = "test"
        logger.info("*** Test ***")
        metrics = trainer.predict(test_dataset)
        trainer.log_metrics("test", metrics.metrics)
        trainer.save_metrics("test", metrics.metrics)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()