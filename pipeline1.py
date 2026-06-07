import scipy.special
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import logging
import os
import pickle
import sys
import scipy
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from scipy.spatial import distance
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed, EarlyStoppingCallback,
)
from transformers.pipelines.text_classification import softmax
from arguments import ModelArguments, DataTrainingArguments
from modelst1 import ModelMultipleChoice, Model, MoEMultipleChoiceWithAutoRouter
from pdt_trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from utilst1 import configure_dataset, DataCollatorForMultipleChoice
from peft import LoraConfig
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel, PeftConfig
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


def is_peft_available():
    try:
        import peft
        return True
    except ImportError:
        return False


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
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
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
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
    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
    if training_args.do_train:
        use_lora = getattr(data_args, 'use_lora', False) or getattr(model_args, 'use_lora', False)
        if data_args.objective == "Score":
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
                model.requires_grad_(False)
            else:
                model.requires_grad_(False)
                for name, param in model.named_parameters():
                    if "classifier" in name or "pooler" in name:
                        param.requires_grad = True
        elif data_args.objective == "Team":
            model = Model(
                name=mname,
                num_choices=13,
                config=config,
            )
            if use_lora:
                model.requires_grad_(False)
            else:
                model.requires_grad_(False)
                model.model.deberta.requires_grad_(True)
        else:
            model = MoEMultipleChoiceWithAutoRouter(
                config=config,
                num_expert=3,
                weight_max_size=4,
                num_choices=13
            )
            if use_lora:
                model.requires_grad_(False)
            else:
                model.requires_grad_(False)
                model.sparse_moe.requires_grad_(True)

    else:
        base_model_path = model_args.base_name if model_args.base_name else model_args.model_name_or_path
        model = ModelMultipleChoice.from_pretrained(
            base_model_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        )
        model.forward = lambda *args, **kwargs: ModelMultipleChoice.forward(model, *args, **kwargs)

        use_lora = getattr(data_args, 'use_lora', False) or getattr(model_args, 'use_lora', False)
        if use_lora:
            logger.info(f"正在将 LoRA 权重加载到基座模型上，来源: {model_args.model_name_or_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
        model.to(training_args.device)
        model.eval()
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024

    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )

        else:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is shorter than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={data_args.max_seq_length}."
            )

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

    # Data collator
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

    import scipy.special

    def compute_metrics(eval_predictions):
        import scipy.special
        import scipy.stats
        from scipy.spatial import distance
        from scipy.stats import wasserstein_distance

        predictions, label_ids = eval_predictions
        kl_list, jsd_list, emd_list, emd_s_list = [], [], [], []
        eps = 1e-10

        for idx, (pred, label) in enumerate(zip(predictions, label_ids)):
            mask = pred > -50.0

            if np.sum(mask) == 0:
                kl_list.append(np.nan)
                jsd_list.append(np.nan)
                emd_list.append(np.nan)
                emd_s_list.append(np.nan)
                continue

            pred_logits = pred[mask]
            pred_probs = scipy.special.softmax(pred_logits)

            label_raw = label[mask]
            label_sum = np.sum(label_raw)
            label_probs = label_raw / label_sum if label_sum > eps else np.ones_like(label_raw) / len(label_raw)

            pred_safe = np.clip(pred_probs, eps, 1.0)
            label_safe = np.clip(label_probs, eps, 1.0)

            kl = scipy.stats.entropy(pred_safe, label_safe)
            jsd = distance.jensenshannon(pred_probs, label_probs)
            emd = wasserstein_distance(pred_probs, label_probs)

            absolute_indices = np.where(mask)[0]
            emd_s = wasserstein_distance(
                u_values=absolute_indices,
                v_values=absolute_indices,
                u_weights=pred_probs,
                v_weights=label_probs
            )

            kl_list.append(kl)
            jsd_list.append(jsd)
            emd_list.append(emd)
            emd_s_list.append(emd_s)

        metrics = {
            "mean_kl_divergence": float(np.nanmean(kl_list)),
            "mean_jsd": float(np.nanmean(jsd_list)),
            "mean_jsd_score": float(1.0 - np.nanmean(jsd_list)),
            "mean_emd": float(np.nanmean(emd_list)),
            "mean_emd_s": float(np.nanmean(emd_s_list)),
        }
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
        # Initialize our Trainer

    # Initialize our Trainer
    if getattr(data_args, 'patience', None) is not None:
        earlystop = EarlyStoppingCallback(early_stopping_patience=data_args.patience)
    use_lora = getattr(data_args, 'use_lora', False)
    if use_lora and training_args.do_train:
        if not is_peft_available:
            raise ImportError("pip install peft")
        logger.info("=" * 50)
        logger.info("Check models:")
        linear_layers = set()
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                layer_name = name.split('.')[-1]
                linear_layers.add(layer_name)
                if any(x in name.lower() for x in ['query', 'key', 'value', 'attention', 'q_', 'k_', 'v_']):
                    logger.info(f"  {name} -> {layer_name}")
        logger.info(f"All type: {sorted(linear_layers)}")
        logger.info("=" * 50)
        if getattr(data_args, 'lora_target_modules', None):
            target_modules = data_args.lora_target_modules.split(",")
            target_modules = [m.strip() for m in target_modules if m.strip()]
        else:
            if "t5" in mname.lower():
                target_modules = ["q", "v", "wi", "wo"]
            else:
                target_modules = ["query_proj", "key_proj", "value_proj", "intermediate_dense", "output_dense"]
        from peft import LoraConfig, get_peft_model, TaskType
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
            if any(k in name for k in ["classifier", "experts", "pooler"]):
                param.requires_grad = True

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

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        running_set = "validation"
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict:
        running_set = "test"
        logger.info("*** Test ***")
        metrics = trainer.predict(test_dataset)
        trainer.log_metrics("test", metrics.metrics)
        trainer.save_metrics("test", metrics.metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()