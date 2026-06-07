from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_name: str = field(
        default=None, metadata={
            "help": "Base model directory for evaluation."
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    n_choices: Optional[int] = field(
        default=4, metadata={"help": "number of choices."}
    )
    mode: Optional[str] = field(
        default="one-hot", metadata={
            "help": "label type - [one-hot, pdt]"
        }
    )
    objective: Optional[str] = field(
        default="Score", metadata={
            "help": "training method - [Score, Team]"
        }
    )
    shift_answer_epoch: Optional[int] = field(
        default=1, metadata={
            "help": "shifting choices with one-hot mode"
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of waits for early stopping"
        }
    )

    # 🌟 新增 LoRA 相关参数定义，适配 lora.sh 脚本
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension (rank)."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for LoRA layers."}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Target modules for LoRA (e.g. 'query_proj,key_proj')."}
    )


@dataclass
class ReasoningArguments(TrainingArguments):
    value_model_name: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    generative_model_name: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The directory of data."}
    )
    experiment: Optional[str] = field(
        default=None, metadata={"help": "The experiment type ['Value', 'Debias']"}
    )
    reasoning_mode: Optional[str] = field(
        default=None, metadata={"help": "Reasoning mode ['Random', 'Preferences']"}
    )
    language: Optional[str] = field(
        default=None, metadata={"help": "Language"}
    )
    diverse_plan: Optional[str] = field(
        default=None, metadata={
            "help": "diverse_plan in AECE evaluation, "
                    "['prompt', 'config', 'pc', 'all'] memory can be altered bt max_memory in generation kwargs"
        }
    )
    generate_kwargs_file: Optional[str] = field(
        default=None, metadata={"help": "generative model kwargs"}
    )
    case_num: Optional[int] = field(
        default=10, metadata={"help": "investigate case number"}
    )