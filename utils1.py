import json
from dataclasses import dataclass
from itertools import chain
from typing import Union, Optional, Any
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def prepare_choices_fine_grain(json_file_path, shuffle):
    content, labels, experts, intt_list, mask_list = [], [], [], [], []  # 新增mask_list
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if shuffle:
        import random
        random.shuffle(data)
    for item_idx, item in enumerate(tqdm(data, desc="Processing JSON data")):
        try:
            # 使用修改后的函数
            _content, _labels, _mask = prepare_line_from_json_v2(item)
            _experts = generate_experts_v2(item, len(_content))
        except Exception as e:
            print(f"处理第{item_idx}个样本时出错: {e}")
            print(f"样本内容: {item}")
            raise
        _intt = len(item.get("options", []))
        content.append(_content)
        labels.append(_labels)
        experts.append(_experts)
        intt_list.append(_intt)
        mask_list.append(_mask)
    return content, labels, experts, intt_list, mask_list


def prepare_line_from_json_v2(item):
    _content = []
    _labels = []
    _mask = [0] * 10

    question_text = item.get("instruction", "") + item.get("input", "")

    options = item.get("options", [])
    options_dist = item.get("options_dist", {})

    total_fine_options = sum(options_dist.values()) if options_dist else len(options)

    option_map = {}
    for option in options:
        if option.startswith("(") and len(option) > 2 and option[2] == ")":
            option_key = option[1]
            option_map[option_key] = option

    option_keys = sorted(list(option_map.keys()))
    num_fine_options = min(10, len(option_keys))

    for i in range(num_fine_options):
        key = option_keys[i]

        fine_pair = (f"{question_text}", f"{option_map[key]}")
        _content.append(fine_pair)

        if options_dist and key in options_dist and total_fine_options > 0:
            prob = options_dist[key] / total_fine_options
        else:
            prob = 1.0 / num_fine_options if num_fine_options > 0 else 0

        _labels.append(prob)
        _mask[i] = 1

    for i in range(num_fine_options, 10):
        _content.append((f"{question_text}", "<PAD>"))
        _labels.append(0)
        _mask[i] = 0

    assert len(_content) == 10
    assert len(_labels) == 10
    assert len(_mask) == 10

    return _content, _labels, _mask


def generate_experts_v2(item, content_length):
    _experts = []

    options = item.get("options", [])
    option_map = {}
    for option in options:
        if option.startswith("(") and len(option) > 2 and option[2] == ")":
            option_key = option[1]
            option_map[option_key] = option

    option_keys = sorted(list(option_map.keys()))
    num_fine_options = min(10, len(option_keys))

    for i in range(10):
        if i < num_fine_options:
            _experts.append([0])
        else:
            _experts.append([-1])
    assert len(_experts) == 10
    return _experts


def prepare_line_from_json(item):
    _content = []
    _labels = []
    question_text = item["instruction"] + item["input"]
    options_dist = item["options_dist"]
    total_options = sum(options_dist.values())
    new_options_dist = item["new_options_dist"]
    total_new_options = sum(new_options_dist.values())
    for category, prob in new_options_dist.items():
        category_text = f"{question_text}{category}"
        _content.append(category_text)
        _labels.append(prob / total_new_options)
    for option in item["options"]:
        content_text = question_text + option
        _content.append(content_text)
        option_key = option[1]
        prob = options_dist.get(option_key, 0) / total_options
        _labels.append(prob)
    return _content, _labels


def generate_experts_from_json(item):
    _experts = []
    new_options = item["new_point"]

    for i, category in enumerate(new_options.keys()):
        _experts.append([i])

    for option in item["options"]:
        option_key = option[1]
        expert_id = -1
        for i, (category, options_list) in enumerate(new_options.items()):
            if option_key in options_list:
                expert_id = i
                break
        _experts.append([expert_id])
    return _experts


class PreferenceDataset(Dataset):

    def __init__(self, json_file_path, shuffle):
        self.shuffle = shuffle
        self.content, self.labels, self.experts, self.intt, self.masks = prepare_choices_fine_grain(json_file_path,
                                                                                                    self.shuffle)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray)):
            return {
                "content": [self.content[i] for i in index],
                "labels": [self.labels[i] for i in index],
                "experts": [self.experts[i] for i in index],
                "intt": [self.intt[i] for i in index],
                "mask": [self.masks[i] for i in index]
            }
        else:
            return {
                "content": self.content[index],
                "labels": self.labels[index],
                "experts": self.experts[index],
                "intt": self.intt[index],
                "mask": self.masks[index]
            }

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase

    padding: Union[bool, str, PaddingStrategy] = True

    max_length: Optional[int] = 512

    pad_to_multiple_of: Optional[int] = None

    objective: Optional[str] = None

    def __call__(self, features):
        experts = [feature.pop("experts") for feature in features]
        intt_list = [feature.pop("intt") for feature in features]
        mask_list = [feature.pop("mask") for feature in features]

        features = self.tokenize_inputs(features)
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(len(feature['input_ids']))]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding='max_length' if self.padding else False,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        num_choices = 10
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        print(f"调试信息 - Batch shape: {batch['input_ids'].shape}")

        max_choices = 10
        padded_labels = []
        for label in labels:
            if len(label) < max_choices:
                padded_label = label + [0] * (max_choices - len(label))
            else:
                padded_label = label[:max_choices]
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.float32)
        batch["experts"] = experts
        batch["intt"] = torch.tensor(intt_list, dtype=torch.long)
        batch["mask"] = torch.tensor(mask_list, dtype=torch.float32)

        return batch

    def tokenize_inputs(self, batch):
        model_inputs = []
        for b_idx, b_data in enumerate(batch):
            inputs = {}
            content_len = len(b_data["content"])
            labels_len = len(b_data["labels"])

            if content_len != 10 or labels_len != 10:
                if content_len < 10:
                    for i in range(10 - content_len):
                        b_data["content"].append(("<PAD>", "<PAD>"))
                        b_data["labels"].append(0.0)
                elif content_len > 10:
                    b_data["content"] = b_data["content"][:10]
                    b_data["labels"] = b_data["labels"][:10]

            questions = [pair[0] for pair in b_data["content"]]
            options = [pair[1] for pair in b_data["content"]]

            tokenized_examples = self.tokenizer(
                text=questions,
                text_pair=options,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                truncation="only_first",
                return_tensors="pt",
            )

            inputs["input_ids"] = tokenized_examples["input_ids"]
            if "token_type_ids" in tokenized_examples:
                inputs["token_type_ids"] = tokenized_examples["token_type_ids"]
            inputs["attention_mask"] = tokenized_examples["attention_mask"]
            inputs["labels"] = b_data["labels"]
            model_inputs.append(inputs)
        return model_inputs


def configure_dataset(
        dataset_path=None,
        do_train=False,
        do_eval=False,
        do_predict=False,
        objective=None,
):
    "Prepare dataloaders"
    train_dataset = None
    val_dataset = None
    test_dataset = None
    if objective == "MoE":
        data_end = ".json"
    else:
        data_end = ".json"
    if do_train:
        train_dataset = PreferenceDataset(dataset_path + "_train" + data_end, True)
    if do_eval:
        val_dataset = PreferenceDataset(dataset_path + "_valid" + data_end, False)
    if do_predict:
        test_dataset = PreferenceDataset(dataset_path + "_test1" + data_end, False)
    return train_dataset, val_dataset, test_dataset


from typing import Optional, Tuple, List