import json
import math
from dataclasses import dataclass
from itertools import chain
from typing import Union, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def prepare_choices_fine_grain(json_file_path, shuffle):
    content, labels, mask_list = [], [], []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if shuffle:
        import random
        random.shuffle(data)

    for item_idx, item in enumerate(tqdm(data, desc="Processing JSON data")):
        try:
            _content, _labels, _mask = prepare_line_from_json_v2(item)
        except Exception as e:
            print(f"处理第{item_idx}个样本时出错: {e}")
            print(f"样本内容: {item}")
            raise

        content.append(_content)
        labels.append(_labels)
        mask_list.append(_mask)

    return content, labels, mask_list


def prepare_line_from_json_v2(item):
    _content = []
    _labels = []
    _mask = [0] * 10

    question_text = item.get("instruction", "") + item.get("input", "")

    options = item.get("options", [])
    options_dist = item.get("options_dist", {})

    option_map = {}
    for option in options:
        if option.startswith("(") and len(option) > 2 and option[2] == ")":
            option_key = option[1]
            option_map[option_key] = option

    option_keys = sorted(list(option_map.keys()))
    valid_keys = option_keys[:10]

    if options_dist:
        total_fine_options = sum(options_dist.get(k, 0) for k in valid_keys)
    else:
        total_fine_options = len(valid_keys)

    for i in range(10):
        if i < len(valid_keys):
            key = valid_keys[i]

            text = f"question：{question_text}\nanswer：{option_map[key]}"
            _content.append(text)

            if options_dist and total_fine_options > 0:
                prob = options_dist.get(key, 0) / total_fine_options
            else:
                prob = 1.0 / len(valid_keys) if valid_keys else 0.0

            _labels.append(prob)
            _mask[i] = 1
        else:
            _content.append("")
            _labels.append(0.0)
            _mask[i] = 0

    assert len(_content) == 10

    total_prob = sum(_labels)
    assert math.isclose(total_prob, 1.0, abs_tol=1e-4) or total_prob == 0.0, f"error: {total_prob}"

    return _content, _labels, _mask


class PreferenceDataset(Dataset):
    def __init__(self, json_file_path, shuffle=False):
        self.content, self.labels, self.masks = prepare_choices_fine_grain(json_file_path, shuffle)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        return {
            "content": self.content[index],
            "labels": self.labels[index],
            "mask": self.masks[index]
        }


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    objective: Optional[str] = None

    def __call__(self, features):
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

        padded_labels = []
        for label in labels:
            if len(label) < num_choices:
                padded_label = label + [0.0] * (num_choices - len(label))
            else:
                padded_label = label[:num_choices]
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.float32)
        batch["mask"] = torch.tensor(mask_list, dtype=torch.float32)

        return batch

    def tokenize_inputs(self, batch):
        model_inputs = []

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for b_data in batch:
            inputs = {}
            content_len = len(b_data["content"])
            labels_len = len(b_data["labels"])

            if content_len != 10 or labels_len != 10:
                if content_len < 10:
                    for i in range(10 - content_len):
                        b_data["content"].append("")
                        b_data["labels"].append(0.0)
                elif content_len > 10:
                    b_data["content"] = b_data["content"][:10]
                    b_data["labels"] = b_data["labels"][:10]

            combined_texts = []
            for text in b_data["content"]:
                if text == "":
                    combined_texts.append(self.tokenizer.pad_token)
                else:
                    combined_texts.append(text)

            tokenized_examples = self.tokenizer(
                combined_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=False,
                truncation=True,
            )

            inputs["input_ids"] = tokenized_examples["input_ids"]
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
    train_dataset, val_dataset, test_dataset = None, None, None
    data_end = ".json"

    if do_train:
        train_dataset = PreferenceDataset(dataset_path + "_train" + data_end, True)
    if do_eval:
        val_dataset = PreferenceDataset(dataset_path + "_valid" + data_end, False)
    if do_predict:
        test_dataset = PreferenceDataset(dataset_path + "_test1" + data_end, False)

    return train_dataset, val_dataset, test_dataset