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
from typing import Optional, Tuple, List
def prepare_choices_fine_grain(json_file_path, shuffle):
    """
    修改点1: 添加mask_list
    功能：为3+10结构生成mask，区分有效/无效数据
    """
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
        # 细粒度选项总数
        _intt = len(item.get("options", []))
        content.append(_content)
        labels.append(_labels)
        experts.append(_experts)
        intt_list.append(_intt)
        mask_list.append(_mask)
    return content, labels, experts, intt_list, mask_list

def prepare_line_from_json_v2(item):
    """
    完整实现3+10结构 (双段文本输入 + Prompt Injection)
    返回: content(13个tuple), labels(13个), mask(13个)
    """
    _content = []
    _labels = []
    _mask = [0] * 13  # 初始化为全0，后面有效位置设为1
    # 构建问题文本
    question_text = item.get("instruction", "") + item.get("input", "")
    # 获取数据
    options = item.get("options", [])
    options_dist = item.get("options_dist", {})
    new_point = item.get("new_point", {})
    new_options_dist = item.get("new_options_dist", {})
    # 计算总数用于概率归一化
    total_fine_options = sum(options_dist.values()) if options_dist else len(options)
    total_coarse_prob = sum(new_options_dist.values()) if new_options_dist else len(new_point)
    # === 1. 粗粒度部分 (前3位) ===
    coarse_categories = list(new_point.keys())[:3]
    for i in range(3):
        if i < len(coarse_categories):
            category = coarse_categories[i]
            # 【修改点】：保存为元组 (问题, 选项)，不要拼成一句话
            coarse_pair = (f"粗粒度：{question_text}", f"分类：{category}")
            _content.append(coarse_pair)
            # 计算粗粒度概率
            if new_options_dist and category in new_options_dist and total_coarse_prob > 0:
                prob = new_options_dist[category] / total_coarse_prob
            else:
                if total_fine_options > 0 and category in new_point:
                    category_options = new_point[category]
                    prob_sum = 0
                    for opt_key in category_options:
                        if opt_key in options_dist:
                            prob_sum += options_dist[opt_key] / total_fine_options
                    prob = prob_sum if prob_sum > 0 else 1.0 / len(coarse_categories)
                else:
                    prob = 1.0 / len(coarse_categories) if coarse_categories else 0
            _labels.append(prob)
            _mask[i] = 1
        else:
            # 【修改点】：append 传入元组，原代码 append(a,b) 会报错
            _content.append((f"粗粒度：{question_text}", "<PAD>"))
            _labels.append(0)
            _mask[i] = 0
            # === 2. 细粒度部分 (后10位) ===
    # 🌟【新增逻辑】：创建选项字母 -> 所属类别的映射字典
    option_to_category = {}
    for category, opt_list in new_point.items():
        for opt_key in opt_list:
            option_to_category[opt_key] = category
    # 创建选项映射：选项字母 -> 选项文本
    option_map = {}
    for option in options:
        if option.startswith("(") and len(option) > 2 and option[2] == ")":
            option_key = option[1]
            option_map[option_key] = option
    option_keys = sorted(list(option_map.keys()))
    num_fine_options = min(10, len(option_keys))
    for i in range(num_fine_options):
        if i < len(option_keys):
            key = option_keys[i]
            # 🌟【核心修改】：Prompt Injection，获取类别名并注入
            category_name = option_to_category.get(key, "未知类别")
            fine_pair = (f"细粒度：{question_text} [前提分类：{category_name}]", f"{option_map[key]}")
            _content.append(fine_pair)
            if options_dist and key in options_dist and total_fine_options > 0:
                prob = options_dist[key] / total_fine_options
            else:
                prob = 1.0 / num_fine_options if num_fine_options > 0 else 0
            _labels.append(prob)
            _mask[3 + i] = 1
        else:
            break
    # 填充剩余细粒度位置
    for i in range(num_fine_options, 10):
        # 【修改点】：同样改为元组
        _content.append((f"细粒度：{question_text}", "<PAD>"))
        _labels.append(0)
        _mask[3 + i] = 0
    assert len(_content) == 13
    assert len(_labels) == 13
    assert len(_mask) == 13
    return _content, _labels, _mask


def generate_experts_v2(item, content_length):
    """
    为3+10结构生成专家分配
    修改：填充位置使用-1表示不参与训练
    """
    _experts = []
    # 获取粗粒度分类
    new_point = item.get("new_point", {})
    coarse_categories = list(new_point.keys())[:3]
    # === 1. 粗粒度选项的专家分配 ===
    for i in range(3):
        if i < len(coarse_categories):
            # 有效的粗粒度类别分配专家[i]
            _experts.append([i])
        else:
            # 填充位置使用-1，表示不参与训练
            _experts.append([-1])  # 修改点：-1表示填充
    # === 2. 细粒度选项的专家分配 ===
    options = item.get("options", [])
    # 创建选项字母到类别的映射
    option_to_category = {}
    for category, opt_list in new_point.items():
        for opt_key in opt_list:
            option_to_category[opt_key] = category
    # 创建选项字母到选项文本的映射
    option_map = {}
    for option in options:
        if option.startswith("(") and len(option) > 2 and option[2] == ")":
            option_key = option[1]
            option_map[option_key] = option
    # 按字母顺序排序
    option_keys = sorted(list(option_map.keys()))
    # 为细粒度选项分配专家
    for i in range(10):
        if i < len(option_keys):
            key = option_keys[i]
            # 确定该选项属于哪个粗粒度类别
            if key in option_to_category:
                category = option_to_category[key]
                # 找到该类别在粗粒度列表中的索引
                expert_id = 0  # 默认专家0
                for j, cat in enumerate(coarse_categories):
                    if cat == category:
                        expert_id = j
                        break
                _experts.append([expert_id])
            else:
                # 如果不在任何类别中，也标记为-1（虽然这不应该发生）
                _experts.append([-1])  # 修改点：-1表示不参与训练
        else:
            # 填充位置使用-1，表示不参与训练
            _experts.append([-1])  # 修改点：-1表示填充
    return _experts


def prepare_line_from_json(item):
    """从JSON项目生成content和labels"""
    _content = []
    _labels = []
    # 构建问题文本
    question_text = item["instruction"] + item["input"]
    # 处理options_dist生成labels
    options_dist = item["options_dist"]
    total_options = sum(options_dist.values())
    # 处理new_options_dist生成labels
    new_options_dist = item["new_options_dist"]
    total_new_options = sum(new_options_dist.values())
    # 首先加入new_options的分类信息
    for category, prob in new_options_dist.items():
        category_text = f"{question_text}{category}"
        _content.append(category_text)
        # 使用new_options_dist的概率
        _labels.append(prob / total_new_options)
    # 然后按照options的顺序生成content和labels
    for option in item["options"]:
        # content: 问题 + 选项
        content_text = question_text + option
        _content.append(content_text)
        # 获取该选项的概率
        option_key = option[1]  # 提取(A)中的A
        prob = options_dist.get(option_key, 0) / total_options
        _labels.append(prob)
    return _content, _labels


def generate_experts_from_json(item):
    """根据new_options和options生成experts分配"""
    _experts = []
    # 第一类专家分配：new_options中的分类
    new_options = item["new_point"]
    # 为new_options分类信息分配专家（用专家0,1,2）
    for i, category in enumerate(new_options.keys()):
        _experts.append([i])  # 第一类=0, 第二类=1, 第三类=2
    # 为每个选项分配专家
    for option in item["options"]:
        option_key = option[1]  # 提取(A)中的A
        # 检查属于哪个new_options类别
        expert_id = -1
        for i, (category, options_list) in enumerate(new_options.items()):
            if option_key in options_list:
                expert_id = i  # 第一类=0, 第二类=1, 第三类=2
                break
        _experts.append([expert_id])
    return _experts

class PreferenceDataset(Dataset):
    """
    修改点6: 修改__init__方法，接收mask
    """
    def __init__(self, json_file_path, shuffle):
        self.shuffle = shuffle
        # 修改：返回5个值，包括mask
        self.content, self.labels, self.experts, self.intt, self.masks = prepare_choices_fine_grain(json_file_path,
                                                                                                self.shuffle)
    def __len__(self):
        return len(self.content)
    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray)):
            # 批处理
            return {
                "content": [self.content[i] for i in index],
                "labels": [self.labels[i] for i in index],
                "experts": [self.experts[i] for i in index],
                "intt": [self.intt[i] for i in index],
                "mask": [self.masks[i] for i in index]  # 新增mask
            }
        else:
            # 单个样本
            return {
                "content": self.content[index],
                "labels": self.labels[index],
                "experts": self.experts[index],
                "intt": self.intt[index],
                "mask": self.masks[index]  # 新增mask
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
        """
        修改点7: 在collate_fn中提取和添加mask
        """
        experts = [feature.pop("experts") for feature in features]
        intt_list = [feature.pop("intt") for feature in features]
        mask_list = [feature.pop("mask") for feature in features]  # 提取mask
        features = self.tokenize_inputs(features)
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)

        # 扁平化特征
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(len(feature['input_ids']))]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        # 批量填充
        batch = self.tokenizer.pad(
            flattened_features,
            padding='max_length' if self.padding else False,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # 重要修改点8: 将展平的特征还原为 [batch_size, num_choices, seq_len]
        # num_choices固定为13
        num_choices = 13
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        print(f"调试信息 - Batch shape: {batch['input_ids'].shape}")  # 应该是 [batch_size, 13, seq_len]
        # 处理labels
        max_choices = 13  # 固定为13个选项
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
        batch["mask"] = torch.tensor(mask_list, dtype=torch.float32)  # 添加mask到batch
        return batch

    def tokenize_inputs(self, batch):
        model_inputs = []
        for b_idx, b_data in enumerate(batch):
            inputs = {}
            content_len = len(b_data["content"])
            labels_len = len(b_data["labels"])
            if content_len != 13 or labels_len != 13:
                # 修复填充逻辑，也要填充成元组
                if content_len < 13:
                    for i in range(13 - content_len):
                        b_data["content"].append(("<PAD>", "<PAD>"))
                        b_data["labels"].append(0.0)
                elif content_len > 13:
                    b_data["content"] = b_data["content"][:13]
                    b_data["labels"] = b_data["labels"][:13]
            # ================== 🌟 核心修改 ==================
            # 把 tuple 列表拆解成两个独立的列表
            questions = [pair[0] for pair in b_data["content"]]
            options = [pair[1] for pair in b_data["content"]]
            # 使用 text 和 text_pair 两个参数传给 tokenizer
            tokenized_examples = self.tokenizer(
                text=questions,
                text_pair=options,  # 👈 触发双段输入，自动添加 [SEP]
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                truncation="only_first",
                return_tensors="pt",
            )
            # ==================================================
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
    data_end = ".json"
    if do_train:
        train_dataset = PreferenceDataset(dataset_path + "_train" + data_end, True)
    if do_eval:
        val_dataset = PreferenceDataset(dataset_path + "_valid" + data_end, False)
    if do_predict:
        test_dataset = PreferenceDataset(dataset_path + "_test" + data_end, False)
    return train_dataset, val_dataset, test_dataset

