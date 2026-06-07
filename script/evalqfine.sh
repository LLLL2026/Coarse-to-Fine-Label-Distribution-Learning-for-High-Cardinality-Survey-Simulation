#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

export dataset_path="/home/dev/mvaplm1/data1/wvs"
export mode="pdt"
export objective="Score"

export base_model_path="/home/dev/models/Qwen3.5-2B"

export checkpoint_path="/home/dev/mvaplm1/output/wvs/celoss-jsd/qwen3.5-2B/Score/pdt/xi/checkpoint-1069"
export output_dir=$checkpoint_path"/eval_results/"

mkdir -p $output_dir

python mvaplm1/pipelineq1.py \
    --dataset_path $dataset_path \
    --base_name $base_model_path \
    --model_name_or_path $checkpoint_path \
    --config_name $base_model_path \
    --tokenizer_name $base_model_path \
    --output_dir $output_dir \
    --per_device_eval_batch_size 8 \
    --max_seq_length 144 \
    --do_train false \
    --do_eval false \
    --do_predict true \
    --mode $mode \
    --objective $objective \
    --use_lora true \
    --fp16 true \
    --report_to none