#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

export dataset_path="/home/dev/mvaplm1/data/wvs"
export mode="pdt"
export objective="Score"

export base_model_path="/home/dev/models/t5-base"

export checkpoint_path="/home/dev/mvaplm1/output/wvs/celoss-jsd/t5-base/Score/pdt/22"
export output_dir=$checkpoint_path"/eval_results/"

mkdir -p $output_dir

python mvaplm1/pipelinet.py \
    --dataset_path $dataset_path \
    --base_name $base_model_path \
    --model_name_or_path $checkpoint_path \
    --config_name $base_model_path \
    --tokenizer_name $base_model_path \
    --output_dir $output_dir \
    --per_device_eval_batch_size 8 \
    --max_seq_length 144 \
    --do_predict true \
    --mode $mode \
    --objective $objective \
    --use_lora true \
    --report_to none