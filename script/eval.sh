#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

export dataset_path="/home/dev/mvaplm1/data/wvs"
export mode="pdt"
export objective="Score"

export base_model_path="/home/dev/models/deberta-v3-base"

export checkpoint_path="/home/dev/mvaplm2/output/wvs/celoss-jsd/deberta-base/Score/pdt/50"   
export output_dir=$checkpoint_path"/eval_results/"

mkdir -p $output_dir

python mvaplm2/pipeline.py \
    --model_name_or_path $checkpoint_path \
    --dataset_path $dataset_path \
    --base_name $base_model_path \
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
    --report_to none