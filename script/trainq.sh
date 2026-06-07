#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

export learning_rate=2e-4
export num_train_epochs=20
export seed=42
export dataset_path="/home/dev/mvaplm1/data/wvs"
export mode="pdt"
export objective="Score"
export shift_answer_epoch=2
export logging_steps=10

export model_name_or_path="/home/dev/mvaplm1/models/Qwen3.5-0.8B"
export model_short="qwen"
export output_dir="/home/dev/mvaplm1/output/wvs/celoss-jsd/"$model_short"/"$objective"/"$mode"/"$num_train_epochs"/"
export WANDB_DIR=$output_dir

export per_device_train_batch_size=2
export max_seq_length=144 
export gradient_accumulation_steps=16
export weight_decay=0.005

export save_strategy="epoch"
export eval_strategy="epoch" 
export patience=3
export save_total_limit=3
export report_to="none"
export run_name="WVS_"$model_short"_"$objective"_"$mode"_"$num_train_epochs

export use_lora=true
export lora_r=32
export lora_alpha=64
export lora_dropout=0.1
export lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

mkdir -p $output_dir

python mvaplm1/pipelineq.py \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --seed $seed \
    --dataset_path $dataset_path \
    --model_name_or_path $model_name_or_path \
    --output_dir $output_dir \
    --per_device_train_batch_size $per_device_train_batch_size \
    --max_seq_length $max_seq_length \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay \
    --save_strategy $save_strategy \
    --save_total_limit $save_total_limit \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --mode $mode \
    --objective $objective \
    --shift_answer_epoch $shift_answer_epoch \
    --logging_steps $logging_steps \
    --patience $patience \
    --load_best_model_at_end true \
    --eval_strategy $eval_strategy \
    --use_lora true \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_target_modules "$lora_target_modules" \
    --report_to $report_to \
    --run_name "$run_name" \
    --fp16 false \
    --gradient_checkpointing true \
    --warmup_steps 100