export learning_rate=2e-4
export num_train_epochs=20
export seed=42
export dataset_path="/home/dev/mvaplm1/data/wvs"
export mode="pdt"
export shift_answer_epoch=2
export objective="Score"
export logging_steps=10

export model_name_or_path="/home/dev/models/t5-base"
export model_short="t5-base"
export output_dir="/home/dev/mvaplm1/output/wvs/celoss-jsd/"$model_short"/"$objective"/"$mode"/"$num_train_epochs"/"
export WANDB_DIR=$output_dir

export per_device_train_batch_size=2
export max_seq_length=144
export gradient_accumulation_steps=16
export effective_batch_size=32

export weight_decay=0.005
export save_strategy="epoch"
export evaluation_strategy="epoch"
export patience=3

export report_to="none"
export run_name="WVS_"$model_short"_"$objective"_"$mode"_"$num_train_epochs
export save_total_limit=3

export CUDA_VISIBLE_DEVICES=7
export use_lora=true
export lora_r=64
export lora_alpha=128
export lora_dropout=0.1
export lora_target_modules="q, v, wi, wo"
export data_seed=42

mkdir -p $output_dir

python mvaplm1/pipelinet.py \
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
    --overwrite_output_dir \
    --mode $mode \
    --objective $objective \
    --shift_answer_epoch $shift_answer_epoch \
    --logging_steps $logging_steps \
    --patience $patience \
    --load_best_model_at_end true \
    --evaluation_strategy $evaluation_strategy \
    --use_lora true \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_target_modules "$lora_target_modules" \
    --report_to $report_to \
    --run_name "$run_name" \
    --warmup_ratio 0.1