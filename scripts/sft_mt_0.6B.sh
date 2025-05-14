#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="$ROOT_DIR/cache/"
export MODELSCOPE_CACHE="$ROOT_DIR/cache/"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NPROC_PER_NODE=8 

# model
model_name=Qwen3-0.6B-Base
model_dir=$ROOT_DIR/model_card/$model_name
config_file=$ROOT_DIR/configs/ds_z0_config_bf16.json
# resume_from_checkpoint=

# data
max_lengths=1024
num_train_epochs=1

sizes=(0.5 1 5 10 30 50 100 200 500)
batch=(8 8 16 32 32 32 32 32 32)
grad_accu=(1 1 1 1 2 2 4 4 6)

for i in ${!sizes[@]}; do 

    data_size=${sizes[$i]}
    batch_size=${batch[$i]}
    gra_ac=${grad_accu[$i]}

    train_dataset=$ROOT_DIR/data/sft_train/train.${data_size}k.jsonl
    val_dataset=$ROOT_DIR/data/sft_train/valid.jsonl

    tag=sft_${data_size}k_${batch}
    output_dir=$ROOT_DIR/exps/$model_name/$tag
    mkdir -p $output_dir
    cp $0 $output_dir
    
    ####### train
    swift sft \
        --deepspeed  $config_file \
        --add_version False \
        --check_model False \
        --load_from_cache_file \
        --model $model_dir \
        --train_type full \
        --attn_impl flash_attn \
        --dataset $train_dataset \
        --split_dataset_ratio 0 \
        --val_dataset $val_dataset \
        --torch_dtype bfloat16 \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size 32 \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps $gra_ac \
        --save_strategy steps \
        --logging_strategy steps \
        --eval_strategy steps \
        --eval_steps 0.1 \
        --save_steps 0.1 \
        --logging_steps 10 \
        --max_length $max_lengths \
        --output_dir $output_dir \
        --create_checkpoint_symlink \
        --warmup_ratio 0.01 \
        --dataloader_num_workers 8 \
        --dataset_num_proc 16 \
        --seed 42  \
        --report_to tensorboard \
        --save_only_model \
        --save_total_limit 3 \
        --ddp_timeout 180000000 | tee $output_dir/train.log

    
    ####### predict
    bash inference.sh $output_dir/best

done 
