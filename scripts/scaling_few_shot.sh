#! /bin/bash

set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))
export HF_HOME="$ROOT_DIR/cache"
export HF_DATASETS_CACHE="$ROOT_DIR/cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

comet_model=$ROOT_DIR/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=$ROOT_DIR/model_card/XCOMET-XXL/checkpoints/model.ckpt

### model
model_name=${1:-"Qwen3-8B-Base"}
model_dir=$ROOT_DIR/model_card/$model_name
config_file=$ROOT_DIR/configs/accelerate_config.yaml


### data

for shot in 3 8 16 32 64 128; do 

    src_file_strs=""
    ref_file_strs=""
    hypo_file_strs=""
    lang_pair_strs=""

    # for lang in de ru zh cs vi ko ne sw ha;do
    # for src in $lang en;do
    for lang in de vi ne;do
    for src in en;do
        if [ $src = $lang ]; then
            src_lang=$lang
            tgt_lang=en
        else 
            src_lang=en 
            tgt_lang=$lang
        fi

        lang_pair=${src_lang}-${tgt_lang}
        lp=${src_lang}2${tgt_lang}

        
        test_file=$ROOT_DIR/data/flores200/en-$lang/test.en-$lang.jsonl
        few_shot_file=$ROOT_DIR/data/flores200/en-$lang/valid.jsonl
        
        src_file=$ROOT_DIR/data/flores200/en-$lang/test.en-$lang.$src_lang
        ref_file=$ROOT_DIR/data/flores200/en-$lang/test.en-$lang.$tgt_lang
        
        save_dir=$ROOT_DIR/exps/$model_name/flores_${shot}shot
        hypo_file=$save_dir/hypo.$lp.txt

        mkdir -p $save_dir
        cp $0 $save_dir
        
        accelerate launch --config_file $config_file --main_process_port 29501 $ROOT_DIR/src/eval_trans_fewshot.py \
            --model_name_or_path $model_dir \
            --test_file $test_file \
            --eval_mode fewshot \
            --few_shot_file $few_shot_file \
            --res_file $hypo_file \
            --lang_pair $lang_pair \
            --shot $shot \
            --num_batch 4 \
            --max_new_tokens 256 \
            --seed 42

        src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
        ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
        hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file
        lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp


        done
    done

    python $ROOT_DIR/src/mt_scoring_exp.py \
        --metric "bleu,comet_22"  \
        --comet_22_path $comet_model \
        --lang_pair $lang_pair_strs \
        --src_file $src_file_strs \
        --ref_file $ref_file_strs \
        --hypo_file $hypo_file_strs \
        --record_file "result_mt.xlsx"

done