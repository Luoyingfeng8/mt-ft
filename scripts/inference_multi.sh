#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="$ROOT_DIR/cache/"
export MODELSCOPE_CACHE="$ROOT_DIR/cache/"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NPROC_PER_NODE=8


# model
model_name=Qwen3-1.7B-Base
model_dir=$ROOT_DIR/exps/$model_name

# eval
comet_model=$ROOT_DIR/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=$ROOT_DIR/model_card/XCOMET-XXL/checkpoints/model.ckpt


for name in `ls $model_dir`; do

    predict_model_dir=$model_dir/$name/best

    lang_pair_strs=""
    src_file_strs=""
    ref_file_strs=""
    hypo_file_strs=""

    for lang in de ru zh cs vi ko ne sw ha;do
        for src in $lang en ;do 

            if [ $src = "en" ]; then # en2zh
                src_lang=en
                tgt_lang=$lang 
            else  # zh2en
                src_lang=$lang
                tgt_lang=en 
            fi

            lp=${src_lang}2${tgt_lang}
            src_file=$ROOT_DIR/data/flores200/en-${lang}/test.en-$lang.$src_lang
            ref_file=$ROOT_DIR/data/flores200/en-${lang}/test.en-$lang.$tgt_lang
            test_file=$ROOT_DIR/data/sft_test/test.$lp.jsonl

            output_dir=$predict_model_dir/decode_result/$lp
            mkdir -p $output_dir
            #############################!!!!!
            rm -rf $output_dir/*
            ########################
            cp $0 $output_dir


            swift infer \
                --infer_backend pt \
                --val_dataset $test_file \
                --dataset_shuffle False \
                --val_dataset_shuffle False \
                --model_type qwen3 \
                --model $predict_model_dir \
                --torch_dtype bfloat16 \
                --max_new_tokens 512 \
                --max_batch_size 8 \
                --num_beams 4 \
                --max_length 512 \
                --dataset_num_proc 8 \
                --temperature 0 \
                --result_path $output_dir/generated_predictions.jsonl | tee $output_dir/train.log


            jq -r '.response' $output_dir/generated_predictions.jsonl > $output_dir/hypo.$lp.txt
            
            hypo_file=$output_dir/hypo.$lp.txt

            lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp
            src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
            ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
            hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file

        done
    done

    # metric="bleu,comet_22,xcomet_xxl" 
    metric="bleu,comet_22" 
    python $ROOT_DIR/src/mt_scoring.py \
        --metric $metric  \
        --comet_22_path $comet_model \
        --xcomet_xxl_path $xcome_model \
        --lang_pair $lang_pair_strs \
        --src_file $src_file_strs \
        --ref_file $ref_file_strs \
        --hypo_file $hypo_file_strs \
        --record_file "result_mt.xlsx"

done