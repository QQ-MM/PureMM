#!/usr/bin/env bash

base_model_path=../model/vicuna-13b-v1.5
model_path=../model/QQMM_v1.0
conv_mode=vicuna_v1

# 生成结果
question_path=../mme_bench/format_mme_bench_v1.2.json
answers_dir=../mme_bench/answers/QQMM

python -m .model_vqa \
        --model-base ${base_model_path} \
        --model-path ${model_path} \
        --question-file ${question_path} \
        --answers-dir ${answers_dir} \
        --conv-mode ${conv_mode}
