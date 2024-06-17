#!/bin/bash

models=("mistral-7b" "gemma-7b", "llama-8b", "llama-70b", "gpt-4-turbo-direct", "baseline")
strategies=("zeroshot" "2-shot" "3-shot" "zeroshot-cot" "2-shot-cot" "3-shot-cot")

for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        python3 run_evaluation.py --model $model --dataset morehopqa --fewshot-dataset morehopqa --output_file final --strategy $strategy
    done
done