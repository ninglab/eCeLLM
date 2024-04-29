#!/bin/tcsh

set model_name = $1
set task = $2
set setting = $3
set output_path = $4

if ($model_name == "NingLab/eCeLLM-M") then
    set prompt_template_name = mistral
else
    set prompt_template_name = alpaca
endif

python inference_merged.py \
    --model_name $model_name \
    --task $task \
    --setting $setting \
    --output_data_path $output_path \
    --prompt_template $prompt_template_name \
