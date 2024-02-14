#!/bin/tcsh

set model_path = $1
set data_path = $2
set output_path = $3
set model = $4

if ($model == "Flan-T5-XXL") then
    set base_model = "google/flan-t5-xxl"
    set prompt_template_name = alpaca
else if ($model == "Llama-2-13B-chat") then
    set base_model = "meta-llama/Llama-2-13b-chat-hf"
    set prompt_template_name = alpaca
else if ($model == "Llama-2-7B-chat") then
    set base_model = "meta-llama/Llama-2-7b-chat-hf"
    set prompt_template_name = alpaca
else if ($model == "Mistral-7B-Instruct-v0.2") then
    set base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    set prompt_template_name = mistral
else if ($model == "Flan-T5-XL") then
    set base_model = "google/flan-t5-xl"
    set prompt_template_name = alpaca
else if ($model == "Phi-2") then
    set base_model = "microsoft/phi-2"
    set prompt_template_name = alpaca
else
    set base_model = ""
    set prompt_template_name = ""
endif

python inference.py \
    --base_model $base_model \
    --lora_weights $model_path \
    --data_path $data_path \
    --output_data_path $output_path \
    --prompt_template $prompt_template_name \
