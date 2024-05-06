# eCeLLM
✨ Our paper was accepted to ICML 2024.

This repo contains the code for [eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data](https://arxiv.org/abs/2402.08831).

## Introduction
We introduce ECInstruct, 
the first open-sourced, large-scale, and high-quality benchmark instruction dataset for e-commerce.
ECInstruct covers 116,528 samples from 10 real, widely performed e-commerce tasks of 4 categories.
ECInstruct undergoes rigorous and thorough scrutiny and is carefully crafted to enable a wide spectrum of empirical testing and exploration, 
including in-domain (IND) evaluation, out-of-domain (OOD) evaluation, and task-specific studies.
Leveraging ECInstruct, we develop eCeLLM, a series of generalist large language models (LLMs) for e-commerce.
Our experimental results demonstrate that eCeLLM models substantially outperform baseline models, 
including the most advanced GPT-4 and
the state-of-the-art (SoTA) task-specific models, on almost
all the 10 tasks in IND evaluation.
eCeLLM also exhibits excellent generalizability to
OOD settings, including unseen products and unseen instructions

## Requirements:

* python = 3.10.6
* torch = 2.1.2
* transformers = 4.36.2
* fire = 0.5.0
* scikit-learn = 1.3.2
* peft = 0.7.1
* datasets = 2.15.0

## ECInstruct Dataset
The dataset is available in [Hugging Face](https://huggingface.co/datasets/NingLab/ECInstruct).
ECInstruct comprises 10 tasks, including attribute value extraction, product relation prediction,
product matching, sentiment analysis, sequential recommendation, multiclass product classification, product
substitute identification, query-product ranking, answerability prediction, and answer generation. 
ECInstruct is split into training sets, validation sets, IND
test sets, and OOD test sets.

We also provide the product labels for the test set of query-product ranking task in `/data_label/label.json`, which can be used for evaluation. Please check https://github.com/amazon-science/esci-data for more details.

## eCeLLM Models
The models are available in [Hugging Face](https://huggingface.co/NingLab).
Tuned on our ECInstruct dataset, we develop eCeLLM by instruction tuning general-purpose LLMs (base models).
The eCeLLM-S is instruction-tuned on base models [Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/), eCeLLM-M is tuned on [Mistral-7B Instruct-v0.2](https://arxiv.org/abs/2310.06825), and eCeLLM-L is tuned on [Llama-2 13B-chat](https://arxiv.org/abs/2307.09288).

## Training
To instruction-tune the base models, run <code>./finetune.sh $number_epoches $base_model $number_validation_samples</code>

<code>$number_epoches</code> is the number of epoches.

<code>$base_model</code> specifies the base model.

<code>$number_validation_samples</code> specifies the number of validation samples.

Example:
```
./finetune.sh 3 Mistral-7B-Instruct-v0.2 10k
```

Please replace "finetune.py" with "finetune_T5.py" in "finetune.sh" when tuning Flan-T5-XXL and Flan-T5-XL.

## Inference
To conduct model inference, run <code>./inference.sh $model_path $task $setting $output_path $base_model</code>.

<code>$model_path</code> is the path of the instruction-tuned model.

<code>$task</code> specifies the task to be tested.

<code>$setting</code> specifies the evaluation setting.

<code>$output_path</code> specifies the path where you want to save the inference output.

<code>$base_model</code> specifies the base model.

Example:
```
./inference.sh eCeLLM/Mistral-7B-Instruct-v0.2 Product_Matching IND_Diverse_Instruction evaluation/IND_Diverse_Instruction.json Mistral-7B-Instruct-v0.2
```

Please replace "inference.py" with "inference_T5.py" in "inference.sh" when inferencing Flan-T5-XXL and Flan-T5-XL.

## Evaluation
To evaluate the instruction-tuned model on specific tasks, run <code>python evaluate.py --task $task --setting $setting</code>.

<code>$task</code> is the task on which you want to conduct the evaluation.

<code>$setting</code> specifies the evaluation setting.

Example:
```
python evaluate.py --task Product_Matching --setting IND_Diverse_Instruction
```

Please use "evaluate_T5.py" when evaluating Flan-T5-XXL and Flan-T5-XL.

## Inference_merged
To conduct inference of the model loaded from huggingface, run <code>./inference_merged.sh $model_name $task $setting $output_path</code>.

<code>$model_name</code> is the name of the huggingface model.

<code>$task</code> specifies the task to be tested.

<code>$setting</code> specifies the evaluation setting.

<code>$output_path</code> specifies the path where you want to save the inference output.

Example:
```
./inference_merged.sh NingLab/eCeLLM-M Product_Matching IND_Diverse_Instruction evaluation/PM.json
```

## Citation
```bibtex
@misc{peng2024ecellm,
      title={eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data}, 
      author={Bo Peng and Xinyi Ling and Ziru Chen and Huan Sun and Xia Ning},
      year={2024},
      eprint={2402.08831},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
