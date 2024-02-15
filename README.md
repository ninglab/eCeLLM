# eCeLLM

This repo contains the code, data, and models for "eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data"

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

* torch = 2.1.2
* transformers = 4.36.2
* fire = 0.5.0
* json = 2.0.9
* sklearn = 1.3.2
* peft = 0.7.1
* datasets = 2.15.0

## ECInstruct Dataset
The dataset is avaliable in [Hugging Face](https://huggingface.co/datasets/xin10/ECInstruct).
ECInstruct comprises 10 tasks, including attribute valua extraction, product relation predict,
product matching, sentiment analysis, senquential recommendation, multiclass product classification, product
substitute identification, query-product ranking, answerability prediction, and answer generation. 
ECInstruct is split into training sets, validation sets, IND
test sets, and OOD test sets.

All the data in ECInstruct is in the "ECInstruct" folder. As detailed in the paper, for each task, we could conduct training and evaluation under multiple settings. All data for these settings are available in the "ECInstruct" folder. For example, the "ECInstruct/Answer_Generation/IND_Diverse_Instruction" folder has the training set for learning models on the answer generation task with diverse instructions and the IND test set for this task.

## eCeLLM Models
The models are avaliable in [Hugging Face](https://huggingface.co/xin10).
Leveraging ECInstruct, we develop eCeLLM by instruction tuning general-purpose LLMs (base models).
The "eCeLLM" folder has the eCeLLM models instruction-tuned from 6 large base models: [Flan-T5 XXL](https://arxiv.org/abs/2210.11416), [Llama-2 13B-chat](https://arxiv.org/abs/2307.09288), [Llama-2 7B-chat](https://arxiv.org/abs/2307.09288), [Mistral-7B Instruct-v0.2](https://arxiv.org/abs/2310.06825), [Flan-T5 XL](https://arxiv.org/abs/2210.11416) and [Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/).

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
To conduct model inference, run <code>./inference.sh $model_path $test_path $output_path $base_model</code>.

<code>$model_path</code> is the path of the instruction-tuned model.

<code>$test_path</code> specifies the path of the test samples.

<code>$output_path</code> specifies the path where you want to save the inference output.

<code>$base_model</code> specifies the base model.

Example:
```
./inference.sh eCeLLM/Mistral-7B-Instruct-v0.2 ECInstruct/Product_Matching/IND_Diverse_Instruction/test.json evaluation/IND_Diverse_Instruction.json Mistral-7B-Instruct-v0.2
```

Please replace "inference.py" with "inference_T5.py" in "inference.sh" when inferencing Flan-T5-XXL and Flan-T5-XL.

## Evaluation
To evaluate the instruction-tuned model on specific tasks, run <code>python evaluate.py --task $task --setting $setting</code>.

<code>$task</code> is the task on which you want to conduct evaluation.

<code>$setting</code> specifies the evaluation setting.

Example:
```
python evaluate.py --task Product_Matching --setting IND_Diverse_Instruction
```

Please use "evaluate_T5.py" when evaluating Flan-T5-XXL and Flan-T5-XL.

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
