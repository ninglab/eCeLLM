import os
import sys
from typing import List

import fire
import torch
import transformers
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from datasets import load_dataset
from datasets import disable_caching
disable_caching()

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.prompter import Prompter
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    dev_data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    #warmup_steps: int = 100,
    warmup_ratio: float = 0.1, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "mistral",
    optim: str = "adamw_torch",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"optim: {optim}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project.strip()) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def generate_and_tokenize_prompt(data_point):

        if "options" not in data_point:
            data_point["options"] = None

        inputs = prompter.generate_prompt(
            data_point["instruction"], 
            data_point["input"], 
            data_point["options"],)
            
        model_inputs = tokenize(
            inputs, add_eos_token=add_eos_token)
            
        model_inputs_len = len(model_inputs["input_ids"])

        if add_eos_token:
            model_inputs_len -= 1

        labels = tokenize(data_point["output"], add_eos_token=add_eos_token)
        labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM")

    model = get_peft_model(model, config)

    # load from huggingface
    dataset = pd.DataFrame(load_dataset("NingLab/ECInstruct")['train'])

    data = dataset[(dataset["split"] == "train") & (dataset["setting"] == "IND_Diverse_Instruction")].drop(["split", "setting", "few_shot_example", "task"], axis=1)
    data = Dataset(pa.Table.from_pandas(data))

    dev_data = dataset[(dataset["split"] == "val") & (dataset["setting"] == "IND_Diverse_Instruction")].drop(["split", "setting", "few_shot_example", "task"], axis=1)
    dev_data = Dataset(pa.Table.from_pandas(dev_data))
    
    ## load from google drive
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)
    
    # if dev_data_path.endswith(".json") or dev_data_path.endswith(".jsonl"):
    #     dev_data = load_dataset("json", data_files=dev_data_path)
    # else:
    #     dev_data = load_dataset(dev_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    train_data = data.shuffle().map(generate_and_tokenize_prompt)
    val_data = dev_data.shuffle().map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    #count the number of tokens
    num_tokens = 0
    for i in range(len(train_data)):
        num_tokens += len(train_data[i]['input_ids'])

    print(f'#tokens: {num_tokens/1000.0}k')

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True, 
            logging_steps=1,
            optim=optim,
            evaluation_strategy="epoch" if val_set_size > 0 else "no",
            save_strategy="epoch",
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, label_pad_token_id=-100,
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    model.save_pretrained(output_dir)
    pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, pytorch_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
