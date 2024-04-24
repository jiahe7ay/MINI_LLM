# coding=utf-8
from typing import Dict, Optional
import time
import os 

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import  TrainingArguments,DataCollatorForLanguageModeling
from trl import DPOTrainer
from peft import LoraConfig, TaskType, PeftModel

from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from dataclasses import dataclass
from os.path import dirname, abspath


#===================================================================================
# 以下为dpo训练配置
@dataclass
class DpoConfig:
    max_seq_len: int = 1024 + 8                  # 8 for eos token
    sft_model_file: str = 'D:\code\python\law_work\MINILLM\MINI_LLM\checkpoint-28500_sftmodel_v1_1.4b' # SFT后的模型路径

    tokenizer_dir: str = 'D:\code\python\law_work\MINILLM\MINI_LLM\checkpoint-28500_sftmodel_v1_1.4b'   # tokenizer一般和model权重放在同一个文件夹

    dpo_train_file: str = r'D:\code\python\law_work\MINILLM\MINI_LLM\datasets\final_dataset\my_dpo_train.json' # dpo的训练集
    dpo_eval_file: str = r'D:\code\python\law_work\MINILLM\MINI_LLM\datasets\final_dataset\my_dpo_eval.json' # dpo的测试集

    adapter_file: str = '/data/dpo/adapter_model.safetensors'
    log_dir: str = 'D:\code\python\law_work\MINILLM\MINI_LLM\logs'

    per_device_train_batch_size: int = 4
    num_train_epochs: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    logging_first_step: bool = True
    logging_steps: int = 20
    save_steps: int = 200
    output_dir: str = 'D:\code\python\law_work\MINILLM\MINI_LLM/dpo'  # dpo模型输出路径
    warmup_steps: int = 1000
    fp16: bool = True
    seed: int = 23333
    beta: float = 0.1

def get_dataset(split: str, file: str, cache_dir: str = '.cache') -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    dataset = load_dataset('json', data_files=file,  split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(sample: dict) -> Dict[str, str]:
        return {
            # add an eos token for signal that end of sentence, using in generate.
            "prompt": f"{sample['prompt']}<|im_end|>",
            "chosen": f"{sample['chosen']}<|im_end|>",
            "rejected": f"{sample['rejected']}<|im_end|>",
        }

    return dataset.map(split_prompt_and_responses).shuffle(2333)



def train_dpo(config: DpoConfig, peft_config: LoraConfig=None) -> None:

    # step 1. 加载tokenizer
    tokenizer = QWenTokenizer.from_pretrained(config.tokenizer_dir)

    tokenizer.pad_token_id = tokenizer.im_end_id
    tokenizer.bos_token_id = tokenizer.im_end_id
    tokenizer.eos_token_id = tokenizer.im_end_id

    # step 2. 加载SFT模型
    # model_train, model_ref = None, None
    # if os.path.isdir(config.sft_model_file):
    # 传入文件夹则 from_pretrained
    model_train = QWenLMHeadModel.from_pretrained(config.sft_model_file)
    model_ref = QWenLMHeadModel.from_pretrained(config.sft_model_file)

    # 4. 加载训练数据集
    train_dataset = get_dataset("train", file=config.dpo_train_file)

    # 5. 加载评估数据集
    # eval_dataset = get_dataset("train", file=config.dpo_eval_file)
    eval_dataset = get_dataset("train", file=config.dpo_eval_file)

    # 6. 初始化训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_first_step=True,
        logging_steps=config.logging_steps, 
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        optim="adafactor",
        report_to="tensorboard",
        log_level='info',
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=config.fp16,
        seed=config.seed,
        logging_dir=config.log_dir,
    )

    # 7. 初始化 DPO trainer
    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=training_args,
        beta=config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        max_target_length=config.max_seq_len,
        max_prompt_length=config.max_seq_len,
        generate_during_eval=True,
        is_encoder_decoder=True,
        # data_collator=data_collator
    )

    # 8. 训练
    dpo_trainer.train(
        # resume_from_checkpoint=True
    )

    # 9. save log
    loss_log = pd.DataFrame(dpo_trainer.state.log_history)
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    loss_log.to_csv(f"{log_dir}/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    
    # 10. 保存模型/lora
    suffixe = '/lora/' if peft_config is not None else '/dpo'
    model_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + suffixe

    dpo_trainer.save_model(model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))

   
if __name__ == "__main__":

    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model 
         inference_mode=False, 
         r=16, 
         lora_alpha=16, 
         lora_dropout=0.1, 
         bias="all",
    )

    dpo_config = DpoConfig()

    train_dpo(dpo_config, peft_config=None)





    