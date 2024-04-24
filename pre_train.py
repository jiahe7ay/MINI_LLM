# %%
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from datasets import Dataset, load_dataset
from qwen.configuration_qwen import QWenConfig
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

# torch._dynamo.config.optimize_ddp = False
# %%
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


attn_implementation = "flash_attention_2"
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = "eager"

# %% [markdown]
# # 1. 训练数据来源

TRAIN_FILES = [
    # './datasets/wiki_fi.parquet',
    # './datasets/baike_chunk_512_5.6M_0.parquet',
    #     './datasets/baike_chunk_512_5.6M_1.parquet',
    #     './datasets/sky1.parquet',
    "./datasets/sky2.parquet",
    "./datasets/sky3.parquet",
    "./datasets/sky4.parquet",
    "./datasets/sky5.parquet",
    "./datasets/sky6.parquet",
    "./datasets/sky7.parquet",
    "./datasets/sky8.parquet",
    "./datasets/sky9.parquet",
    "./datasets/sky10.parquet",
    "./datasets/sky11.parquet",
    "./datasets/sky12.parquet",
    "./datasets/sky13.parquet",
    "./datasets/sky14.parquet",
    "./datasets/sky15.parquet",
    "./datasets/sky16.parquet",
    "./datasets/sky17.parquet",
    "./datasets/sky18.parquet",
    "./datasets/sky19.parquet",
    "./datasets/sky20.parquet",  #     './datasets/sky10.parquet',
    #     './datasets/mbvc1.parquet',
    #     './datasets/sky1.parquet',
]

EVAL_FILE = "./datasets/pretrain_eval_512_1w.parquet"

# %%


@dataclass
class PretrainArguments:
    tokenizer_dir: str = "./qwen/"
    model_save_dir: str = "./model_save/pre/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512

    # Windows 使用默认的attention实现，
    attn_implementation: str = (
        "eager" if platform.system() == "Windows" else attn_implementation
    )


pretrain_args = PretrainArguments()

# %% [markdown]
# # 2. 加载训练好的tokenizer
# 如果你使用的`add_tokens`方法添加了自己的token，必须要用`len(tokenizer)`获取长度，`tokenizer.vocab_size`统计不包含你添加的字符。

# %%
tokenizer = QWenTokenizer.from_pretrained(pretrain_args.tokenizer_dir)
tokenizer.pad_token_id = tokenizer.im_end_id
# %% [markdown]
# # 5. 定义模型
# 从`config`定义，不是`from_pretrained`。
# 为了方便cuda计算，词表的大小注意一下，如果不是64的整数倍，可以手动向上取整为64的整数倍，也可以是其他 $2^x$ 数值的整数倍，如32、128、256都行。

# %%
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"final vocab size: {vocab_size}")

# %% [markdown]
# ## token to id缓存到文件，使用的时候不用再次tokenize
# 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
# %%
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32


def token_to_id(samples: dict) -> dict:

    batch_txt = samples["text"]
    outputs = tokenizer(
        batch_txt,
        padding=False,
        return_attention_mask=False,
        truncation=True,
        max_length=pretrain_args.max_seq_len
    )

    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {"input_ids": input_ids}


# print(token_to_id({'text':['判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。\n','下面是一篇文章的开头: "为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。']}))

# step 3 加载数据集


# %%
def get_maped_dataset(files) -> Dataset:
    dataset = load_dataset(
        path="parquet",
        data_files=files,
        split="train",
        cache_dir=".cache",
        keep_in_memory=False,
    )
    maped_dataset = dataset.map(
        token_to_id,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=24,
        keep_in_memory=False,
    )
    return maped_dataset


train_dataset = get_maped_dataset(pretrain_args.train_files)
eval_dataset = get_maped_dataset(pretrain_args.eval_file)

print(train_dataset, eval_dataset)
# %% [markdown]
# # 4. 定义data_collator
# `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型

# %%
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%
# 如果配置了flash_attention_2，请手动设置set_default_dtype为float16
#  Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes.
if pretrain_args.attn_implementation == "flash_attention_2":
    torch.set_default_dtype(torch.bfloat16)


config = QWenConfig.from_pretrained("./qwen")
# model = QWenLMHeadModel.from_pretrained("./1")
model = QWenLMHeadModel(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"QWen size: {model_size / 1000**2:.1f}M parameters")

# %% [markdown]
# # 6. cuda cache回调函数


# %%
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        """
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


my_trainer_callback = MyTrainerCallback()

# %% [markdown]
# # 6. 定义训练参数

# %%
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=10,
    num_train_epochs=1,
    weight_decay=0.1,
    ddp_find_unused_parameters=False,
    warmup_steps=0,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=50,
    save_strategy="steps",
    save_total_limit=4,
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    log_level="info",
    logging_first_step=True,
    # group_by_length=True,
    # deepspeed='./ds_config_one_gpu.json',
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_trainer_callback],
)

# %% [markdown]
# # 7. 开始训练
# `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练

# %%
trainer.train(  #'model_save/pre/checkpoint-3400'
    # resume_from_checkpoint=True
)

# %% [markdown]
#  计算困惑度Perplexity

# %%
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

# %% [markdown]
# # 8. 最后保存训练的loss日志和模型

# %%

# loss_log = pd.DataFrame(trainer.state.log_history)
# loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(pretrain_args.model_save_dir)
