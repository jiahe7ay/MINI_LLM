# %%
import time

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from datasets import load_dataset
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

# %% [markdown]
# # 1. 定义训练数据，tokenizer，预训练模型的路径及最大长度

# %%
SFT_FILES = [
    "./datasets/aplca1.parquet",
    "./datasets/aplca2.parquet",
    "./datasets/aplca3.parquet",
]
tokenizer_dir = "./model_save/pre3/checkpoint-16600"
sft_from_checkpoint_file = "./model_save/pre3/checkpoint-16600"
model_save_dir = "./model_save/sft/"
max_seq_len = 512

# %% [markdown]
# # 2. 加载训练数据集

PROMPT_DICT = {
    "prompt_input": ("你是一个助手 " "用户: {instruction} {input} 回答: "),
    "prompt_no_input": ("你是一个助手 " "用户: {instruction}  回答: "),
}

# %%
dataset = load_dataset(
    path="parquet", data_files=SFT_FILES, split="train", keep_in_memory=False
)

print(dataset)


# %%
# samples = dataset[0:2]
# print(samples)

# %%
tokenizer = QWenTokenizer.from_pretrained(tokenizer_dir)
print(f"vicab size: {len(tokenizer)}")
tokenizer.pad_token_id = tokenizer.im_end_id


map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32


def format_example(example):
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )
    if example.get("input"):
        target = example["output"] + "<|im_end|>"
        context = prompt_input.format_map(
            dict(instruction=example["instruction"], input=example["input"])
        )

        example["context"] = context
        example["target"] = target
    else:
        target = example["output"] + "<|im_end|>"
        context = prompt_no_input.format_map(dict(instruction=example["instruction"]))

        example["context"] = context
        example["target"] = target
    return example


def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    input_ids = tokenizer(
        prompt + target,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    )
    seq_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    )
    input_ids_len = seq_ids.input_ids.ne(tokenizer.pad_token_id).sum().item()

    return {"input_ids": input_ids.input_ids[0], "seq_len": input_ids_len}


# print(batched_formatting_prompts_func(samples))

# %%
tokenized_datasets = dataset.map(
    function=format_example, num_proc=32, keep_in_memory=False
)
print("1")
print(tokenized_datasets)
tokenized_datasets = tokenized_datasets.map(
    function=preprocess, num_proc=32, keep_in_memory=False
).shuffle(23333)
print("2")
print(tokenized_datasets)
# %% [markdown]
# ## 2.2 定义data_collator


# %%
# mlm=False表示训练的是CLM模型
def data_collator(fetures):
    len_ids = [len(feture["input_ids"]) for feture in fetures]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    postion_ids_list = []
    labels_list = []
    for ids_l, feture in sorted(zip(len_ids, fetures), key=lambda x: -x[0]):
        ids = feture["input_ids"]
        seq_len = feture["seq_len"]
        labels = [-100] * seq_len + ids[seq_len:] + [-100] * (longest - ids_l)
        ids = ids + [tokenizer.im_end_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {"input_ids": input_ids, "labels": labels}


# %% [markdown]
# # 4. 加载预训练模型

# %%

model = QWenLMHeadModel.from_pretrained(sft_from_checkpoint_file)

model_size = sum(t.numel() for t in model.parameters())
print(f"Qwen size: {model_size / 1000**2:.2f}M parameters")

# %% [markdown]
# ## 定义训练过程中的回调函数
# N次log之后情况cuda缓存，能有效缓解低显存机器显存缓慢增长的问题


# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()


empty_cuda_cahce = EmptyCudaCacheCallback()

# %%
my_datasets = tokenized_datasets.train_test_split(test_size=4096)
print("m")
print(my_datasets)
# %% [markdown]
# # 5. 定义训练参数

# %%
args = TrainingArguments(
    output_dir=model_save_dir,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=0,
    learning_rate=6e-5,
    ddp_find_unused_parameters=False,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    report_to="tensorboard",
    optim="adamw_torch",
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    log_level="info",
    logging_first_step=True,
    # group_by_length=True,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=my_datasets["train"],
    eval_dataset=my_datasets["test"],
    callbacks=[empty_cuda_cahce],
)


# %% [markdown]
# # 6. 开始训练

# %%
trainer.train(
    # resume_from_checkpoint=True
)

# %% [markdown]
#  计算困惑度Perplexity

# %%
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

# %% [markdown]
# # 7. 保存日志和模型

# %%
loss_log = pd.DataFrame(trainer.state.log_history)
# loss_log.to_csv(f"./logs/sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(model_save_dir)

# %%
