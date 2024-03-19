# Mini-llm
Created by Lil2J
## 📝介绍
本项目是我个人关于一个小参数量的中文大模型的一个实践复现。

主要参考这两个开源项目：

1.https://github.com/charent/Phi2-mini-Chinese

2.https://github.com/DLLXW/baby-llama2-chinese

包含：预训练、SFT指令微调、**奖励模型以及强化学习**（待做）完整流程。

希望分享给大家，也希望大家一起来完善！


## 📚项目简介
- 训练一个参数量1.4b预训练模型，基座模型选的是QWEN,训练的token数量为8b左右
- 构建包含预训练、SFT指令微调整个完整流程的LLM代码仓库，包含DeepSpeed分布式训练技术

## 🌟Quick Start
```bash
# 1. 在“Baby-llama2-chinese Corpus”的百度网盘中下载维基百科和百度百科的预训练语料和aplca数据。
#    在https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main 上下载数据
#    在https://huggingface.co/BelleGroup 上下载train_2M_CN,train_1M_CN和train_0.5M_CN
#    因为算力资源有限，我只下载了前20个数据文件
#    将所有数据tokenize之后，token数量大概为8b
# 2. 将下载好的数据放到你想要的目录下
# 3. 切换到dataset_utils目录下运行generate_data.py,运行前修改py文件，将处理数据的函数的注释去掉，才能运行起来
# 4. 运行generate_data.py.py，在./datasets/目录下生成parquet文件
cd dataset_utils
python3 generate_data.py
#5. 修改train.sh 文件 如果是单卡运行的话  把--multi_gpu 去掉，然后--config_file 后面接accelerate_one_gpu.yaml  如果是多卡的话，就把 accelerate_multi_gpu.yaml中 num_processes: 4
#改为自己的卡数

#开启预训练
sh train.sh pre_train.py

#6.预训练完之后，修改sft.py中的模型权重加载路径
#开启sft微调
sh train.sh sft.py

#7.修改test.py的权重路径，就能进行测试了
python3 test.py

```



## 🤖预训练
1. **模型底座**：模型的底座使用了qwen的模型，选择它的原因是：1.它是一个很成熟的中文大模型开源项目 2.我懒得自己构建tokenizer了，我看到qwen的tokenizer的压缩率挺好的，就直接拿来用了，既然tokenizer都拿了，就也直接用它的模型了


2. **预训练语料（Corpus for pre-training ）**：
   这次预训练用了以下几个经典数据集：

   Wiki中文百科：[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) 

   BaiduBaiKe：[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb    

   天工数据集：https://huggingface.co/datasets/Skywork/SkyPile-150B/tree/main/data

      

### 预训练语料预处理
数据预处理采取QWEN的通用做法，在末尾加上一个结束符号`<|im_end|>`，与下一个文章区分开。
如果文章超过规定的长度，将其截断，截断部分作为下个样本
   
## 💡SFT指令微调
LLM微调的目的是将预训练模型中的知识引导出来的一种手段，通俗的讲就是教会模型说人话。
1. **微调方法**：自然语言处理目前存在一个重要的范式：一般领域数据的大规模预训练，对特定任务或领域的适应。因此，为了让预训练模型在特定任务或领域有不错的表现，需要对模型进行微调。

   ### LLM微调方法
 
2. **SFT微调数据**：LLM在垂直领域的适应已经是2023年的主格调，因此各个领域的SFT语料和微调模型层出不穷。目前已经有大佬整理并持续更新这方面的[最新进展](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)，大家有需要可以自己访问。
   
   本项目主要针对两类SFT语料进行模型微调，如下：
      
   **日常问答SFT数据**：

   | SFT语料                                                                       | 描述                                                                  |
   |-----------------------------------------------------------------------------|---------------------------------------------------------------------|
   | alpaca-zh：[alpaca-zh]([https://github.com/hiyouga/ChatGLM-Efficient-Tuning/tree/main/data](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/data/self_cognition.json)) | 该数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。 |
   | bell：[bell]([https://huggingface.co/datasets/BelleGroup/](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN))         | 源自BelleGroup的一部分SFT数据。包含约300万条由BELLE项目生成的中文指令数据。|

  

### SFT样本构建
因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。所以自行参考sft.py即可！



## 🥇模型权重以及评测

**权重下载**

预训练权重：https://huggingface.co/Lil2J/mini_llm/tree/main

sft模型权重：https://huggingface.co/Lil2J/mini_llm_sft/tree/main

1. **预训练模型**

我首先先跑了Wiki中文百科 + BaiduBaiKe 
![wiki+baidu.png](assets/wiki+baidu.png)
预训练语料： Wiki中文百科 + BaiduBaiKe 

然后再跑天工的数据
![sky.png](assets/sky.png)
预训练语料： 天工数据集前20个文件

2. **sft模型**

![sft.png](assets/sft.png)
微调语料： aplca数据+bell:train_2M_CN,train_1M_CN和train_0.5M_CN

3. **sft模型效果**

```bash
#SFT微调模型的推理：test.py。
python3 test.py
```

![wiki+baidu.png](assets/4.png)
![wiki+baidu.png](assets/1.png)
![wiki+baidu.png](assets/2.png)
![wiki+baidu.png](assets/3.png)
![wiki+baidu.png](assets/5.png)
![wiki+baidu.png](assets/6.png)
![wiki+baidu.png](assets/7.png)
![wiki+baidu.png](assets/8.png)
![wiki+baidu.png](assets/10.png)



## 其他
有什么问题和想一起搞大模型的可以加wx:ForeverM1LAn 进行交流





