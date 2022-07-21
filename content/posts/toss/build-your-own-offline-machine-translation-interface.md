---
title: "搭建你自己的离线机器翻译服务"
date: 2022-05-05T14:20:35+08:00
draft: false
tags: [
  "pytorch",
  "transformer",
  "huggingface",
  "api",
  "server",
]
categories: [
  "折腾", 
]
---

目前市场上大多数翻译服务都是在线调用谷歌或者百度的翻译API，或者付费下载离线SDK（不开源）自行部署，这种方法虽然简单，但是很没有掌控感。我也是一名自建服务爱好者，在自己腾讯云小破机上折腾过不少自建服务，既然翻译服务这么常用，为什不自己搭一个呢？
<!--more-->

## 翻译效果

~~~shell
Input：我像是山羊立足在陡峭的悬崖
Output：I'm like a goat standing on a steep cliff.
~~~

~~~shell
Input：life is Brief, but love is long
Output：生命是短暂的,但爱是漫长的。
~~~

## 环境准备

1. Windows/Linux/Mac
2. Python == 3.10
3. torch == 1.11 (cpu or gpu)
4. transformers == 4.18
5. pinferencia[uvicorn] == 0.1.0

### transformers

Transformers是[huggingface](https://huggingface.co/)推出的一个自然语言处理库，可由pip直接安装。Transformers 公开了大量的自然语言业界预训练模型，免费提供给公众使用。使用预训练模型可以降低计算成本、碳足迹，并节省从头开始训练模型的时间。

Transformers 对模型调用接口进行了高度封装，即使像我这种NLP小白也能轻松上手部署模型。本次的翻译模型就是从 Transformers 中获取的预训练模型。详情点击[这里](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)。

### pinferencia

Pinferencia 致力于成为最简单的机器学习模型部署工具,它可以快速且方便地把你的模型、算法、甚至是一个简简单单的函数部署成API供人访问。部署API是个体力活，对于小项目没必要浪费精力自己用Flask或者FastApi重写一套，直接采用现成的方案。Pinferencia详情点击[这里](https://pinferencia.underneathall.app/rc/pinferencia-is-different/#pinferencia_1)。

### uvicorn

一种ASGI Server，安装pinferencia时使用下面的命令可以一同安装

~~~shell
pip install pinferencia[uvicorn]
~~~

## 开发代码

### 下载模型及配置文件

~~~shell
git lfs install  # Git Large Files 提供大文件（模型）下载支持
git clone https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
~~~

其他模型及配置文件也可以用这种方式下载到本地，如果还是嫌速度慢，可以不运行第一条命令，这样clone下载的文件夹中的模型文件只是个占位文件，然后去官网手动下载模型文件，下载完成后改名覆盖一下占位模型即可。

### 编写app.py代码

得益于transformers库的模型接口的统一规范，以及Pinferencia的简单便捷，我们仅仅需要几行便可以成功搭建起翻译服务。

app.py
~~~python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pinferencia import Server

tokenizer = AutoTokenizer.from_pretrained("opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("opus-mt-en-zh")
trans = pipeline("translation", model=model, tokenizer=tokenizer)


def translate(text):
    return trans(text)


service = Server()
service.register(model_name="en-zh", model=translate)
~~~

其中，from_pretrained方法从提供的本地路径或者“在线路径”中获取模型，如果是在线路径则会自动下载模型，但受限于网络可能无法下载。经过测试我自用的电脑可以在线下载模型（开了梯子），但是服务器速度极慢，所以我推荐手动下载。

> 对于本地路径和在线路径的区别
>  - 如果是本地路径则传入从transformers官网下载好的模型文件夹名称
>  - 如果是在线路径，则需要从官网提供的模型列表中寻找项目名再传入函数，本例中为 Helsinki-NLP/opus-mt-en-zh

[模型列表](https://huggingface.co/models)


### 编写test.py代码

~~~python
import requests


def en2zh(text):
    response = requests.post(
        url="http://82.156.199.33:8001/v1/models/en-zh/predict",
        json={
            "data": text
        },
    )
    return response.json()['data'][0]['translation_text']


if __name__ == "__main__":
    print(en2zh("life is Brief, but love is long"))
~~~

> url参数将在“部署”步骤中定义

## 部署

~~~shell
uvicorn app:service --host 0.0.0.0 --port 8001
INFO:     Started server process [3420]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
~~~
部署成功，测试一下效果

## 测试

~~~shell
python test.py

生命是短暂的,但爱是漫长
~~~

## 后记

其实huggingface上还有很多有趣的自然语言模型，尤其是有一个问答机器人
，问什么答什么，非常智能，连脑筋急转弯都不在话下。不过缺点是模型是由英文语料训练而成的，输入输出都
只能是英语。于是我就在问答机器人头和尾各加了一个翻译处理，使得机器人最后可
以中文输入中文输出，这也是本文诞生的其中一个原因。

关于如何部署中文问答机器人，我会抽空再写一篇博客。