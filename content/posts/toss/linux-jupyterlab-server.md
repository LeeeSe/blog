---
title: "Linux服务器配置在线JupyterLab服务，中文界面"
date: 2021-03-23T12:57:41+08:00
draft: false
tags: [
  "linux",
  "jupyterlab",
]
categories: [
  "折腾", 
]
---
机房机器太卡?电脑不在身边?Linux服务器配置JupyterLab,中文界面
<!--more-->
## 安装jupyterlab
```bash
$ pip install jupyterlab
```
## 配置文件
### 生成 jupyter_server_config.py配置文件
```bash
$ jupyter server --generate-config
```
### 修改jupyter_server_config.py配置文件

```bash
$ sudo vi /root/name/.jupyter/jupyter_server_config.py
```
**找到并修改以下语句（记得取消注释）**
```bash
c.ServerApp.ip = '*'  # 设置为 * 使得外网可以通过IP访问服务器上的jupyter服务
c.ServerApp.port = 8890  # 端口自己设置
c.ServerApp.root_dir = '/home/ls/code/DL'  # 工作路径自己设置
```
## 启动
### 启动jupyter lab 服务，并复制token
**启动服务**
```bash
$ jupyter lab
```
**启动服务后，在输出信息里找到token字样，复制token后面的一串编码(注意不要把“token=”这几个字符复制进去)**

### 在本地浏览器上输入服务器的  IP：端口(`注意端口要开放`)

```bash
IP:端口
举例：
192.168.0.1：8888
```
### 进入jupyter lab后设置密码
**进入jupyter后，会有个界面让你设置密码（`共有两个输入框，选择最靠下的那一对输入框`），刚才得到的token是“钥匙”，输入“钥匙”。并输入你想要设置的密码，便成功在本地进入服务器上的jupyter服务**

## 汉化
### CTRL + C 退出jupyter lab 服务，准备汉化
### 安装汉化包

```bash
$ pip install jupyterlab-language-pack-zh-CN
```
**不出意外应该提示找不到这个包，从这里下载包：[zh_CN语言包](http://82.156.3.172:8889/index.php/s/pi5FnjSbG87gJ8T)**

**pip 此本地安装语言包**

```bash
$ pip install jupyterlab_language_pack_zh_CN-0.0.1.dev0-py2.py3-none-any.whl
```
## 创建后台任务
**若关闭终端，jupyter lab服务会自动停止，为了让我们关闭服务器终端后jupyter lab仍然服务在公网上，我们需要创建后台任务**

```bash
$ screen -S jupyter_server
$ conda activate your_env
$ jupyter lab
CTRL + A + D 剥离当前screen，退出终端，screen仍然运行在后台
```
## 开始使用
关闭终端，本地浏览器输入IP：端口，输入密码（非token），进入汉化后的jpyter lab，尽情享受线上编程带来的便捷。

