---
title: "pytorch复现——模型篇——LeNet"
date: 2020-09-10T12:57:41+08:00
draft: false
tags: [
  "论文复现",
  "pytorch",
]
categories: [
  "论文复现", 
]
---

LeNet的介绍及pytorch复现
<!--more-->

## 模型简介

我们常说的LeNet应该是指1998年 LeCun 发表的论文中的LeNet-5，它是CNN卷积神经网络的开山之作，至此之后卷积神经网络遍地开花，各种基于卷积神经网络的巧妙网络结构不断地被创造出来并取得了良好的效果。

![LeNet模型图](https://img-service.csdnimg.cn/img_convert/0e9275234851ffaa0966717234135aba.png#pic_center)




### 输入

输入均为32 *32pixel的黑白色手写数字图，维度为1 *32 *32。注：手写数字图来自mnist数据集，原数据集最大尺寸为28 * 28 ， 这样做是希望潜在的明显特征，比如笔画断续，角点等能够出现在最高层卷积核感受野的中心。 

### 隐藏层

C1为卷积层，kernel_size = 5*5，stride = 1，padding = 0，kernel_num = 6。由于没有padding，输出为6 *28 *28。

S2为池化层，kernel_size = 2*2，stride = 2，padding = 0，kernel_num = 6。`注意，此池化层与我们平时所见的最大池化和平均池化均不同`，他是将四个点加和并乘以一个权重再加上一个偏置得到的。即这个池化层是可训练的，而我们平时所见的池化是不可训练的。输出为6 *14 *14。LeNet的下采样层pytorch代码如下：

```python
class Downsampling2d(nn.Module):  # lenet中的降采样层非普通的池化层，lenet中的降采样层具有可学习的参数
    
    def __init__(self, in_channel, kernel_size = (2,2)):
        super(Downsampling2d, self).__init__()
        # lenet中的降采样是对卷积对应的四个点加和再乘以一个权重，再加上偏置
        # 可以用平均池化代替加和，尽管平均池化有除以4的过程，但因为设置有权重而线性抵消
        self.avg_pool2d = nn.AvgPool2d(kernel_size)
        self.in_channel = in_channel
        self.weights = nn.Parameter(torch.randn(in_channel), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(in_channel), requires_grad=True)


    def forward(self, x):
        # input.shape = (n, in_channel, h, w)   
        x = self.avg_pool2d(x)
        outs = []
        for i in range(self.in_channel):
            out = x[:,i] * self.weights[i] + self.bias[i]
            outs.append(out.unsqueeze(1))

        return torch.cat(outs, 1)
```

C3为卷积层，kernel_size = 5*5，stride = 1，padding = 0，kernel_num = 16。`C3层在LeNet中比较特殊，它的每一个 Feature Map 并不都是是与前一层的所有Feature Map连接`，而是故意地选择前一层Feature Map的一部分来与之连接，有很浓重的人工设计的味道。这样做一方面是减少了C3层的参数，另一方面是强迫C3层卷积核接受不同的输入，从而强迫其学习到不同的特征，很巧妙的一个小设计。输出为16 *10 *10。C3卷积层代码：

```python
class DropoutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(DropoutConv2d, self).__init__()

        mapping = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        self.in_channel = in_channels
        self.out_channel = out_channels
        mapping = torch.tensor(mapping, dtype=torch.long)

        self.register_buffer('mapping', mapping)
        self.convs = {}   # 用列表或者字典等装载nn的各种方法后，需要挨个注册到模块中
        for i in range(mapping.size(1)):
            conv = nn.Conv2d(mapping[:,i].sum().item(),1 ,kernel_size)
            module_name = 'conv{}'.format(i)
            self.convs[module_name] = conv
            # 通过 add_module 将 conv 中的参数注册到当前模块中
            # 若不注册则不会作为权重放入GPU中更新参数
            self.add_module(module_name, conv)

    def forward(self, x):
        out = []
        for i in range(self.mapping.size(1)):
            # .nonzero 返回矩阵中非零元素的索引的张量
            # squeeze,去掉维数为1的维度
            # in_channels是mapping中1的index
            index_channels = self.mapping[:, i].nonzero().squeeze()
            in_tensors = x.index_select(1, index_channels)
            conv_out = self.convs['conv{}'.format(i)](in_tensors)
            out.append(conv_out)

        return torch.cat(out, 1)
```

S4为池化层，与S2相似。kernel_size = 2*2，stride = 2，padding = 0，kernel_num = 16。输出为6 *5 *5。

C5为卷积层，kernel_size = 5* 5，kernel_num = 120。每个卷积核的大小为16* 5* 5，而输入刚好也为为16* 5 *5的特征图，则卷积后输出为120 *1 *1。（先flatten再全连接效果是一样的，采用哪个方法都行）

F6为全连接层，输出为 84* 1* 1，公式为:
$$
y_i = \sum_j{w_{ij} \cdot x_j +b_{i}}
$$
至于为什么设计成84而不是74或者94，实际上是对一些常用的字符用一张7 * 12的比特图进行了编码，-1表示白色，1表示黑色。当模型训练好之后，F6层的输出就对应着输入数字的编码，这个编码的长度是84。这里给出数字0的编码：

```python
_zero = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]
```

C1~ F6层所采用的激活函数为tanh函数，可通过线性变换为sigmoid函数。tanh函数输出值范围为（-1~1）。
$$
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

$$
sigmoid(x) = \frac{1}{1+e^{-x}}
$$



### 输出

Output层采用全连接层的连接方式，`但输出的计算方法与F6全连接层输出的计算方法不同`，采用 径向基函数（RBF） ，输出为 10 *1 *1，公式为：
$$
y_i = \sum_j{(xj - w_{ij})^2}
$$
其中Output层参数Wij由`人工设计并固定`，那他们是如何进行设计的呢？那么我们先回头看F6层的编码表，如果我们向网络中输入数字0的图像，假设此时网络已经训练好了，F6层会输出一排长度为84的编码：“1 -1 1 1 1 -1 1 -1·····”，那么我们拿之前编好的数字0的编码表与之对应，会发现两个编码相同或非常接近。这里看不明白的再去看看F6层的介绍。这里给出数字0的人工编码。

我们将10输出单元排成一列，从上到下分别代表数字0~9。我们先给第一个输出单元0人工设置权重，由于是全连接，则每个输出单元对应84个输入，那么就有84个权重Wij，我们把这84个权重Wij依照字符0的编码表设置成1或者-1，然后按照RBF公式计算，会发现如果输入的84个单元都等于84个权重时，得出的y是0。那么说明输入的图片是数字0。同理，如果第i个节点的值为0，则表示网络识别的结果是数字i。

这里给Output（RBF）层代码：

```python
class RBF(nn.Module):
    def __init__(self, in_features, out_features, init_weight=None):
        super(RBF, self).__init__()
        ## register_buffer 在内存中定义一个常量，不会被optimizer更新
        if init_weight is not None:
            self.register_buffer('weight', torch.tensor(init_weight))
        else:
            self.register_buffer('weight', torch.rand(in_features, out_features))
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2)
        return x
```



## 完整代码复现（pytorch）

本文的LeNet复现`不同于其他博客中的简化版`，如C3层采用dropout或完全放弃dropout，池化层使用了不可训练的平均或最大池化，output层使用了softmax等，本文严格按照论文进行模型复现。

```python
import torch
from torch import nn


class Downsampling2d(nn.Module):  # lenet中的降采样层非普通的池化层，lenet中的降采样层具有可学习的参数
    
    def __init__(self, in_channel, kernel_size = (2,2)):
        super(Downsampling2d, self).__init__()
        # lenet中的降采样是对卷积对应的四个点加和再乘以一个权重，再加上偏置
        # 可以用平均池化代替加和，尽管平均池化有除以4的过程，但因为设置有权重而线性抵消
        self.avg_pool2d = nn.AvgPool2d(kernel_size)
        self.in_channel = in_channel
        self.weights = nn.Parameter(torch.randn(in_channel), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(in_channel), requires_grad=True)


    def forward(self, x):
        # input.shape = (n, in_channel, h, w)   
        x = self.avg_pool2d(x)
        outs = []
        for i in range(self.in_channel):
            out = x[:,i] * self.weights[i] + self.bias[i]
            outs.append(out.unsqueeze(1))

        return torch.cat(outs, 1)


class DropoutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(DropoutConv2d, self).__init__()

        mapping = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                   [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        self.in_channel = in_channels
        self.out_channel = out_channels
        mapping = torch.tensor(mapping, dtype=torch.long)

        self.register_buffer('mapping', mapping)
        self.convs = {}   # 用列表或者字典等装载nn的各种方法后，需要挨个注册到模块中
        for i in range(mapping.size(1)):
            conv = nn.Conv2d(mapping[:,i].sum().item(),1 ,kernel_size)
            module_name = 'conv{}'.format(i)
            self.convs[module_name] = conv
            # 通过 add_module 将 conv 中的参数注册到当前模块中
            # 若不注册则不会作为权重放入GPU中更新参数
            self.add_module(module_name, conv)

    def forward(self, x):
        out = []
        for i in range(self.mapping.size(1)):
            # .nonzero 返回矩阵中非零元素的索引的张量
            # squeeze,去掉维数为1的维度
            # in_channels是mapping中1的index
            index_channels = self.mapping[:, i].nonzero().squeeze()
            in_tensors = x.index_select(1, index_channels)
            conv_out = self.convs['conv{}'.format(i)](in_tensors)
            out.append(conv_out)

        return torch.cat(out, 1)


class RBF(nn.Module):
    def __init__(self, in_features, out_features, init_weight=None):
        super(RBF, self).__init__()
        ## register_buffer 在内存中定义一个常量，不会被optimizer更新
        if init_weight is not None:
            self.register_buffer('weight', torch.tensor(init_weight))
        else:
            self.register_buffer('weight', torch.rand(in_features, out_features))
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = (x - self.weight).pow(2).sum(-2)
        return x

import numpy as np

_zero = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, +1, +1, -1] + \
        [-1, -1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_one = [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, -1, -1, +1, +1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_two = [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, -1, -1, -1, -1, +1, +1] + \
       [-1, -1, -1, -1, +1, +1, -1] + \
       [-1, -1, +1, +1, +1, -1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, +1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_three = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_four = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [-1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, +1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1]

_five = [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [+1, +1, +1, +1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [+1, +1, -1, -1, -1, -1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

_six = [-1, -1, +1, +1, +1, +1, -1] + \
       [-1, +1, +1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, -1, -1, -1, -1, -1] + \
       [+1, +1, +1, +1, +1, +1, -1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, -1, -1, -1, +1, +1] + \
       [+1, +1, +1, -1, -1, +1, +1] + \
       [-1, +1, +1, +1, +1, +1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1] + \
       [-1, -1, -1, -1, -1, -1, -1]

_seven = [+1, +1, +1, +1, +1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, -1, +1, +1] + \
         [-1, -1, -1, -1, +1, +1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, -1, +1, +1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, +1, +1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_eight = [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [+1, +1, -1, -1, -1, +1, +1] + \
         [-1, +1, +1, +1, +1, +1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1] + \
         [-1, -1, -1, -1, -1, -1, -1]

_nine = [-1, +1, +1, +1, +1, +1, -1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, -1, +1, +1] + \
        [+1, +1, -1, -1, +1, +1, +1] + \
        [-1, +1, +1, +1, +1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, -1, +1, +1] + \
        [-1, -1, -1, -1, +1, +1, -1] + \
        [-1, +1, +1, +1, +1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1] + \
        [-1, -1, -1, -1, -1, -1, -1]

# 84 x 10
RBF_WEIGHT = np.array([_zero, _one, _two, _three, _four, _five, _six, _seven, _eight, _nine]).transpose()


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.samp2 = Downsampling2d(6, (2, 2))
        self.conv3 = DropoutConv2d(6, 16, 5)
        self.samp4 = Downsampling2d(16, (2, 2))
        self.conv5 = nn.Conv2d(16, 120, 5)
        self.fc6   = nn.Linear(120, 84)
        self.output= RBF(84, 10,init_weight= RBF_WEIGHT)
        self.active= nn.Tanh()
    
    def forward(self, x):
        x = self.active(self.conv1(x))
        x = self.active(self.samp2(x))
        x = self.active(self.conv3(x))
        x = self.active(self.samp4(x))
        x = self.active(self.conv5(x))
        x = torch.squeeze(x)
        x = self.active(self.fc6(x))
        x = self.output(x)
        return x

from torchsummary import summary
net = LeNet().cuda()
summary(net, (1,32,32))

#############################输出#################################
----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1            [-1, 6, 28, 28]             156
#               Tanh-2            [-1, 6, 28, 28]               0
#          AvgPool2d-3            [-1, 6, 14, 14]               0
#     Downsampling2d-4            [-1, 6, 14, 14]               6
#               Tanh-5            [-1, 6, 14, 14]               0
#             Conv2d-6            [-1, 1, 10, 10]              76
#             Conv2d-7            [-1, 1, 10, 10]              76
#             Conv2d-8            [-1, 1, 10, 10]              76
#             Conv2d-9            [-1, 1, 10, 10]              76
#            Conv2d-10            [-1, 1, 10, 10]              76
#            Conv2d-11            [-1, 1, 10, 10]              76
#            Conv2d-12            [-1, 1, 10, 10]             101
#            Conv2d-13            [-1, 1, 10, 10]             101
#            Conv2d-14            [-1, 1, 10, 10]             101
#            Conv2d-15            [-1, 1, 10, 10]             101
#            Conv2d-16            [-1, 1, 10, 10]             101
#            Conv2d-17            [-1, 1, 10, 10]             101
#            Conv2d-18            [-1, 1, 10, 10]             101
#            Conv2d-19            [-1, 1, 10, 10]             101
#            Conv2d-20            [-1, 1, 10, 10]             101
#            Conv2d-21            [-1, 1, 10, 10]             151
#     DropoutConv2d-22           [-1, 16, 10, 10]               0
#              Tanh-23           [-1, 16, 10, 10]               0
#         AvgPool2d-24             [-1, 16, 5, 5]               0
#    Downsampling2d-25             [-1, 16, 5, 5]              16
#              Tanh-26             [-1, 16, 5, 5]               0
#            Conv2d-27            [-1, 120, 1, 1]          48,120
#              Tanh-28            [-1, 120, 1, 1]               0
#            Linear-29                   [-1, 84]          10,164
#              Tanh-30                   [-1, 84]               0
#               RBF-31                   [-1, 10]             840
# ================================================================
# Total params: 60,818
# Trainable params: 59,956
# Non-trainable params: 862
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.15
# Params size (MB): 0.23
# Estimated Total Size (MB): 0.38
# ----------------------------------------------------------------
```








