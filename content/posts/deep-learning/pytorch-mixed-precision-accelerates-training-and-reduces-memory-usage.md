---
title: "pytorch混合精度加速训练，减少显存占用"
date: 2020-10-28T12:57:41+08:00
draft: false
tags: [
  "pytorch",
  "cuda",
]
categories: [
  "深度学习", 
]

---
GPU显存太小，混合精度加速训练，减少显存占用
<!--more-->
## Pytorch 自带方法实现自动混合精度训练

```python
import torch
scaler = torch.cuda.amp.GradScaler()

for input, target in train_loader:  #取数据
        input = input.cuda()
        target = target.cuda()
        with torch.cuda.amp.autocast():  # 混合精度加速训练
            output = model(input)
            loss = criterion(output, target)
        optimizer.zero_grad()  # 重置梯度，不加会爆显存
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```
Pytorch在1.6版本的更新中实现了混合精度计算，可以直接调用。实测可以减少三分之一的训练时间和显存占用。 
修复头像问题