---
title: "Python对两个list用相同方式打乱"
date: 2020-12-24T12:57:41+08:00
draft: false
tags: [
  "random",
  "python",
  "数据预处理",
]
categories: [
  "深度学习", 
]
---
很多情况下需要把两个相关联的list打乱，打乱时还不能破坏两个list中元素的相对顺序，可以通过设置随机种子来实现。

<!--more-->
## 对两个列表用相同的方式打乱
此处不可省略第二句random.seed(0)
```python
import random
import numpy as np

a = np.array([1,3,2,4,6,5,])
b = np.array([1,3,2,4,6,5,])
random.seed(0)
random.shuffle(a)
random.seed(0)
random.shuffle(b)

# 输出 [6 2 3 1 5 4] [6 2 3 1 5 4]
# 重复运行N次输出不变
```
若省略掉第二句random.seed(0）

```python
a = np.array([1,3,2,4,6,5,])
b = np.array([1,3,2,4,6,5,])
random.seed(0)
random.shuffle(a)
random.shuffle(b)

# 输出 [6 2 3 1 5 4] [6 1 3 5 2 4]
# 重复运行N次输出不变
```



