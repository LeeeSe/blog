---
title: "机器视觉——2019试卷"
date: 2020-12-08T12:57:41+08:00
draft: true
tags: [
  "opencv",
]
categories: [
  "作业与考试", 
]
---
河南理工大学机器视觉期末考试2019试卷及答案
<!--more-->
##  名词解释题

1. **数字图像**

   数字图像是指由被称作像素的小块区域组成的二维矩阵。将物理图像行列划分后，每个小块区域称为像素。

   每个像素包括两个属性：位置和亮度（或色彩）

2. **图像的灰度直方图**

   灰度直方图是关于灰度级分布的函数，是对图像中灰度级分布的统计。灰度直方图将数字图像中的所有像素，按照灰度值的大小，统计其出现的频率。直方图x轴表示0-255,256个像素值，y轴表示像素值的个数

3. **欧式距离，街区距离，棋盘距离**

   欧氏距离：平面或空间中两点的直线距离

   二维：

   ![img](https://img-blog.csdnimg.cn/img_convert/f39d74769a00ddd0fe193c8c432b7e7c.png)

   N维：

   ![img](https://img-blog.csdnimg.cn/img_convert/156180eece5ec0ee06db31eb0a2aed77.png)

   

   街区距离：两点在标准坐标系上的绝对轴距之和。<font color=red>（对于二维，街区距离就是两点沿x方向的距离和沿y方向的距离的和）</font>

   二维：

    ![img](https://img-blog.csdnimg.cn/img_convert/4fa2bd9fbbf349affd55fe5859204d14.png)
N维：

      ![img](https://img-blog.csdnimg.cn/img_convert/411cb295a9f83ce7edba9a69e66ff18b.png)
​		棋盘距离：两点间沿某标准轴的最大距离。<font color=red>（对于二维，棋盘距离就是两点沿x方向的距离和沿y方向的距离中最大的那个）</font>
二维：
![img](https://img-blog.csdnimg.cn/img_convert/057933952a03f74b141308d795f9090a.png)

	N维：
![img](https://img-blog.csdnimg.cn/img_convert/f43e546b7baf1910d686cd48711c935f.png)



4. **信息量与信息熵**

   信息量是随机事件X中某件事发生的概率的负对数，它是对信息多少的度量。一句话中的事发生的概率越大，则这句话的信息量越小（“明天太阳会升起”这句话的信息量就很小）。<font color=red>（体现在机器视觉中，它表示该符号所需的二进制位数。如序列aabbaccbaa，符号a出现的概率为0.5, 符号b出现的概率为0.3，符号c出现的概率为0.2，则a，b，c的信息量为1，1.737，2.322。总信息量：5* 1+3* 1.737+2* 2.322=14.855位，表示此字符串所需的总二进制位数）</font>（信息量单位：bit，对应底数为2）

   
   $$
   h(x) = - \log_2p( x)
   $$
   

   信息熵是随机事件X的信息量的期望

   

$$
H(X) = E(h(xi)) =-\sum^n_{i=1} p(xi)*\log_2p( xi)
$$

## 简答与证明题

1. **证明：二维离散灰度图像的均值与其`(傅里叶)`频谱的直流成分成线性关系**

   对于一副长宽为N*N的图像f（x，y）其傅里叶变换为(exp是以e为底的指数函数，j是复数符号，j^2=-1)：
   $$
   F(u,v)=\frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)exp[-j2π(ux+vy)/N]
   $$
   其频谱的直流成分为F(0,0)，得：
   $$
   F(0,0)=\frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)
   $$
   图像均值：
      $$
    MEAN=\frac{1}{N^2}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)
       $$
   则频谱的直流成分是二维离散灰度图像的均值的N倍，故二维离散灰度图像的均值与其`(傅里叶)`频谱的直流成分成线性关系
   

2. **高斯拉普拉斯算子是经典的边缘检测算子，叙述其实现过程并推导高斯拉普拉斯算子的形式**

   听说不考？？ 

3. **给出分段线性变换的定义并讨论线性变换的斜率参数取值对输出图像对比度的影响**

   定义：有时为了更好地调节图象的对比度，需要在一些亮度段拉伸，而在另一些亮度段压缩，这种变换称为分段线性变换。

   线性点运算的灰度变换函数形式可以采用线性方程描述，即：s=ar+b，则a为斜率，b为截距

   对a，b做以下讨论：

   ① 当a>1时，输出灰度扩展，图像对比度增大。

   ②当a=1时，输出灰度既不扩展也不压缩，图像对比度不变。

   ③当0<a<1时，输出灰度压缩，图像对比度减小。

   ④当a<0时，较亮的区域变暗，较暗的区域变亮。若a=-1图像发生反色变换。

4. **梯度是灰度图像边缘检测的重要工具，将梯度推广到向量梯度并构造彩色图像边缘检测方法。**

   听说也不考？？？

## 计算题

1. **给出Prewitt算子对应模板的公式，对如下图所示数据计算梯度向量的值（忽略最外侧的数据），并`近似`计算梯度的模。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221228224.png#pic_center)
​px =![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221257472.png#pic_center)

py =![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221356795.png#pic_center)
计算px,py梯度：如下图所示，px如滑动窗口般在矩阵上移动，每移动一次计算出一个数值，实际上计算出的梯度是针对px中心点的梯度，所以原矩阵最外层一圈的数值的梯度无法计算，故px计算过后得到的矩阵会比原矩阵小一圈。py计算过程与之相同。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221412930.gif#pic_center)
这里只说px第一个点的计算过程：
$$
-1* （218+217+211）+1*（207+199+189） = -51
$$




得到tx，ty矩阵（左三列tx，右三列ty）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221843127.png#pic_center)

                           

梯度的模(`题目中要求近似，故用绝对值相加，否则用平方加和再开方`)：
$$
M = |tx| + |ty|
$$
得矩阵：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208221752245.png#pic_center)




2.  **给出欧拉数的计算公式，并计算2018四个字母的欧拉数**
 <font color=red>(对于此题，我们暂不考虑联通类型)</font>
对于二维图像，欧拉数E = C - H，其中C为连接体数，H为孔洞数
对于数字2：连接体数为1，孔洞数为0，欧拉数为1
对于数字0：连接体数为1，孔洞数为1，欧拉数为0
对于数字1：连接体数为1，孔洞数为0，欧拉数为1
对于数字8：连接体数为1，孔洞数为2，欧拉数为-1
 故2018四个字母的欧拉数分别为1，0，1，-1
 <font color=red>对于连接体数，可以参考小写字母“j”，连接体分别为上方的点和下方的勾，故连接体数为2，孔洞为零。
 对于孔洞数，可以参考大写字母“B”，连接体只有一个就是B本身，但孔洞数为2。</font>
 
 

4.  **对于如下图所示区域边界的4向链码，归一化链码和差分链码（以A为起点）。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208222132568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208222413834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)


​	四向链码：0010 0033 3232 2212 2101

​	归一化链码： 对于闭合的边界，无论我们平移它，得到的链码都是一样的。但是如果选取的**起点**不同，那么得	到的链码也会有所不同。这时候我们就要将它**归一化**。原理就是把我们得到的链码看成是一个自然数，将链码	循环写下去，选取构成的**自然数数值**最小的那组链码。这个就是归一化链码(**选取长度与原链码相同**)。上图的链码循换      0010 0033 3232 2212 2101 0010 0033 3232 2212 2101 0010 0033 3232 2212 2101 归一化链码为0 0033 3232 2212 2101 001（这个自然数以三个0开头，最小）

​	差分链码：归一化链码解决了因为起点坐标不同而编码不同的问题，但仍有不足。如果我们将边界旋转，那么它的归一化链码也会发生变化。此时我们用差分链码来表示。差分链码的计算方法如图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208222438150.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)


4.  **给出从RGB彩色模型到HSI彩色模型的转换公式，并计算RGB颜色空间中（255,240,0）在HSI颜色模型中的取值（如需要，可以用开方，反余弦等函数来表示）**

   RGB转HSI公式：
   $$
   I = \frac{1}{3}(R+G+B)
   $$

   $$
   S=1-\frac{3}{(R+G+B)}[min(R,G,B)]
   $$

   $$
   H=\arccos{\frac{[(R-G)+(R-B)]/2}{[(R-G)^2+(R-B)*(G-B)]^{0.5}}}
   $$
将（255,240,0）归一化：
   $$
  r = R / (R+G+B)；g = G / (R+G+B)；b = B / (R+G+B)
   $$
   
   得（0.515，0.485，0）
   带入得HSI取值为（0.333，1，arccos(0.273/√0.501)）

   

## 程序设计题

```python
import cv2
import numpy as np

img = cv2.imread('img.png') # 以BGR读入图片 

cv2.imshow('src',img)
img = img.astype('float64')


kenelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])  # 水平梯度
kenely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])  # 垂直梯度

prewittx = cv2.filter2D(img,-1,kenelx)  # 二维卷积，计算垂直梯度
prewitty = cv2.filter2D(img,-1,kenely)  # 二维卷积，计算水平梯度
prewittimg = np.sqrt(np.square(prewittx) + np.square(prewitty))  # 计算梯度的模，绝对值加和是梯度模的近似

# 计算模后img的像素值分布不再是0-255，而可能超过了255，为了回到0-255，需要归一化
max, min = np.max(prewittimg), np.min(prewittimg)
newimg = (prewittimg-min)/(max-min)*255
newimg = 255 - newimg  # 反色操作
newimg = newimg.astype('uint8')

cv2.imshow('new_img',newimg)
cv2.waitKey(0)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208222843722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020120822291042.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)
## 三次作业
**作业二：把某地天气预报的内容看作一个信源，它有6种可能的天气：晴天（概率为0.30），阴天（概率为0.20），多云（概率为0.15），雨天（概率为0.13），大雾（概率为0.12）和下雪（概率为0.10）。如何用霍夫曼编码对其进行编码？平均码长分别是多少？**

编码步骤：先将所有概率从小到大排一行，然后找到其中最小的两个概率，从行中取出并相加，得到新概率，添加到行中得到新行，重复上述步骤直到最终结果为1。哈夫曼树则是从下往上构造，每找到两个最小节点，就当做子结点，并将两者的和当做父节点，从下往上构造，直到根节点1![在这里插入图片描述](https://img-blog.csdnimg.cn/20201210213752945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70)
得出几种天气的哈夫曼编码`（哈夫曼编码不唯一）`：
晴天（0.30）：11`（码长为2）`
阴天（0.20）：00
多云（0.15）：101 `（码长为3）`
雨天（0.13）：100
大雾（0.12）：011
下雪（0.10）：010

$$
ACL = \sum_{i=1}^{n}L_i*P_i
$$
ACL：平均码长，L：码长，P：概率
代入公式得：
$$ACL= (0.3+0.2)*2 + (0.15+0.13+0.12+0.10)*3=2.5
$$
平均码长为2.5

**作业一及作业三在计算题第一题Prewitt算子中考察过，Prewitt算子做会即可，同时要注意记忆Roberts和Sobel算子（P137）**

## 四次实验
[点我下载代码](https://wws.lanzoui.com/ikXFkj8sjhg)

二维码下载:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201210220145230.png#pic_center)
## 特别添加
**通道混合器代码**

```python
import cv2

img = cv2.imread('img.png')
cv2.imshow('src', img)

B = img[:,:,0]  # cv2读取出来的图像通道0,1,2分别对应BGR
G = img[:,:,1]  # 而非平时惯用的RGB
R = img[:,:,2]

# w1 w2 w3 加和最好不要大于1，否则可能使被混合的通道
# 像素值大于255造成溢出
w1 = 0.2
w2 = 0.3
w3 = 0.4

# 假设我们给通道R进行通道混合（你也可以选择其他通道）
R = w1*R + w2*G + w3*B  # 将RGB三通道的值分别乘以一个系数并加和，赋值给R
img[:,:,0] = B
img[:,:,1] = G
img[:,:,2] = R

cv2.imshow('img_mix', img)
cv2.waitKey(0)
```
**效果图**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201211162608253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center)

