---
title: "图像分割任务标签图片的保存格式问题"
date: 2020-12-25T12:57:41+08:00
draft: false
tags: [
  "数据预处理",
]
categories: [
  "深度学习", 
]
---
处理深度学习图像分割任务时，标签文件格式一定要保存为png格式，否则原标签信息会被破坏
<!--more-->
## 起因
在做图像分割任务时，用到了cityscapes数据集，这些数据的标签文件是用json格式存储的，无法直接用于训练。于是我尝试将json信息转换成图片格式（jpg），但是图片中总有一些意想不到的像素值。我们知道标签图像的像素值就是原图上对应位置物体的类别，这就等于说出现了未知类别。可是反反复复查看了代码没有发现任何问题。正当我对着文件列表发呆时，突然注意到我的图片格式与往常看到的图像分割标签文件格式不太一样，往常的都是用PNG格式，然而我生成的是用JPG，于是我尝试将图片保存格式改为PNG，谁知问题就这么迎刃而解。

## JPG和PNG格式的区别
### JPG
> **JPG（92年) **：使用一种**失真压缩**方法，源图片保存为JPG格式后，同一位置上的像素值很可能发生改变，意味着标签类别发生了改变，这是在图像分割任务中绝不能允许发生的。
> 
> **优点** 　　
JPEG在色调及颜色平滑变化的相片或是写实绘画上可以达到它最佳的效果。在这种情况下,它通常比完全无失真方法作得更好,仍然可以产生非常好看的影像
> 
> **缺点** 　　
它并不适合于线条绘图（drawing）和其他文字或图示（iconic）的图形,因为它的压缩方法用在这些图形的型态上,会得到不适当的结果；

### PNG
> **PNG（96年）**：**格式是无损数据压缩**,PNG格式有8位、24位、32位三种形式,其中8位PNG支持两种不同的透明形式（索引透明和alpha透明）,24位PNG不支持透明,32位PNG在24位基础上增加了8位透明通道（32-24=8）,因此可展现256级透明程度。 
> 
> **优点** 　　
> * 支持256色调色板技术以产生小体积文件 　　
> * 最高支持48位真彩色图像以及16位灰度图像。 　　
> * 支持Alpha通道的半透明特性。 　
> * 支持图像亮度的gamma校正信息。 　　
> * 支持存储附加文本信息,以保留图像名称、作者、版权、创作时间、注释等信息。 　　
> * **使用无损压缩。** 　　
> * 渐近显示和流式读写,适合在网络传输中快速显示预览效果后再展示全貌。 　　
> * 使用CRC循环冗余编码防止文件出错。 　　
> * 最新的PNG标准允许在一个文件内存储多幅图像。 
> 
> **缺点** 　　
> * 但也有一些软件不能使用适合的预测,而造成过分臃肿的PNG文件。

## cityscapes数据集json转labelimg
```python
# 生成label图像
folder_label = '/content/data/label/'
folder_img = '/content/data/img/'
folder_path = '/content/data/'
file_list = os.listdir(folder_path)
label_dict = {'flat':0, 'human':1, 'vehicle':2, 'construction':3, 'object':4, 'nature':5, 'sky':6}
color = {'flat':(255,0,0), 'human':(255,128,0), 'vehicle':(255,255,0), 'construction':(0,255,0), 'object':(0,255,255), 'nature':(0,0,255), 'sky':(128,0,255)}
if len(os.listdir('/content/data/img')) <= 20:
  for i in tqdm(file_list):
      num = 0
      name, type_ = os.path.splitext(i)
      if type_ == '.json':
          file_path = os.path.join(folder_path, i)
          file = open(file_path).read()
          file = json.loads(file)
          imgHeight = file['imgHeight']
          imgWidth = file['imgWidth']
          # img = np.zeros((imgHeight, imgWidth, 3), np.uint8)  # 二十四位彩图
          img = np.zeros((imgHeight, imgWidth), np.uint8)  # 八位图
          img.fill(7)
          # print(img.shape)
          objects = file['objects']
          for j in objects:
              label = j['label']
              contours = np.array(j['polygon'])
              cv2.drawContours(img, [contours], -1,label_dict[str(label)], -1)
          save_path = os.path.join(folder_label, str(name) + '.png')
          cv2.imwrite(save_path, img)
```

