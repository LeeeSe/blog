---
title: "cv2.waitKey的入门级理解"
date: 2020-05-22T13:54:52+08:00
draft: false
tags: [
  "opencv",
  "python",
]
categories: [
  "图像处理"
]
---
最初用opencv处理图像时，大概查过 `cv2.waitKey`这个函数,当时查的迷迷糊糊的，只知道加上 `cv2.waitKey`之后 `cv2.imshow`就可以显示图像了。今天做视频逐帧截取时再次碰见了它，我盯着它想了半天也不知道这个函数有什么用，于是打开浏览器，一逛就是大半天。现在把我的收获及想法总结一下。

<!--more-->

## 为什么cv2.imshow之后要跟cv2.waitkey

我们先说说它的好兄弟 `cv2.imshow`。我们都知道imshow的作用是在GUI里显示一幅图像，但是它有个特点我们没有太注意，就是它的持续时间。看看下面的测试你就明白了。
{{< figure src="https://img-blog.csdnimg.cn/20200522021632628.gif">}}
{{< figure src="https://img-blog.csdnimg.cn/20200522021651424.gif">}}
{{< figure src="https://img-blog.csdnimg.cn/20200522021701222.gif">}}

实际上，waitkey控制着imshow的持续时间，当imshow之后不跟waitkey时，相当于没有给imshow提供时间展示图像，所以只有一个空窗口一闪而过。添加了waitkey后，哪怕仅仅是 `cv2.waitkey(1)`,我们也能截取到一帧的图像。所以 `cv2.imshow`后边是必须要跟 `cv2.waitkey`的。

给一段imshow源码里的注释来印证下：

> This function should be followed by cv::waitKey function which displays the image for specified . milliseconds. Otherwise, it won't display the image.

非官方翻译：这个函数之后应接cv2.waitKey函数来显示指定图像。否则，它不会显示图像。

## 为什么要这么麻烦的设计

来自官方的解释：

> This function is the only method in HighGUI that can fetch and handle events, so it needs to be .

非官方翻译：这个函数是HighGUI窗口中唯一的获取和处理事件的方法，因此它必须存在。

## cv2.waitKey(1000) & 0xFF == ord('q') 是什么意思

先解释下字面意思：

* `cv2.waitKey(1000)`：在1000ms内根据键盘输入返回一个值
* `0xFF` ：一个十六进制数
* `ord('q')` ：返回q的ascii码

`0xFF`是一个十六进制数，转换为二进制是11111111。waitKey返回值的范围为（0-255），刚好也是8个二进制位。那么我们将 `cv2.waitKey(1) & 0xFF`计算一下（不知怎么计算的可以百度位与运算）发现结果仍然是waitKey的返回值，那为何要多次一举呢？直接 `cv2.waitKey(1) == ord('q')`不就好了吗。

实际上在linux上使用waitkey有时会出现waitkey返回值超过了（0-255）的范围的现象。通过 `cv2.waitKey(1) & 0xFF`运算，当waitkey返回值正常时 `cv2.waitKey(1)` = `cv2.waitKey(1000) & 0xFF`,当返回值不正常时，`cv2.waitKey(1000) & 0xFF`的范围仍不超过（0-255），就避免了一些奇奇怪怪的BUG。

## cv2.waitkey和time.sleep的区别

肯定有人写在代码时把waitkey当sleep用过，你会发现有时waitkey并不起作用。

先来一段官方解释：

> The function only works if there is at least one HighGUI window created and the window is active`

非官方翻译：这个函数只有在至少一个HighGUI窗口存在的情况下才会起作用。

也就是说waitkey的延时机制是有条件的，必须在它之前创造HighGUI窗口它才会起作用。而time.sleep是无条件的延时机制。

那么 `cv2.waitKey`能不能代替 `time.sleep`在 `cv2.imshow`心中的地位呢？

{{< figure src="https://img-blog.csdnimg.cn/20200522021538269.gif">}}
{{< figure src="https://img-blog.csdnimg.cn/20200522021601696.gif">}}

答案很清楚：不能。
Mac开机运行sh文件