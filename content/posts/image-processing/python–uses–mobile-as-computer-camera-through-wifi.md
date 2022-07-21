---
title: "Python通过wifi把手机做电脑摄像头"
date: 2020-04-22T12:57:41+08:00
draft: false
tags: [
  "opencv",
  "python",
]
categories: [
  "图像处理"
]
---
只需下载一个app，通过python把同一wifi（局域网）下的手机摄像头作为电脑摄像头

<!--more-->
## 下载app

蓝奏云十秒下载。
[点击下载（安卓）](https://ww.lanzoui.com/icwomsd)
ios同名软件需要18元，自行斟酌，可寻找同类软件替换。

## 获取rtsp地址

安装并打开app，切记`<u>`不要点击更新`</u>`。

### 第一步，打开IP摄像头服务器

![](https://img-blog.csdnimg.cn/20200522233544809.jpg)

### 第二步，获取账号密码

![](https://img-blog.csdnimg.cn/2020052223355961.jpg)

### 第三步，打开RTSP服务器

![](https://img-blog.csdnimg.cn/20200522233609462.jpg)

### 第四步，获取RTSP地址

![](https://img-blog.csdnimg.cn/20200522233639891.jpg)

### 第五步，将你的地址替换到代码中的url处

代码在下面，替换后点击运行即可。运行后手机端和电脑端同时有画面，但电脑端会有不到一秒的延迟，不影响使用。app界面右上角可以切换摄像头。

## 代码

> 注意：url中的 `admin：admin`不要删掉，它是你的账号密码，只替换后边的地址即可

```python
import cv2

url = 'rtsp://admin:admin@192.168.1.105:8554/live'
cap = cv2.VideoCapture(url)  # 带有摄像头的笔记本用户将url替换为 0 即可

while(cap.isOpened()):
    ret, frame = cap.read()  # frame为一帧图像，当frame为空时，ret返回false，否则为true
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # release the capture  
cv2.destroyAllWindows()
```
