+++

title = "YOLO 中的数据增强"

date = 2023-11-30

[taxonomies]

categories = ["2023"]

tags = ["yolo", "data"]

+++

## mixup

将两张图像以不同的透明度叠放成一张图像，计算损失时 GT 也要乘以相应的透明度

## Mosaic

将四张图片边贴边拼成一张图片

