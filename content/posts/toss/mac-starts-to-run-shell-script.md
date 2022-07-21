---
title: "mac开机后台运行shell文件"
date: 2022-04-30T18:04:16+08:00
draft: false
tags: [
  "shell",
  "macos",
  "launchctl",
]
categories: [
  "折腾", 
]
---
天下苦百度云盘久矣，2021年阿里👨推出了量大不限速的阿里云盘后，我便将所有文件迁移到了阿里云盘中。然而阿里云盘的mac app使用体验并不是很好，我喜欢统一在文件管理器中管理文件，于是便另寻他路。终于在GitHub上发现了[aliyundrive-webdav](https://github.com/messense/aliyundrive-webdav)这个项目。然而每次开机都需要运行一次命令，很是麻烦，于是花了些时间探索mac如何开机后台运行shell文件。
<!--more-->
## 开机启动的两种途径
目前我了解到的mac开机运行文件有两种途径：
1. 用户与群组->登录项->添加新的应用程序
2. launchctl

方法一很简单，将包含有webdav服务启动命令的sh文件直接添加到启动项中即可。然而这种方法在开机时会自动启动终端，因为方法一的sh文件是以终端执行的方式运行的。我是有一些强迫症的，我需要的是安安静静地后台运行，不要丝毫影响到我。于是矛头便指向了方法二。

方法二稍微麻烦些，但它完美地满足了我的需求，我们接着往下看

## Launchctl
>launchctl是一个统一的服务管理框架，可以启动、停止和管理守护进程、应用程序、进程和脚本等。 launchctl是通过配置文件来指定执行周期和任务的。

launchctl管理的是plist文件，plist文件在macos中的分布如下：
- ~/Library/LaunchAgents 由用户自己定义的任务项 
- /Library/LaunchAgents 由管理员为用户定义的任务项 
- /Library/LaunchDaemons 由管理员定义的守护进程任务项 
- /System/Library/LaunchAgents 由Mac OS X为用户定义的任务项

> 说明:Agents文件夹下的plist是需要用户登录后，才会加载的，而Daemons文件夹下得plist是只要开机，可以不用登录就会被加载

### 常用命令

调试命令：
~~~shell
launchctl start xxx.plist  # 手动启动服务
launchctl stop xxx.plist  # 手动停止服务
~~~

加载/卸载

~~~shell
# 加载脚本，开机启动
launchctl load -w xxx.plist 

# 卸载脚本，开机不启动
launchctl unload -w xxx.plist 
~~~

> 注意：服务被修改后，需要重新加载才能生效

### 文件示例

文件com.aliyun.webdav.plist
~~~xml
<!-- 开机运行aliyun_webdav.sh文件 -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aliyun.webdav</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>/Users/ls/my_ssh/aliyun.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
~~~
各个参数的意义请参考[文档](https://www.launchd.info)

## 用法

如何使用写好的plist文件：

1. 根据需求将plist文件放入上述文件夹下（我放入了`~/Library/LaunchAgents`文件夹下）
2. 赋予shell文件可执行权限`chmod 755 xxx.sh`
3. `cd ~/Library/LaunchAgents`
4. `launchctl load -w com.aliyun.webdav.plist`

若要停止服务：

`launchctl unload -w com.aliyun.webdav.plist`

