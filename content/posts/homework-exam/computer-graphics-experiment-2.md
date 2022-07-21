---
title: "计算机图形学——实验二"
date: 2020-12-02T12:57:41+08:00
draft: true
tags: [
  "opencv",
  "opengl",
]
categories: [
  "作业与考试", 
]
---

河南理工大学计算机图形学上机实验——实验二
<!--more-->

## 实验一
```c
// 提示：写完代码请保存之后再进行评测
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

void init(void)
{  
    glClearColor(0.0, 0.0, 0.0, 0.0);       //设置背景颜色
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-5.0, 5.0, -5.0, 5.0);       //设置显示的范围是X:-5.0~5.0, Y:-5.0~5.0
	glMatrixMode(GL_MODELVIEW);
}

void drawSquare(void)						//绘制中心在原点，边长为2的正方形
{
   // 请在此添加你的代码
   /********** Begin ********/
    glBegin(GL_POLYGON);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f(1.0f, -1.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
    
   /********** End **********/
}

void myDraw(void)                           //二维几何变换
{
   // 请在此添加你的代码
   /********** Begin ********/
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(0.0f, 2.0f, 0.0f);
    glScalef(3.0, 0.5, 1.0);
    glColor3f(1.0, 1.0, 1.0);
    drawSquare();
    glPopMatrix();

    glPushMatrix();
	glRotatef(0.0,0.0,0.0,1.0);
	glColor3f (1.0, 0.0, 0.0);  
	drawSquare();              				//中间右菱形
	glPopMatrix();


   /********** End **********/  						
	glFlush();
}

int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(400 * 400 * 3);//分配内存
   GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(400, 400);
	glutCreateWindow("几何变换示例");
    
    init();
	 glutDisplayFunc(&myDraw);
    glutMainLoopEvent();     
     
     
    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

	 cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 400, CV_8UC3);
    cv::split(img, imgPlanes);
 
        for(int i = 0; i < 400; i ++) {
            unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
            unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
            unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
            for(int j = 0; j < 400; j ++) {
                int k = 3 * (i * 400 + j);
                plane2Ptr[j] = pPixelData[k];
                plane1Ptr[j] = pPixelData[k+1];
                plane0Ptr[j] = pPixelData[k+2];
            }
        }
        cv::merge(imgPlanes, img);
        cv::flip(img, img ,0); 
        cv::namedWindow("openglGrab");
        cv::imshow("openglGrab", img);
        //cv::waitKey();
        cv::imwrite("../img_step1/test.jpg", img);
	return 0;
}

```
![结果图](https://img-blog.csdnimg.cn/20201202113146956.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center "result")

## 实验二

```c
// 提示：写完代码请保存之后再进行评测
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

void init(void)
{  
    glClearColor(0.0, 0.0, 0.0, 0.0);       //设置背景颜色
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-5.0, 5.0, -5.0, 5.0);       //设置显示的范围是X:-5.0~5.0, Y:-5.0~5.0
	glMatrixMode(GL_MODELVIEW);
}
void drawSquare(void)						//绘制中心在原点，边长为2的正方形
{
   // 请在此添加你的代码
   /********** Begin ********/
    glBegin (GL_POLYGON);					//顶点指定需要按逆时针方向
	   glVertex2f (-1.0f,-1.0f);			//左下点
	   glVertex2f (1.0f,-1.0f);				//右下点
	   glVertex2f (1.0f, 1.0f);				//右上点
	   glVertex2f (-1.0f,1.0f);				//左上点
	glEnd ( );


   /********** End **********/
}

void myDraw(void)                           //二维几何变换
{
   // 请在此添加你的代码
   /********** Begin ********/
    glClear (GL_COLOR_BUFFER_BIT);			//清空
	glLoadIdentity();     

 glTranslatef(-3.0,0.0,0.0);  
	
	glPushMatrix();
	glRotatef(45.0,0.0,0.0,1.0);
	glColor3f (0.0, 1.0, 0.0);  
	drawSquare();              				//中间左菱形
	glPopMatrix();
   
   	glTranslatef(3.0,0.0,0.0); 
    
	glPushMatrix();
	glRotatef(0.0,0.0,0.0,1.0);
	glColor3f (1.0, 0.0, 0.0);  
	drawSquare();              				//中间中菱形
	glPopMatrix();

	glTranslatef(3.0,0.0,0.0); 
    
	glPushMatrix();
	glRotatef(45.0,0.0,0.0,1.0);
	glColor3f (0.0, 0.7, 0.0);  
	drawSquare();              				//中间右菱形
	glPopMatrix();





   /********** End **********/  						
	glFlush();
}

int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(400 * 400 * 3);//分配内存
   GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(400, 400);
	glutCreateWindow("几何变换示例");
    
    init();
	 glutDisplayFunc(&myDraw);
    glutMainLoopEvent();     
     
     
    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

	 cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 400, CV_8UC3);
    cv::split(img, imgPlanes);
 
        for(int i = 0; i < 400; i ++) {
            unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
            unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
            unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
            for(int j = 0; j < 400; j ++) {
                int k = 3 * (i * 400 + j);
                plane2Ptr[j] = pPixelData[k];
                plane1Ptr[j] = pPixelData[k+1];
                plane0Ptr[j] = pPixelData[k+2];
            }
        }
        cv::merge(imgPlanes, img);
        cv::flip(img, img ,0); 
        cv::namedWindow("openglGrab");
        cv::imshow("openglGrab", img);
        //cv::waitKey();
        cv::imwrite("../img_step2/test.jpg", img);
	return 0;
}
```
![实验二](https://img-blog.csdnimg.cn/20201202113206997.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center "exp 2")

## 实验三

```c
// 提示：写完代码请保存之后再进行评测
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

void init(void)
{  
    glClearColor(0.0, 0.0, 0.0, 0.0);       //设置背景颜色
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-5.0, 5.0, -5.0, 5.0);       //设置显示的范围是X:-5.0~5.0, Y:-5.0~5.0
	glMatrixMode(GL_MODELVIEW);
}
void drawSquare(void)						//绘制中心在原点，边长为2的正方形
{
   // 请在此添加你的代码
   /********** Begin ********/
 
glBegin (GL_POLYGON);					//顶点指定需要按逆时针方向
	   glVertex2f (-1.0f,-1.0f);			//左下点
	   glVertex2f (1.0f,-1.0f);				//右下点
	   glVertex2f (1.0f, 1.0f);				//右上点
	   glVertex2f (-1.0f,1.0f);				//左上点
	glEnd ( );

   /********** End **********/
}

void myDraw(void)                           //二维几何变换
{
   // 请在此添加你的代码
   /********** Begin ********/
glClear (GL_COLOR_BUFFER_BIT);			//清空
	glLoadIdentity();       					//将当前矩阵设为单位矩阵
	
	glPushMatrix();
	glTranslatef(0.0f,2.0f,0.0f);
	glScalef(3.0,0.5,1.0); 
	glColor3f (1.0, 1.0, 1.0); 
	drawSquare();      						//上面红色矩形
	glPopMatrix();

	glPushMatrix();
	
	glTranslatef(-3.0,0.0,0.0);  
	
	glPushMatrix();
	glRotatef(45.0,0.0,0.0,1.0);
	glColor3f (0.0, 1.0, 0.0);  
	drawSquare();              				//中间左菱形
	glPopMatrix();
   
   	glTranslatef(3.0,0.0,0.0); 
    
	glPushMatrix();
	glRotatef(0.0,0.0,0.0,1.0);
	glColor3f (1.0, 0.0, 0.0);  
	drawSquare();              				//中间中菱形
	glPopMatrix();

	glTranslatef(3.0,0.0,0.0); 
    
	glPushMatrix();
	glRotatef(45.0,0.0,0.0,1.0);
	glColor3f (0.0, 0.7, 0.0);  
	drawSquare();              				//中间右菱形
	glPopMatrix();
    
	glPopMatrix();

	glTranslatef(0.0,-3.0,0.0);  
	glScalef(4.0,1.5,1.0); 
	glColor3f (0.0, 0.0, 1.0);
	drawSquare(); 




   /********** End **********/  						
	glFlush();
}

int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(400 * 400 * 3);//分配内存
   GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(400, 400);
	glutCreateWindow("几何变换示例");
    
    init();
	 glutDisplayFunc(&myDraw);
    glutMainLoopEvent();     
     
     
    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

	 cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 400, CV_8UC3);
    cv::split(img, imgPlanes);
 
        for(int i = 0; i < 400; i ++) {
            unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
            unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
            unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
            for(int j = 0; j < 400; j ++) {
                int k = 3 * (i * 400 + j);
                plane2Ptr[j] = pPixelData[k];
                plane1Ptr[j] = pPixelData[k+1];
                plane0Ptr[j] = pPixelData[k+2];
            }
        }
        cv::merge(imgPlanes, img);
        cv::flip(img, img ,0); 
        cv::namedWindow("openglGrab");
        cv::imshow("openglGrab", img);
        //cv::waitKey();
        cv::imwrite("../img_step3/test.jpg", img);
	return 0;
}

```
![experment 3](https://img-blog.csdnimg.cn/20201202113223128.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center "experment 3")

## 实验四

```c
// 提示：写完代码请保存之后再进行评测
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

void init(void)
{  
    glClearColor(0.0, 0.0, 0.0, 0.0);       //设置背景颜色
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-5.0, 5.0, -5.0, 5.0);       //设置显示的范围是X:-5.0~5.0, Y:-5.0~5.0
	glMatrixMode(GL_MODELVIEW);
}
void drawDiamond(void)						//绘制一个菱形
{
   // 请在此添加你的代码
   /********** Begin ********/
	glBegin (GL_POLYGON);					//顶点指定需要按逆时针方向
	   glVertex2f (0.0f,0.0f);			//左下点
	   glVertex2f (1.0f,2.0f);				//右下点
	   glVertex2f (0.0f, 4.0f);				//右上点
	   glVertex2f (-1.0f,2.0f);				//左上点
	glEnd ( );


   /********** End **********/
}

void myDraw(void)                           //二维几何变换
{
   // 请在此添加你的代码
   /********** Begin ********/
    glClear (GL_COLOR_BUFFER_BIT);			//清空
	glLoadIdentity();       					//将当前矩阵设为单位矩阵
	
	glTranslatef(0.0,0.0,0.0);  
	
	glPushMatrix();
	glRotatef(0.0,0.0,0.0,1.0);
	glColor3f (1.0, 0.0, 0.0);  
	drawDiamond();              				//中间左菱形
	glPopMatrix();

    glTranslatef(0.0,0.0,0.0);  
	
	glPushMatrix();
	glRotatef(120,0.0,0.0,1.0);
	glColor3f (0.0, 1.0, 0.0);  
	drawDiamond();              				//中间左菱形
	glPopMatrix();

    glTranslatef(0.0,0.0,0.0); 
    glPushMatrix();
	glRotatef(240,0.0,0.0,1.0);
	glColor3f (0.0, 0.0, 1.0);  
	drawDiamond();              				//中间左菱形
	glPopMatrix();





   /********** End **********/  						
	glFlush();
}

int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(400 * 400 * 3);//分配内存
    GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(400, 400);
	glutCreateWindow("几何变换示例");
    
    init();
	glutDisplayFunc(&myDraw);
    glutMainLoopEvent();     
     
     
    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

	 cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 400, CV_8UC3);
    cv::split(img, imgPlanes);
 
        for(int i = 0; i < 400; i ++) {
            unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
            unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
            unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
            for(int j = 0; j < 400; j ++) {
                int k = 3 * (i * 400 + j);
                plane2Ptr[j] = pPixelData[k];
                plane1Ptr[j] = pPixelData[k+1];
                plane0Ptr[j] = pPixelData[k+2];
            }
        }
        cv::merge(imgPlanes, img);
        cv::flip(img, img ,0); 
        cv::namedWindow("openglGrab");
        cv::imshow("openglGrab", img);
        //cv::waitKey();
        cv::imwrite("../img_step4/test.jpg", img);
	return 0;
}
```
![experment4](https://img-blog.csdnimg.cn/20201202113239995.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDA0OTY5Mw==,size_16,color_FFFFFF,t_70#pic_center "experment 4")

