---
title: "计算机图形学——实验三"
date: 2020-12-09T12:57:41+08:00
draft: true
tags: [
  "opencv",
  "opengl",
]
categories: [
  "作业与考试", 
]
---
河南理工大学计算机图形学上机实验——实验三
<!--more-->
## 题1

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 400, winHeight =400 ; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 0.0, yy = 0.0, z0 = 5.0;	   //设置观察坐标系原点 
GLfloat xref = 0.0, yref = 0.0, zref = 0.0;	//设置观察坐标系参考点（视点） 
GLfloat Vx = 0.0, Vy = 1.0, Vz = 0.0;	   //设置观察坐标系向上向量（y轴） 

/*观察体参数设置 */
GLfloat xwMin = -1.0, ywMin = -1.0, xwMax = 1.0, ywMax = 1.0;//设置裁剪窗口坐标范围
GLfloat dnear = 1.5, dfar = 20.0;	      //设置远、近裁剪面深度范围

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    /*观察变换*/
    gluLookAt(x0, yy, z0, xref, yref, zref, Vx, Vy, Vz);        //指定三维观察参数

    // 请在此添加你的代码
    /********** Begin ********/
    glPushMatrix();
    glTranslatef(-2.0f,0.0f,0.0f);
    glColor3f (0.0, 0.0, 1.0);	
    glutSolidCube (1.0);	//绘制单位立方体实体
    glPopMatrix();

    glPushMatrix();
    glColor3f (1.0, 0.0, 0.0);	//设置前景色为
    // glLineWidth (1.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    glPushMatrix();
    glTranslatef(2.0f,0.0f,0.0f);
    glColor3f (0.0, 1.0, 0.0);	//设置前景色为
    glLineWidth (2.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    /********** End **********/
    glFlush();
}

void reshape(GLint newWidth, GLint newHeight)
{
    /*视口变换*/
    glViewport(0, 0, newWidth, newHeight);	//定义视口大小

    /*投影变换*/
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    /*透视投影，设置透视观察体*/
	glFrustum(xwMin, xwMax, ywMin, ywMax, dnear, dfar);

    /*模型变换*/
    glMatrixMode(GL_MODELVIEW);
    
    winWidth = newWidth;
    winHeight = newHeight;
}
int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 400 , 400 );        //设置初始化窗口大小
    glutCreateWindow("三维观察");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
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

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 400; j++) {
            int k = 3 * (i * 400 + j);
            plane2Ptr[j] = pPixelData[k];
            plane1Ptr[j] = pPixelData[k + 1];
            plane0Ptr[j] = pPixelData[k + 2];
        }
    }
    cv::merge(imgPlanes, img);
    cv::flip(img, img, 0);
    cv::namedWindow("openglGrab");
    cv::imshow("openglGrab", img);
    //cv::waitKey();
    cv::imwrite("../img_step1/test.jpg", img);
    return 0;
}
```
## 题2

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 400, winHeight =400 ; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 0.0, yy = 0.0, z0 = 5.0;	   //设置观察坐标系原点 
GLfloat xref = 0.0, yref = 0.0, zref = 0.0;	//设置观察坐标系参考点（视点） 
GLfloat Vx = 0.0, Vy = 1.0, Vz = 0.0;	   //设置观察坐标系向上向量（y轴） 

/*观察体参数设置 */
GLfloat xwMin = -1.0, ywMin = -1.0, xwMax = 1.0, ywMax = 1.0;//设置裁剪窗口坐标范围
GLfloat dnear = 1.5, dfar = 20.0;	      //设置远、近裁剪面深度范围

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    /*观察变换*/
     gluLookAt(1.0, 1.5, 8.0, 0, 0, 0, 0, 1, 0);        //指定三维观察参数

    // 请在此添加你的代码
    /********** Begin ********/
    glPushMatrix();
    glTranslatef(-2.0f,0.0f,0.0f);
    glColor3f (0.0, 0.0, 1.0);	
    glutSolidCube (1.0);	//绘制单位立方体实体
    glPopMatrix();

    glPushMatrix();
    glColor3f (1.0, 0.0, 0.0);	//设置前景色为
    // glLineWidth (1.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    glPushMatrix();
    glTranslatef(2.0f,0.0f,0.0f);
    glColor3f (0.0, 1.0, 0.0);	//设置前景色为
    glLineWidth (2.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();
    /********** End **********/
    glFlush();
}

void reshape(GLint newWidth, GLint newHeight)
{
    /*视口变换*/
    glViewport(0, 0, newWidth, newHeight);	//定义视口大小

    /*投影变换*/
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    /*透视投影，设置透视观察体*/
	 gluPerspective(45,1,1,100);

    /*模型变换*/
    glMatrixMode(GL_MODELVIEW);
    
    winWidth = newWidth;
    winHeight = newHeight;
}
int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 400 , 400 );        //设置初始化窗口大小
    glutCreateWindow("三维观察");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
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

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 400; j++) {
            int k = 3 * (i * 400 + j);
            plane2Ptr[j] = pPixelData[k];
            plane1Ptr[j] = pPixelData[k + 1];
            plane0Ptr[j] = pPixelData[k + 2];
        }
    }
    cv::merge(imgPlanes, img);
    cv::flip(img, img, 0);
    cv::namedWindow("openglGrab");
    cv::imshow("openglGrab", img);
    //cv::waitKey();
    cv::imwrite("../img_step2/test.jpg", img);
    return 0;
}
```

## 题3

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 400, winHeight =400 ; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 0.0, yy = 0.0, z0 = 5.0;	   //设置观察坐标系原点 
GLfloat xref = 0.0, yref = 0.0, zref = 0.0;	//设置观察坐标系参考点（视点） 
GLfloat Vx = 0.0, Vy = 1.0, Vz = 0.0;	   //设置观察坐标系向上向量（y轴） 

/*观察体参数设置 */
GLfloat xwMin = -1.0, ywMin = -1.0, xwMax = 1.0, ywMax = 1.0;//设置裁剪窗口坐标范围
GLfloat dnear = 1.5, dfar = 20.0;	      //设置远、近裁剪面深度范围

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    /*观察变换*/
    gluLookAt(x0, yy, z0, xref, yref, zref, Vx, Vy, Vz);        //指定三维观察参数

    // 请在此添加你的代码
    /********** Begin ********/
    glPushMatrix();
    glTranslatef(-2.0f,0.0f,0.0f);
    glColor3f (0.0, 0.0, 1.0);	
    glutSolidCube (1.0);	//绘制单位立方体实体
    glPopMatrix();

    glPushMatrix();
    glColor3f (1.0, 0.0, 0.0);	//设置前景色为
    // glLineWidth (1.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    glPushMatrix();
    glTranslatef(2.0f,0.0f,0.0f);
    glRotatef(30.0,1.0,0.0,0.0);
    glColor3f (0.0, 1.0, 0.0);	//设置前景色为
    glLineWidth (2.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    /********** End **********/
    glFlush();
}

void reshape(GLint newWidth, GLint newHeight)
{
    /*视口变换*/
    glViewport(0, 0, newWidth, newHeight);	//定义视口大小

    /*投影变换*/
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    /*透视投影，设置透视观察体*/
	glFrustum(xwMin, xwMax, ywMin, ywMax, dnear, dfar);

    /*模型变换*/
    glMatrixMode(GL_MODELVIEW);
    
    winWidth = newWidth;
    winHeight = newHeight;
}
int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 400 , 400 );        //设置初始化窗口大小
    glutCreateWindow("三维观察");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
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

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 400; j++) {
            int k = 3 * (i * 400 + j);
            plane2Ptr[j] = pPixelData[k];
            plane1Ptr[j] = pPixelData[k + 1];
            plane0Ptr[j] = pPixelData[k + 2];
        }
    }
    cv::merge(imgPlanes, img);
    cv::flip(img, img, 0);
    cv::namedWindow("openglGrab");
    cv::imshow("openglGrab", img);
    //cv::waitKey();
    cv::imwrite("../img_step4/test.jpg", img);
    return 0;
}
```
## 题4

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 400, winHeight =400 ; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 0.0, yy = 0.0, z0 = 5.0;	   //设置观察坐标系原点 
GLfloat xref = 0.0, yref = 0.0, zref = 0.0;	//设置观察坐标系参考点（视点） 
GLfloat Vx = 0.0, Vy = 1.0, Vz = 0.0;	   //设置观察坐标系向上向量（y轴） 

/*观察体参数设置 */
GLfloat xwMin = -1.0, ywMin = -1.0, xwMax = 1.0, ywMax = 1.0;//设置裁剪窗口坐标范围
GLfloat dnear = 1.5, dfar = 20.0;	      //设置远、近裁剪面深度范围

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    /*观察变换*/
    gluLookAt(x0, yy, z0, xref, yref, zref, Vx, Vy, Vz);        //指定三维观察参数

    // 请在此添加你的代码
    /********** Begin ********/
    glPushMatrix();
    glTranslatef(-2.0f,0.0f,0.0f);
    glColor3f (0.0, 0.0, 1.0);	
    glutSolidCube (1.0);	//绘制单位立方体实体
    glPopMatrix();

    glPushMatrix();
    glColor3f (1.0, 0.0, 0.0);	//设置前景色为
    // glLineWidth (1.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    glPushMatrix();
    glTranslatef(2.0f,0.0f,0.0f);
    glColor3f (0.0, 1.0, 0.0);	//设置前景色为
    glLineWidth (2.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();


    /********** End **********/
    glFlush();
}

void reshape(GLint newWidth, GLint newHeight)
{
    /*视口变换*/
    glViewport(0, 0, newWidth, newHeight);	//定义视口大小

    /*投影变换*/
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    /*平行投影*/
    glOrtho( -3.0, 3.0, -3.0, 3.0,-100.0 , 100.0);

    /*模型变换*/
    glMatrixMode(GL_MODELVIEW);
    
    winWidth = newWidth;
    winHeight = newHeight;
}
int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 400 , 400 );        //设置初始化窗口大小
    glutCreateWindow("三维观察");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
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

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 400; j++) {
            int k = 3 * (i * 400 + j);
            plane2Ptr[j] = pPixelData[k];
            plane1Ptr[j] = pPixelData[k + 1];
            plane0Ptr[j] = pPixelData[k + 2];
        }
    }
    cv::merge(imgPlanes, img);
    cv::flip(img, img, 0);
    cv::namedWindow("openglGrab");
    cv::imshow("openglGrab", img);
    //cv::waitKey();
    cv::imwrite("../img_step3/test.jpg", img);
    return 0;
}
```
## 题5

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 800, winHeight = 400; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 0.0, yy = 0.0, z0 = 5.0;	   //设置观察坐标系原点 
GLfloat xref = 0.0, yref = 0.0, zref = 0.0;	//设置观察坐标系参考点（视点） 
GLfloat Vx = 0.0, Vy = 1.0, Vz = 0.0;	   //设置观察坐标系向上向量（y轴） 

/*观察体参数设置 */
GLfloat xwMin = -1.0, ywMin = -1.0, xwMax = 1.0, ywMax = 1.0;//设置裁剪窗口坐标范围
GLfloat dnear = 1.5, dfar = 20.0;	      //设置远、近裁剪面深度范围

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    /*观察变换*/
    gluLookAt(x0, yy, z0, xref, yref, zref, Vx, Vy, Vz);        //指定三维观察参数

    // 请在此添加你的代码
    /********** Begin ********/

    glPushMatrix();
    glTranslatef(-2.0f,0.0f,0.0f);
    glColor3f (0.0, 0.0, 1.0);	
    glutSolidCube (1.0);	//绘制单位立方体实体
    glPopMatrix();

    glPushMatrix();
    glColor3f (1.0, 0.0, 0.0);	//设置前景色为
    // glLineWidth (1.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();

    glPushMatrix();
    glTranslatef(2.0f,0.0f,0.0f);
    glColor3f (0.0, 1.0, 0.0);	//设置前景色为
    glLineWidth (2.0);         	//设置线宽
    glutWireCube (1.0);	//绘制单位立方体线框
    glPopMatrix();






    /********** End **********/
    glFlush();
}

void reshape(GLint newWidth, GLint newHeight)
{
    /*视口变换*/
    glViewport(0, 0, newWidth, newHeight);	//定义视口大小

    /*投影变换*/
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    /*透视投影，设置透视观察体*/
    gluPerspective( 45,2 ,1 ,100 );

    /*模型变换*/
    glMatrixMode(GL_MODELVIEW);
    
    winWidth = newWidth;
    winHeight = newHeight;
}
int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(  800, 400 );        //设置初始化窗口大小
    glutCreateWindow("三维观察");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoopEvent();





    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);
    cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 800, CV_8UC3);
    cv::split(img, imgPlanes);

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 800; j++) {
            int k = 3 * (i * 800 + j);
            plane2Ptr[j] = pPixelData[k];
            plane1Ptr[j] = pPixelData[k + 1];
            plane0Ptr[j] = pPixelData[k + 2];
        }
    }
    cv::merge(imgPlanes, img);
    cv::flip(img, img, 0);
    cv::namedWindow("openglGrab");
    cv::imshow("openglGrab", img);
    //cv::waitKey();
    cv::imwrite("../img_step5/test.jpg", img);
    return 0;
}
```

