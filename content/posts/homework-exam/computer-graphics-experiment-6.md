---
title: "计算机图形学——实验六"
date: 2020-12-30T12:57:41+08:00
draft: true
tags: [
  "opencv",
  "opengl",
]
categories: [
  "作业与考试", 
]
---
河南理工大学计算机图形学上机实验——实验六
<!--more-->
## 题一

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdlib.h>
#include<stdio.h>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束


#define checkImageWidth 64
#define checkImageHeight 64
static GLubyte checkImage[checkImageHeight][checkImageWidth][4];
static GLuint texName;

void makeCheckImage(void)
{
	int i, j, c;
	for (i = 0; i < checkImageHeight; i++)
	{
		for (j = 0; j < checkImageWidth; j++)
		{
			c = (((i & 0x8) == 0 ^ ((j & 0x8)) == 0)) * 255;
			checkImage[i][j][0] = (GLubyte)c;
			checkImage[i][j][1] = (GLubyte)c;
			checkImage[i][j][2] = (GLubyte)c;
			checkImage[i][j][3] = (GLubyte)255;
		}
	}
}

void init(void)
{
	glClearColor(0.5, 2.0, 0.5, 0.0);
	glShadeModel(GL_FLAT);
	glEnable(GL_DEPTH_TEST);

	makeCheckImage();
	glBindTexture(GL_TEXTURE_2D, texName);
    //********1.2.请输入代码设置控制纹理映射函数和纹理的定义函数********//

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA, checkImageWidth, checkImageHeight,0, GL_RGBA, GL_UNSIGNED_BYTE, checkImage);

   //********************************************************//              

}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
    //********3.请输入代码进行纹理映射方式设置*********//
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    //**********************************************//
	glBindTexture(GL_TEXTURE_2D, texName);
	glBegin(GL_QUADS);
    //********4.请输入代码进行纹理坐标设置**********//

    glTexCoord2f(0.0, 0.0); glVertex3f(-2.0, -1.0, 0.0);
    glTexCoord2f(0.0, 1.0); glVertex3f(-2.0, 1.0, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex3f(0.0, 1.0, 0.0);
    glTexCoord2f(1.0, 0.0); glVertex3f(0.0, -1.0, 0.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(1.0, -1.0, 0.0);
    glTexCoord2f(0.0, 1.0); glVertex3f(1.0, 1.0, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex3f(2.41421, 1.0, -1.41421);
    glTexCoord2f(1.0, 0.0); glVertex3f(2.41421, -1.0, -1.41421);

    //********************************************//
	glEnd();
	glFlush();
	glDisable(GL_TEXTURE_2D);
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, ((GLsizei)w / (GLsizei)h), 1.0, 30.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.6);
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

## 题二

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdlib.h>
#include<stdio.h>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束

#define stripeImageWidth 32
GLubyte stripeImage[4 * stripeImageWidth];


void makeStripeImage(void) 		//生成纹理
{
    
    int j;
    for (j = 0; j < stripeImageWidth; j++)
    {
        stripeImage[4 * j + 0] = (GLubyte)((j <= 4) ? 255 : 0);
        stripeImage[4 * j + 1] = (GLubyte)((j > 4) ? 255 : 0);
        stripeImage[4 * j + 2] = (GLubyte)0;
        stripeImage[4 * j + 3] = (GLubyte)255;
    }
}
// 平面纹理坐标生成
static GLfloat xequalzero[] = { 1.0, 1.0, 1.0, 1.0 };
static GLfloat slanted[] = { 1.0, 1.0, 1.0, 0.0 };
static GLfloat* currentCoeff;
static GLenum currentPlane;
static GLint currentGenMode;
static float roangles;
void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    makeStripeImage();
    //********1.2.请输入代码设置纹理映射控制函数和纹理的定义函数********//
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);    
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);   
    glTexImage1D(GL_TEXTURE_1D, 0, 4, stripeImageWidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, stripeImage);

    //*********************************************************//
    
    
    //********3.请输入代码进行纹理映射方式设置*********//
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);  

    //********************************************//
    currentCoeff = xequalzero;
    currentGenMode = GL_OBJECT_LINEAR;
    currentPlane = GL_OBJECT_PLANE;
    //****************4.自动纹理坐标生成功能函数******************//

    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE,  GL_OBJECT_LINEAR);       
    glTexGenfv(GL_S, currentPlane, currentCoeff);

    //*******************************************************//
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_1D);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    glMaterialf(GL_FRONT, GL_SHININESS, 64.0);
    roangles = 45.0f;
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    glRotatef(roangles, 0.0, 0.0, 1.0);
    glutSolidSphere(2.0, 32, 32);
    glPopMatrix();
    glFlush();
}
void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (w <= h)
        glOrtho(-3.5, 3.5, -3.5 * (GLfloat)h / (GLfloat)w,
            3.5 * (GLfloat)h / (GLfloat)w, -3.5, 3.5);
    else
        glOrtho(-3.5 * (GLfloat)w / (GLfloat)h,
            3.5 * (GLfloat)w / (GLfloat)h, -3.5, 3.5, -3.5, 3.5);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
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

