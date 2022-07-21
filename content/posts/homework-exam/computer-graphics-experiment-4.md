---
title: "计算机图形学——实验四"
date: 2020-12-16T12:57:41+08:00
draft: true
tags: [
  "opencv",
  "opengl",
]
categories: [
  "作业与考试", 
]
---
河南理工大学计算机图形学上机实验——实验四
<!--more-->
## 题一

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>
#include<iostream>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束
// 请在此添加你的代码
/********** Begin ********/
GLfloat points1[4][3] = {{-1,0,1},{1,0,1},{0,0,-0.7},{0,1.7,0}};
GLfloat Colors1[4][3] = {{0,1,0},{1,0,0},{1,1,0},{0,0,1}};
int vertice1[4][3] = {{0,1,2},{1,2,3},{0,2,3},{0,1,3}};
/********** End **********/
void InitGL(GLvoid)
{
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_COLOR_MATERIAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}
void Create()                //创建三棱锥
{
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 4; i++)
    {
        glColor3fv(Colors1[i]);
        for (int j = 0; j < 3; j++)
        {
            int VtxId = vertice1[i][j];
            glVertex3fv(points1[VtxId]);
        }
    }
    glEnd();
}
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();	               
    // 请在此添加你的代码
    /********** Begin ********/
    glPushMatrix();
    glTranslatef(0.0f,0.2f,-3.0f);	//平移至左侧
    glRotatef(95.0,1.0,0.0,0.0);
    /********** End **********/
    Create(); 	              //三棱锥
    glPopMatrix();
    glutSwapBuffers();
}
void reshape(int width, int height)
{
    if (height == 0)
    height = 1;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
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
    InitGL();
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
## 题二

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>

// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束

GLint winWidth = 500, winHeight =500 ; 	      //设置初始化窗口大小

/*观察坐标系参数设置*/
GLfloat x0 = 2.0, yy = 2.0, z0 = 5.0;	   //设置观察坐标系原点 
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
    glColor3f(1.0,1.0,1.0);
    glTranslatef(0.0f,1.4f,0.0f); // 脖子
    glScalef(0.5,0.7,0.5);
    glutSolidCube(0.5);
    glPopMatrix();


    glPushMatrix();
    glColor3f(1.0, 0.5, 0.2);
    glTranslatef(0.0f,1.9f,0.0f); // 头
    glScalef(1.5,1.5,0.5);
    glutSolidCube(0.5);
    glPopMatrix();

    glPushMatrix();
    glColor3f(1.0,0.0,0.0);
    glTranslatef(0.0f,0.25f,0.0f); // 身体
    glScalef(4.0,4.0,0.5);
    glutSolidCube(0.5);
    glPopMatrix();

    glPushMatrix();
    glColor3f(1.0, 1.0, 0.0);
    glTranslatef(-1.25f,0.5f,0.0f); // 手
    glScalef(1.0,3.0,0.5);
    glutSolidCube(0.5);
    glPopMatrix();
    glPushMatrix();
    glColor3f(1.0, 1.0, 0.0);
    glTranslatef(1.25f,0.5f,0.0f); 
    glScalef(1.0,3.0,0.5);
    glutSolidCube(0.5);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.5,0.5,1.5);  // 腿
    glTranslatef(-0.5f,-1.5f,0.0f); 
    glScalef(1.0,3.0,0.5);
    glutSolidCube(0.5);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.5,0.5,1.5);
    glTranslatef(0.5f,-1.5f,0.0f); 
    glScalef(1.0,3.0,0.5);
    glutSolidCube(0.5);
    glPopMatrix();

    GLUquadricObj *sphere;  //定义二次曲面对象
    sphere=gluNewQuadric(); 

    glPushMatrix();  // 球
    glColor3f(1.0, 0.5, 0.2);
    glTranslatef(-1.25f,-0.5f,0.0f); 
    glScalef(1,1.5,1);
    gluSphere(sphere,0.25,50,50);
    glPopMatrix();
    glPushMatrix();
    glColor3f(1.0, 0.5, 0.2);
    glTranslatef(1.25f,-0.5f,0.0f); 
    glScalef(1,1.5,1);
    gluSphere(sphere,0.25,50,50);
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
    GLubyte* pPixelData = (GLubyte*)malloc(500 * 500 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 500 , 500 );        //设置初始化窗口大小
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
    img.create(500, 500, CV_8UC3);
    cv::split(img, imgPlanes);

    for (int i = 0; i < 500; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 500; j++) {
            int k = 3 * (i * 500 + j);
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

## 题三

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdio.h>
#include <stdlib.h>
#include <vector>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
// 评测代码所用头文件-结束


using namespace std;

struct Point
{
	int x, y;
};

Point bz[11], bspt[11];     //bz为Bezier曲线，bspt为B样条曲线

int nInput;
//Bezier曲线控制点
Point pt[4] = { { 50, 100}, { 140,300},{250, 320}, {290, 120} };
//B样条曲线控制点
/**********请在此处添加你的代码*******/
 /************** Begin ************/
Point pt2[4]={ { 450, 100}, { 540,300},{650, 320}, {690, 120}  };
/*************** end **************/

void CalcBZPoints()                              //Bezier曲线算法
{
	float a0, a1, a2, a3, b0, b1, b2, b3;
	a0 = pt[0].x;
	a1 = -3 * pt[0].x + 3 * pt[1].x;
	a2 = 3 * pt[0].x - 6 * pt[1].x + 3 * pt[2].x;
	a3 = -pt[0].x + 3 * pt[1].x - 3 * pt[2].x + pt[3].x;
	b0 = pt[0].y;
	b1 = -3 * pt[0].y + 3 * pt[1].y;
	b2 = 3 * pt[0].y - 6 * pt[1].y + 3 * pt[2].y;
	b3 = -pt[0].y + 3 * pt[1].y - 3 * pt[2].y + pt[3].y;

	float t = 0;
	float dt = 0.01;
	for (int i = 0; t < 1.1; t += 0.1, i++)
	{
		bz[i].x = a0 + a1 * t + a2 * t * t + a3 * t * t * t;
		bz[i].y = b0 + b1 * t + b2 * t * t + b3 * t * t * t;
	}
}


void CalcBSPoints()                          //B样条曲线
{
/**********请在此处添加你的代码*******/
 /************** Begin ************/
    float a0, a1, a2, a3, b0, b1, b2, b3;
	a0 = pt2[0].x + 4 * pt2[1].x + pt2[2].x;
	a1 = -3 * pt2[0].x + 3 * pt2[2].x;
	a2 = 3 * pt2[0].x - 6 * pt2[1].x + 3 * pt2[2].x;
	a3 = -pt2[0].x + 3 * pt2[1].x - 3 * pt2[2].x + pt2[3].x;
	b0 = pt2[0].y + 4 * pt2[1].y + pt2[2].y;
	b1 = -3 * pt2[0].y + 3 * pt2[2].y;
	b2 = 3 * pt2[0].y - 6 * pt2[1].y + 3 * pt2[2].y;
	b3 = -pt2[0].y + 3 * pt2[1].y - 3 * pt2[2].y + pt2[3].y; 

	float t = 0;
	float dt = 0.01;
	for (int i = 0; t < 1.1; t += 0.1, i++)
	{
		bspt[i].x =  ( a0 + a1 * t + a2 * t * t + a3 * t * t * t ) / 6;
		bspt[i].y =  ( b0 + b1 * t + b2 * t * t + b3 * t * t * t ) / 6;
	}
}
/************* end ************/
void ControlPoint()
{
	glPointSize(2);
	for (int i = 0; i < 4; i++)
	{
		glBegin(GL_POINTS);
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex2i(pt[i].x, pt[i].y);
		glEnd();
	
	}
}

void PolylineGL(Point* pt, int num)
{
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < num; i++)
	{   
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex2i(pt[i].x, pt[i].y);
	}
	glEnd();
}
void PolylineGL1(Point* pt, int num)
{
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < num; i++)
	{   
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertex2i(pt[i].x, pt[i].y);
	}
	glEnd();
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0f, 1.0f, 1.0f);

	ControlPoint();              //画4个控制点

	PolylineGL1(pt, 4);           //画4个点之间的线段

	CalcBZPoints();

	PolylineGL(bz, 11);

	PolylineGL1(pt2, 4);

	CalcBSPoints();

	PolylineGL(bspt, 11);

	glFlush();
}

void init()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
}




int main(int argc, char* argv[])
{
    GLubyte* pPixelData = (GLubyte*)malloc(800 * 400 * 3);//分配内存
    GLint viewport[4] = { 0 };
    glutInit(&argc, argv);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize( 800 , 400 );        //设置初始化窗口大小
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
    cv::imwrite("../img_step3/test.jpg", img);
    return 0;
}
```

