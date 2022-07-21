---
title: "计算机图形学——实验五"
date: 2020-12-23T12:57:41+08:00
draft: true
tags: [
  "opencv",
  "opengl",
]
categories: [
  "作业与考试", 
]
---
河南理工大学计算机图形学上机实验——实验五
<!--more-->
## 题1

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdlib.h>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束

void init(void)
{
	GLfloat position[] = {0.0,3.0,2.0,0.0};
	GLfloat ambient[]={0.0,0.0,0.0,1.0};
    GLfloat diffuse[]={1.0,1.0,1.0,1.0};
    GLfloat specular[]={1.0,1.0,1.0,1.0};
    GLfloat lmodel_ambient[]={0.4,0.4,0.4,1.0};
    GLfloat local_view[]={0.0};

    glClearColor(0.0, 0.1, 0.1,0.0); 
    glShadeModel(GL_SMOOTH);
    
    glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuse);
    glLightfv(GL_LIGHT0,GL_POSITION,position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT,lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER,local_view);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);

}

void display(void)
{
   GLfloat no_mat[]={0.0,0.0,0.0,1.0};                     //没有光
   GLfloat mat_ambient[]={0.7,0.7,0.7,1.0};                //环境光
   GLfloat mat_ambient_color[]={0.8, 0.8, 0.2, 1.0};       //彩色环境光 
   GLfloat mat_diffuse[]={0.1,0.5,0.8,1.0};                //漫反射
   GLfloat mat_specular[]={1.0,1.0,1.0,1.0};               //镜面反射
   GLfloat no_shininess[] = {0.0};                         //没有镜面反射
   GLfloat low_shininess[]={5.0};                          //低镜面反射
   GLfloat high_shininess[]={100.0};                       //高镜面反射
   GLfloat mat_emission[]={0.3,0.2,0.2,0.0};               //材料辐射光颜色 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();
    glTranslatef(-3.0, 0.0, 0.0);
    // 请在此添加你的代码，左侧圆球只有漫反射
    /********** Begin ********/
    glMaterialfv(GL_FRONT, GL_AMBIENT, no_mat);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();

	glPushMatrix();
    // 请在此添加你的代码，中间圆球有环境光和漫反射
    /********** Begin ********/

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();

    glPushMatrix();
    glTranslatef(3.0, 0.0, 0.0);
    // 请在此添加你的代码，右侧圆球有彩色环境光和漫反射
    /********** Begin ********/

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient_color);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();

	glFlush();
	glutPostRedisplay();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}





int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(800 * 600 * 3);//分配内存
    GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
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
    img.create(600, 800, CV_8UC3);
    cv::split(img, imgPlanes);

    for (int i = 0; i < 600; i++) {
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
## 题2

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdlib.h>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束

void init(void)
{
	GLfloat position[] = {0.0,3.0,2.0,0.0};
	GLfloat ambient[]={0.0,0.0,0.0,1.0};
    GLfloat diffuse[]={1.0,1.0,1.0,1.0};
    GLfloat specular[]={1.0,1.0,1.0,1.0};
    GLfloat lmodel_ambient[]={0.4,0.4,0.4,1.0};
    GLfloat local_view[]={0.0};

    glClearColor(0.0, 0.1, 0.1,0.0); 
    glShadeModel(GL_SMOOTH);
    
    glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuse);
    glLightfv(GL_LIGHT0,GL_POSITION,position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT,lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER,local_view);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);

}

void display(void)
{
   GLfloat no_mat[]={0.0,0.0,0.0,1.0};                     //没有光
   GLfloat mat_ambient[]={0.7,0.7,0.7,1.0};                //环境光
   GLfloat mat_ambient_color[]={0.8, 0.8, 0.2, 1.0};       //彩色环境光 
   GLfloat mat_diffuse[]={0.1,0.5,0.8,1.0};                //漫反射
   GLfloat mat_specular[]={1.0,1.0,1.0,1.0};               //镜面反射
   GLfloat no_shininess[] = {0.0};                         //没有镜面反射
   GLfloat low_shininess[]={5.0};                          //低镜面反射
   GLfloat high_shininess[]={100.0};                       //高镜面反射
   GLfloat mat_emission[]={0.3,0.2,0.2,0.0};               //材料辐射光颜色 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();
    glTranslatef(-3.0, 0.0, 0.0);
    // 请在此添加你的代码，左侧圆球有环境光、漫反射镜面低反射
    /********** Begin ********/
    glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
    glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
    glMaterialfv(GL_FRONT,GL_SHININESS,low_shininess);
    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();

	glPushMatrix();
    // 请在此添加你的代码，中间圆球有环境光、漫反射镜面高反射
    /********** Begin ********/

    glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
    glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
    glMaterialfv(GL_FRONT,GL_SHININESS,high_shininess);
    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();

    glPushMatrix();
    glTranslatef(3.0, 0.0, 0.0);
    // 请在此添加你的代码，右侧圆球有彩色环境光、漫反射、镜面高反射和材料辐射
    /********** Begin ********/

    glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient_color);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
    glMaterialfv(GL_FRONT,GL_SHININESS,high_shininess);
    glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
    glMaterialfv(GL_FRONT,GL_EMISSION,mat_emission);
    
    /********** End *********/
	glutSolidSphere(1.0, 50, 50);  
    glPopMatrix();



	glFlush();
	glutPostRedisplay();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}





int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(800 * 600 * 3);//分配内存
    GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
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
    img.create(600, 800, CV_8UC3);
    cv::split(img, imgPlanes);

    for (int i = 0; i < 600; i++) {
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
    cv::imwrite("../img_step1/test.jpg", img);
    return 0;
}
```
## 题3

```c
// 提示：在合适的地方修改或添加代码
#include <GL/freeglut.h>
#include<stdlib.h>
// 评测代码所用头文件-开始
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;
// 评测代码所用头文件-结束
// 初始化材质、光源、光照模型、深度缓存

//定义左边茶壶材质系数 
GLfloat left_ambient[] = { 5.0f, 0.0f, 0.0f, 1.0f };
GLfloat left_diffuse[] = { 0.8f, 0.8f, 0.0f, 1.0f };
GLfloat left_specular[] = { 1.0f, 1.0f, 0.0f, 1.0f };
GLfloat left_emission[] = { 0.1f, 0.0f, 0.0f, 1.0f };

//定义中间茶壶材质系数  
GLfloat mid_ambient[] = { 0.0f, 0.2f, 0.0f, 1.0f };
GLfloat mid_diffuse[] = { 0.0f, 0.8f, 0.0f, 1.0f };
GLfloat mid_specular[] = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat mid_shininess[] = { 80.0f };
GLfloat mid_emission[] = { 0.0f, 0.1f, 0.0f, 1.0f };

//定义右边茶壶材质系数
GLfloat right_ambient[] = { 0.0f, 0.0f, 0.1f, 1.0f };
GLfloat right_diffuse[] = { 0.0f, 0.0f, 0.8f, 1.0f };
GLfloat right_specular[] = { 0.0f, 0.0f, 0.9f, 1.0f };
GLfloat right_shininess[] = { 50.0f };
GLfloat right_emission[] = { 0.0f, 0.1f, 0.0f, 1.0f };





void myInit(void)
{
   
    GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
    GLfloat white_light[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_ambient[] = { 0.2 , 0.2 , 0.2 , 1.0 };


    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_SMOOTH);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST); 

}

void myDisplay(void)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    // 请在此添加你的代码，中间茶壶
    /********** Begin ********/
    glMaterialfv(GL_FRONT,GL_AMBIENT,mid_ambient);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,mid_diffuse);
    glMaterialfv(GL_FRONT,GL_EMISSION ,mid_emission);

    /********** End *********/
    glutSolidTeapot(0.5);
    glPopMatrix();
    glFlush();


    glPushMatrix();
    glTranslatef(2.0,0.0,0.0);
    // 请在此添加你的代码,右边茶壶
    /********** Begin ********/
    glMaterialfv(GL_FRONT,GL_AMBIENT,right_ambient);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,right_diffuse);
    glMaterialfv(GL_FRONT,GL_SHININESS,right_shininess);
    glMaterialfv(GL_FRONT,GL_SPECULAR,right_specular);
    glMaterialfv(GL_FRONT,GL_EMISSION ,right_emission);

    /********** End *********/
    glutSolidTeapot(0.5);
    glPopMatrix();
    glFlush();


    glPushMatrix();
    glDisable(GL_LIGHT0);
    glTranslatef(-2.0,0.0,0.0);
    // 请在此添加你的代码，左边茶壶
    /********** Begin ********/
    glMaterialfv(GL_FRONT,GL_AMBIENT,left_ambient);
    glMaterialfv(GL_FRONT,GL_EMISSION ,left_emission);
    /********** End *********/
    glutSolidTeapot(0.5);
    glPopMatrix();
    glFlush();


}



void myReshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (w <= h)
        glOrtho(-1.5, 1.5, -1.5 * (GLfloat)h / (GLfloat)w,
            1.5 * (GLfloat)h / (GLfloat)w, -10.0, 10.0);
    else
        glOrtho(-1.5 * (GLfloat)w / (GLfloat)h,
            1.5 * (GLfloat)w / (GLfloat)h, -1.5, 1.5, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}










int main(int argc, char *argv[])
{
	GLubyte* pPixelData = (GLubyte*)malloc(1000 * 400 * 3);//分配内存
    GLint viewport[4] = {0};
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(1000, 400);
	glutCreateWindow("几何变换示例");
    myInit();
    glutDisplayFunc(myDisplay);
    glutReshapeFunc(myReshape);
    glutMainLoopEvent();     
     
    /*************以下为评测代码，与本次实验内容无关，请勿修改**************/
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pPixelData);
    cv::Mat img;
    std::vector<cv::Mat> imgPlanes;
    img.create(400, 1000, CV_8UC3);
    cv::split(img, imgPlanes);

    for (int i = 0; i < 400; i++) {
        unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);
        unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);
        unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);
        for (int j = 0; j < 1000; j++) {
            int k = 3 * (i * 1000 + j);
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


