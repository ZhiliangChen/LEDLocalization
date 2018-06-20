#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include "imageXY.h"

BYTE   *m_RGBData;


void CvImageXY::ShowImage()
{


	
	IplImage* img = cvCreateImage(cvSize(1280, 1024), IPL_DEPTH_8U, 3); // ������ͨ��ͼ��
	//img->imageSize = 1280 * 1024 * 3;
	//img->width = 1280;
	//img->height = 1024;
	memcpy(img->imageData, m_RGBData, 1280*1024 * 3);
	//img = cvLoadImage("RawToRGBData.bmp", 1);
	
	gray = cvCreateImage(cvGetSize(img), 8, 1);
	storage = cvCreateMemStorage(0);
	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvSmooth(gray, gray, CV_GAUSSIAN, 9, 9); // smooth it, otherwise a lot of false circles may be detected
											 /*cvHoughCircles(CvArr* image, void* circle_storage,int method, double dp, double min_dist,double param1 CV_DEFAULT(100),
											 double param2 CV_DEFAULT(100),int min_radius CV_DEFAULT(0),int max_radius CV_DEFAULT(0));*/
											 //���ӹ���Բ�������С�뾶����Ϊ325���Լ�⵽���ۼ�����ֵ50�ᵼ�³���Ӧ�������ĳ�30��������
	circles = cvHoughCircles(gray, storage, CV_HOUGH_GRADIENT, 1, gray->height / 4, 100, 30, 10, 326);
	//cv::seqToMat(seq, _circles);
	int i;
	for (i = 0; i < circles->total; i++)
	{
		float* p = (float*)cvGetSeqElem(circles, i);
		//��Բ��Բ��
		cvCircle(img, cvPoint(cvRound(p[0]), cvRound(p[1])), 3, CV_RGB(0, 255, 0), -1, 8, 0);
		cvCircle(img, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 3, 8, 0);
		x_center = cvRound(p[0]);
		y_center = cvRound(p[1]);
		//������С��λ������
		//x_center = p[0];
		//y_center = p[1];
	}
	////cout << "Բ����=" << circles->total << endl;

	cvNamedWindow("circles", 1);
	cvShowImage("circles", img);
	
	
}

void CvImageXY::FindPoint()
{




}