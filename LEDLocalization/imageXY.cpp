#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include "imageXY.h"
#include "opencv2/features2d/features2d.hpp"  //SimpleBlobDetector头文件
#include <opencv2\opencv.hpp>//imread和imshow头文件
#include "LEDLocalizationDlg.h"
#include "resource.h"//引用控件名

BYTE   *m_RGBData;
std::vector<cv::KeyPoint> detectKeyPoint;


void CvImageXY::Test()
{
	/*CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);
	pEdit->AddString("test");*/

	cv::Mat image(cvSize(1280, 1024), CV_8UC3, cv::Scalar(0));
	memcpy(image.data, m_RGBData, 1280 * 1024 * 3);
	imshow("src image", image);
}

void CvImageXY::ShowImage()
{


	
	//IplImage* img = cvCreateImage(cvSize(1280, 1024), IPL_DEPTH_8U, 3); // 创建单通道图像
	//memcpy(img->imageData, m_RGBData, 1280*1024 * 3);

	img = cvLoadImage("test.jpg", 1);
	
	gray = cvCreateImage(cvGetSize(img), 8, 1);
	
	storage = cvCreateMemStorage(0);
	cvCvtColor(img, gray, CV_BGR2GRAY);
	
	//高斯平滑也影响圆的识别
	//cvSmooth(gray, gray, CV_GAUSSIAN, 9, 9); // smooth it, otherwise a lot of false circles may be detected
	 /*cvHoughCircles(CvArr* image, void* circle_storage,int method, double dp, double min_dist,double param1 CV_DEFAULT(100),
					double param2 CV_DEFAULT(100),int min_radius CV_DEFAULT(0),int max_radius CV_DEFAULT(0));*/
	//min_dist，该参数是让算法能明显区分的两个不同圆之间的最小距离；param1用于Canny的边缘阀值上限，param2累加器的阀值，都越大对圆要求越高。
	 //检测接箍内圆，最大最小半径设置为325可以检测到；累加器阈值50会导致程序反应过慢，改成30有所缓解
	circles = cvHoughCircles(gray, storage, CV_HOUGH_GRADIENT, 1, 2, 100, 5, 1, 20);

	int i;
	for (i = 0; i < circles->total; i++)
	{
		float* p = (float*)cvGetSeqElem(circles, i);
		//画圆和圆心
		//cvCircle(img, cvPoint(cvRound(p[0]), cvRound(p[1])), 3, CV_RGB(0, 255, 0), -1, 8, 0);
		cvCircle(img, cvPoint(cvRound(p[0]), cvRound(p[1])), 1, CV_RGB(255, 0, 0), 1, 8, 0);
		x_center = cvRound(p[0]);
		y_center = cvRound(p[1]);
		//如何输出小数位？？？
		//x_center = p[0];
		//y_center = p[1];
	}
	////cout << "圆数量=" << circles->total << endl;

	cvNamedWindow("circles", 1);
	cvShowImage("circles", img);
	
	
}

void CvImageXY::BlobDetector()
{
	int i;
	cv::Mat imgg = cv::imread("test.jpg", 1);
	cv::Mat srcGrayImage;
	
	if (imgg.channels() == 3)
	{
		cvtColor(imgg, srcGrayImage, CV_RGB2GRAY);
	}
	else
	{
		imgg.copyTo(srcGrayImage);
	}
	
	cv::Mat keyPointImage1, keyPointImage2;

	cv::SimpleBlobDetector::Params params;
	//params.filterByInertia = true;
	//params.filterByColor = true;
	params.blobColor = 255;
	//params.filterByArea = true;
	params.minThreshold = 50;
	params.minArea = 1;
	params.thresholdStep = 1;
	params.minDistBetweenBlobs = 1;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(params);
	//sbd->create("SimpleBlob");
	sbd->detect(imgg, detectKeyPoint);
	for (i = 0; i < 15; i++)
	{
		img_point[i][0] = detectKeyPoint[i].pt.x;
		img_point[i][1] = detectKeyPoint[i].pt.y;
	}
	//test = detectKeyPoint[1].pt.y;
	//test = params.maxThreshold;
	drawKeypoints(imgg, detectKeyPoint, keyPointImage1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(imgg, detectKeyPoint, keyPointImage2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

	imshow("src image", imgg);
	imshow("keyPoint image1", keyPointImage1);
	imshow("keyPoint image2", keyPointImage2);


}