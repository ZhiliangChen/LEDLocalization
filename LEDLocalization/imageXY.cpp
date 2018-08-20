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
std::vector<cv::KeyPoint> detectKeyPoint_binary;


void CvImageXY::Test()
{
	cv::Mat img_binary;
	img = cvLoadImage("test.jpg", 1);
	gray = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);
	
	cv::Mat image = cv::cvarrToMat(gray);
	cv::threshold(image, img_binary, 50, 255, CV_THRESH_TOZERO);
	cvSmooth(gray, gray, CV_GAUSSIAN, 3, 3);
	cvNamedWindow("circles", 1);
	cvShowImage("circles", gray);
	imshow("binary", img_binary);
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
	cv::Mat image_fliped;
	cv::Mat image(cvSize(1280, 1024), CV_8UC3, cv::Scalar(0));
	memcpy(image.data, m_RGBData, 1280 * 1024 * 3);
	cv::flip(image, image_fliped, 0);
	cv::Mat srcGrayImage;
	
	if (image_fliped.channels() == 3)
	{
		cvtColor(image_fliped, srcGrayImage, CV_RGB2GRAY);
	}
	else
	{
		image_fliped.copyTo(srcGrayImage);
	}
	
	cv::Mat keyPointImage1, keyPointImage2;

	cv::SimpleBlobDetector::Params params;
	params.filterByInertia = true;
	//params.filterByColor = true;
	params.blobColor = 255;
	params.filterByArea = true;
	params.minThreshold = 50;//原来是50，200太大肯定不行
	params.minArea = 5;
	params.thresholdStep = 1;
	params.minDistBetweenBlobs = 1;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(params);
	//sbd->create("SimpleBlob");
	sbd->detect(srcGrayImage, detectKeyPoint);

	//test = detectKeyPoint[1].pt.y;
	//test = params.maxThreshold;
	//drawKeypoints(srcGrayImage, detectKeyPoint, keyPointImage1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcGrayImage, detectKeyPoint, keyPointImage2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

	//imshow("src image", srcGrayImage);
	//imshow("keyPoint image1", keyPointImage1);
	imshow("keyPoint image2", keyPointImage2);
}

void CvImageXY::BlobDetector_threshold()
{
	cv::Mat image_fliped;
	cv::Mat image(cvSize(1280, 1024), CV_8UC3, cv::Scalar(0));
	memcpy(image.data, m_RGBData, 1280 * 1024 * 3);
	cv::flip(image, image_fliped, 0);
	cv::Mat srcGrayImage, img_binary;

	if (image_fliped.channels() == 3)
	{
		cvtColor(image_fliped, srcGrayImage, CV_RGB2GRAY);
	}
	else
	{
		image_fliped.copyTo(srcGrayImage);
	}
	cv::threshold(srcGrayImage, img_binary, 50, 255, CV_THRESH_TOZERO);//初始值阈值为50，改为100更差，改为200更差
	cv::Mat keyPointImage1, keyPointImage2;

	cv::SimpleBlobDetector::Params params;
	params.filterByInertia = true;
	//params.filterByColor = true;
	params.blobColor = 255;
	params.filterByArea = true;
	params.minThreshold = 50;
	params.minArea = 1;
	params.thresholdStep = 1;
	params.minDistBetweenBlobs = 1;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(params);
	//sbd->create("SimpleBlob");
	sbd->detect(img_binary, detectKeyPoint);

	//test = detectKeyPoint[1].pt.y;
	//test = params.maxThreshold;
	//drawKeypoints(srcGrayImage, detectKeyPoint, keyPointImage1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_binary, detectKeyPoint, keyPointImage2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

	//imshow("src image", srcGrayImage);
	//imshow("keyPoint image1", keyPointImage1);
	imshow("keyPoint image2", keyPointImage2);
}
void CvImageXY::BlobDetector_static()
{
	cv::Mat imgg = cv::imread("test.jpg", 1);
	cv::Mat srcGrayImage, img_binary, img_smooth;
	
	if (imgg.channels() == 3)
	{
		cvtColor(imgg, srcGrayImage, CV_RGB2GRAY);
	}
	else
	{
		imgg.copyTo(srcGrayImage);
	}

	cv::threshold(srcGrayImage, img_binary, 50, 255, CV_THRESH_TOZERO);
	cv::Mat keyPointImage1, keyPointImage2;

	cv::SimpleBlobDetector::Params params;
	params.filterByInertia = true;
	//params.filterByColor = true;
	params.blobColor = 255;
	//params.filterByArea = true;
	params.minThreshold = 50;
	params.minArea = 1;
	params.thresholdStep = 1;
	params.minDistBetweenBlobs = 1;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(params);
	//sbd->create("SimpleBlob");
	sbd->detect(srcGrayImage, detectKeyPoint);
	
	//test = params.maxThreshold;
	//drawKeypoints(srcGrayImage, detectKeyPoint, keyPointImage1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcGrayImage, detectKeyPoint, keyPointImage2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
	
	//cvSmooth(&(IplImage)srcGrayImage, &(IplImage)img_smooth, CV_GAUSSIAN, 3, 3);
	
	cv::threshold(srcGrayImage, img_binary, 50, 255, CV_THRESH_TOZERO);//二值化确实有影响，等待验证
	cv::SimpleBlobDetector::Params params_binary;
	params_binary.filterByInertia = true;
	//params.filterByColor = true;
	params_binary.blobColor = 255;
	//params.filterByArea = true;
	params_binary.minThreshold = 10;
	params_binary.minArea = 0.5;
	params_binary.thresholdStep = 1;
	params_binary.minDistBetweenBlobs = 1;
	cv::Ptr<cv::SimpleBlobDetector> sbd_binary = cv::SimpleBlobDetector::create(params_binary);
	sbd_binary->detect(img_binary, detectKeyPoint_binary);
	drawKeypoints(img_binary, detectKeyPoint_binary, keyPointImage1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

	//imshow("src image", srcGrayImage);
	imshow("keyPoint image_binary", keyPointImage1);
	imshow("keyPoint image2", keyPointImage2);
	

}