#pragma once

class CvImageXY
{

public:
	void ShowImage();
	
	void BlobDetector();
	void BlobDetector_threshold();
	void BlobDetector_static();
	void Test();
	IplImage* img;
	IplImage* gray;
	CvMemStorage* storage;
	CvSeq* circles;
	int x_center;
	int y_center;
	float test;

};
extern BYTE   *m_RGBData;
extern std::vector<cv::KeyPoint> detectKeyPoint;
extern std::vector<cv::KeyPoint> detectKeyPoint_binary;
