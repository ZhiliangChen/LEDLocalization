#pragma once

class CvImageXY
{

public:
	void ShowImage();
	
	void BlobDetector();
	void BlobDetector_static();
	void Test();
	IplImage* img;
	IplImage* gray;
	CvMemStorage* storage;
	CvSeq* circles;
	int x_center;
	int y_center;
	float test;
	float img_point[15][2];
	


};
extern BYTE   *m_RGBData;
extern std::vector<cv::KeyPoint> detectKeyPoint;
