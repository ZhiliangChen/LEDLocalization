#pragma once
class CvCalibrate
{

public:
	void Calibrate();

	void BlobDetector();
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