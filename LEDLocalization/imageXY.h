#pragma once

class CvImageXY
{

public:
	void ShowImage();
	void FindPoint();
	IplImage* img;
	IplImage* gray;
	CvMemStorage* storage;
	CvSeq* circles;
	int x_center;
	int y_center;


};
