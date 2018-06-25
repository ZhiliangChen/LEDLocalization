#pragma once
#include "opencv2/highgui/highgui.hpp"
class CvCalibrate
{

public:
	void Calibrate();

	CString m_str;


};
extern cv::Mat cameraMatrix;
extern cv::Mat distCoeffs;