#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include "imageXY.h"

void CvImageXY::ShowImage()
{

	img = cvLoadImage("RawToRGBData.bmp", 1);
	cvNamedWindow("circles", 1);
	cvShowImage("circles", img);

}

void CvImageXY::FindPoint()
{




}