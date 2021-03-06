
#include "stdafx.h"
//#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "calibrate.h"
#include "LEDLocalizationDlg.h"//临时用，在listbox显示calibrate的进度
#include "resource.h"//临时用，引用控件名

using namespace cv;
using namespace std;

//相机内外参数初始化赋值-MVC-1000F相机
//float cameraMatrix_init[3][3] = { {1152.195197178761, 0, 611.574705796087},{ 0, 1152.841197789385, 502.5554719939521 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { -0.2132339464479368 },{ 0.3584186449258908 },{ -0.00121915745714334 },{ -0.001944591250376644 },{ -0.07942466741526423 } };
//相机内外参数初始化赋值-组7-JAI GO5000 PGE
//float cameraMatrix_init[3][3] = { { 10175.68500053747, 0, 1220.875697016915 },{ 0, 10178.89792599261, 1015.893016060104 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { 0.4194629330609266 },{ 0.820017412107885 },{ -0.001629398971545851 },{ -0.008878575029553925 },{ -0.0534669305095578 } };
//相机内外参数初始化赋值-JAI换6mm镜头
//float cameraMatrix_init[3][3] = { { 1221.0822, 0, 1220.875697016915 },{ 0, 1221.467751119113, 1015.893016060104 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { 0.4194629330609266 },{ 0.820017412107885 },{ -0.001629398971545851 },{ -0.008878575029553925 },{ -0.0534669305095578 } };

//相机内外参数初始化赋值-JAI换25mm镜头
float cameraMatrix_init[3][3] = { { 5087.842500268735, 0, 1220.875697016915 },{ 0, 5089.448962996305, 1015.893016060104 },{ 0, 0, 1 } };
float distCoeffs_init[5][1] = { { 0.4194629330609266 },{ 0.820017412107885 },{ -0.001629398971545851 },{ -0.008878575029553925 },{ -0.0534669305095578 } };
//相机内外参数初始化赋值-组2
//float cameraMatrix_init[3][3] = { { 10203.47872662463, 0, 1208.487534546189 },{ 0, 10209.72735750312, 972.6484385277352 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { 0.4478602683069452 },{ -6.977186927749671 },{ -0.002989837743736836 },{ -0.007250628326077741 },{ 194.8294231570288 } };

cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, cameraMatrix_init); /* 摄像机内参数矩阵 */
cv::Mat distCoeffs = cv::Mat(5, 1, CV_32FC1, distCoeffs_init); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */


void CvCalibrate::Calibrate()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	ifstream fin("calibdata.txt"); /* 标定所用图像文件的路径 */
	ofstream fout("caliberation_result.txt");  /* 保存标定结果的文件 */
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	pEdit->AddString("开始提取角点");
	int image_count = 0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(8, 11);    /* 标定板上每行、列的角点数 ======================================标定板参数*/
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq;/* 保存检测到的所有角点 */

	string filename;
	int count = -1;//用于存储角点个数。
	while (getline(fin, filename))
	{
		image_count++;
		// 用于观察检验输出
		m_str.Format("image_count = %d", image_count);
		pEdit->AddString(m_str);
		/* 输出检验*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		//waitKey(500);//暂停0.5S	
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			m_str.Format("image_size.width = %f", image_size.width);
			pEdit->AddString(m_str);
			m_str.Format("image_size.height = %f", image_size.height);
			pEdit->AddString(m_str);
		}

		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			pEdit->AddString("can not find chessboard corners!");//找不到角点
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
			/* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray, board_size, image_points_buf, true); //用于在图片中标记角点
			imshow("Camera Calibration", view_gray);//显示图片
			waitKey(200);//暂停0.5S		
		}
	}
	int total = image_points_seq.size();//所有角点个数
	m_str.Format("total =  %d", total);
	pEdit->AddString(m_str);
	int CornerNum = board_size.width*board_size.height;  //每张图片上总的角点数
	for (int ii = 0; ii < total; ii++)
	{

		m_str.Format("第 %d 幅图片的数据", ii + 1);
		pEdit->AddString(m_str);

		//输出所有的角点
		m_str.Format("x = %f,y = %f", image_points_seq[ii][1].x, image_points_seq[ii][1].y);
		pEdit->AddString(m_str);
	}

	pEdit->AddString("角点提取完成!");


	//以下是摄像机标定
	pEdit->AddString("开始标定");
	/*棋盘三维信息*/
	Size square_size = Size(30, 30);  /* 实际测量得到的标定板上每个棋盘格的大小 ,单位是mm吗===============标定板参数*/
	vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
	/*内外参数*/
	//cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
	vector<int> point_counts;  // 每幅图像中角点的数量
	//distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
	 /* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* 开始标定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	pEdit->AddString("标定完成！");


	//对标定结果进行评价
	pEdit->AddString("开始评价标定结果！");
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
	fout << "每幅图像的标定误差：\n";
	for (i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	pEdit->AddString("评价完成！");


	//保存定标结果  	
	pEdit->AddString("开始保存定标结果！");
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << tvecsMat[i] << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(tvecsMat[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	pEdit->AddString("完成保存！");
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	fout << endl;
}
	/************************************************************************
	显示定标结果，矫正畸变后的结果
	*************************************************************************/
//	Mat mapx = Mat(image_size, CV_32FC1);
//	Mat mapy = Mat(image_size, CV_32FC1);
//	Mat R = Mat::eye(3, 3, CV_32F);
//	pEdit->AddString("保存矫正图像！");
//	string imageFileName;
//	std::stringstream StrStm;
//	for (int i = 0; i != image_count; i++)
//	{
//		m_str.Format("Frame # %d", i + 1);
//		pEdit->AddString(m_str);
//
//		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
//		StrStm.clear();
//		imageFileName.clear();
//		string filePath = "chess";
//		StrStm << i + 1;
//		StrStm >> imageFileName;
//		filePath += imageFileName;
//		filePath += ".bmp";
//		Mat imageSource = imread(filePath);
//		Mat newimage = imageSource.clone();
//		//另一种不需要转换矩阵的方式
//		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);
//		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
//		imshow("原始图像", imageSource);
//		imshow("矫正后图像", newimage);
//		waitKey(500);//暂停0.5S		
//		StrStm.clear();
//		filePath.clear();
//		StrStm << i + 1;
//		StrStm >> imageFileName;
//		imageFileName += "_d.jpg";
//		imwrite(imageFileName, newimage);
//	}
//	pEdit->AddString("保存结束！");
//}