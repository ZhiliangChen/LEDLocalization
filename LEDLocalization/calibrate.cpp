
#include "stdafx.h"
//#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "calibrate.h"
#include "LEDLocalizationDlg.h"//��ʱ�ã���listbox��ʾcalibrate�Ľ���
#include "resource.h"//��ʱ�ã����ÿؼ���

using namespace cv;
using namespace std;

//������������ʼ����ֵ-MVC-1000F���
//float cameraMatrix_init[3][3] = { {1152.195197178761, 0, 611.574705796087},{ 0, 1152.841197789385, 502.5554719939521 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { -0.2132339464479368 },{ 0.3584186449258908 },{ -0.00121915745714334 },{ -0.001944591250376644 },{ -0.07942466741526423 } };
//������������ʼ����ֵ-��7
float cameraMatrix_init[3][3] = { { 10175.68500053747, 0, 1220.875697016915 },{ 0, 10178.89792599261, 1015.893016060104 },{ 0, 0, 1 } };
float distCoeffs_init[5][1] = { { 0.4194629330609266 },{ 0.820017412107885 },{ -0.001629398971545851 },{ -0.008878575029553925 },{ -0.0534669305095578 } };
//������������ʼ����ֵ-��4
//float cameraMatrix_init[3][3] = { { 10229.01703639699, 0, 1224.159109211674 },{ 0, 10233.83288897842, 992.3885470755396 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { 0.4679801617786223 },{ -5.055978633815441 },{ -0.002957403346868477 },{ -0.008347294354659016 },{ -0.04287388056296532 } };
//������������ʼ����ֵ-��2
//float cameraMatrix_init[3][3] = { { 10203.47872662463, 0, 1208.487534546189 },{ 0, 10209.72735750312, 972.6484385277352 },{ 0, 0, 1 } };
//float distCoeffs_init[5][1] = { { 0.4478602683069452 },{ -6.977186927749671 },{ -0.002989837743736836 },{ -0.007250628326077741 },{ 194.8294231570288 } };

cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, cameraMatrix_init); /* ������ڲ������� */
cv::Mat distCoeffs = cv::Mat(5, 1, CV_32FC1, distCoeffs_init); /* �������5������ϵ����k1,k2,p1,p2,k3 */


void CvCalibrate::Calibrate()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	ifstream fin("calibdata.txt"); /* �궨����ͼ���ļ���·�� */
	ofstream fout("caliberation_result.txt");  /* ����궨������ļ� */
	//��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��	
	pEdit->AddString("��ʼ��ȡ�ǵ�");
	int image_count = 0;  /* ͼ������ */
	Size image_size;  /* ͼ��ĳߴ� */
	Size board_size = Size(8, 11);    /* �궨����ÿ�С��еĽǵ��� ======================================�궨�����*/
	vector<Point2f> image_points_buf;  /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	vector<vector<Point2f>> image_points_seq;/* �����⵽�����нǵ� */

	string filename;
	int count = -1;//���ڴ洢�ǵ������
	while (getline(fin, filename))
	{
		image_count++;
		// ���ڹ۲�������
		m_str.Format("image_count = %d", image_count);
		pEdit->AddString(m_str);
		/* �������*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		//waitKey(500);//��ͣ0.5S	
		if (image_count == 1)  //�����һ��ͼƬʱ��ȡͼ������Ϣ
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			m_str.Format("image_size.width = %f", image_size.width);
			pEdit->AddString(m_str);
			m_str.Format("image_size.height = %f", image_size.height);
			pEdit->AddString(m_str);
		}

		/* ��ȡ�ǵ� */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			pEdit->AddString("can not find chessboard corners!");//�Ҳ����ǵ�
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* �����ؾ�ȷ�� */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
			image_points_seq.push_back(image_points_buf);  //���������ؽǵ�
			/* ��ͼ������ʾ�ǵ�λ�� */
			drawChessboardCorners(view_gray, board_size, image_points_buf, true); //������ͼƬ�б�ǽǵ�
			imshow("Camera Calibration", view_gray);//��ʾͼƬ
			waitKey(200);//��ͣ0.5S		
		}
	}
	int total = image_points_seq.size();//���нǵ����
	m_str.Format("total =  %d", total);
	pEdit->AddString(m_str);
	int CornerNum = board_size.width*board_size.height;  //ÿ��ͼƬ���ܵĽǵ���
	for (int ii = 0; ii < total; ii++)
	{

		m_str.Format("�� %d ��ͼƬ������", ii + 1);
		pEdit->AddString(m_str);

		//������еĽǵ�
		m_str.Format("x = %f,y = %f", image_points_seq[ii][1].x, image_points_seq[ii][1].y);
		pEdit->AddString(m_str);
	}

	pEdit->AddString("�ǵ���ȡ���!");


	//������������궨
	pEdit->AddString("��ʼ�궨");
	/*������ά��Ϣ*/
	Size square_size = Size(30, 30);  /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С ,��λ��mm��===============�궨�����*/
	vector<vector<Point3f>> object_points; /* ����궨���Ͻǵ����ά���� */
	/*�������*/
	//cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ������ڲ������� */
	vector<int> point_counts;  // ÿ��ͼ���нǵ������
	//distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* �������5������ϵ����k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* ÿ��ͼ�����ת���� */
	vector<Mat> rvecsMat; /* ÿ��ͼ���ƽ������ */
	 /* ��ʼ���궨���Ͻǵ����ά���� */
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* ��ʼ�궨 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	pEdit->AddString("�궨��ɣ�");


	//�Ա궨�����������
	pEdit->AddString("��ʼ���۱궨�����");
	double total_err = 0.0; /* ����ͼ���ƽ�������ܺ� */
	double err = 0.0; /* ÿ��ͼ���ƽ����� */
	vector<Point2f> image_points2; /* �������¼���õ���ͶӰ�� */
	fout << "ÿ��ͼ��ı궨��\n";
	for (i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;
	pEdit->AddString("������ɣ�");


	//���涨����  	
	pEdit->AddString("��ʼ���涨������");
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */
	fout << "����ڲ�������" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "����ϵ����\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << tvecsMat[i] << endl;
		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		Rodrigues(tvecsMat[i], rotation_matrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotation_matrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	pEdit->AddString("��ɱ��棡");
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	fout << endl;
}
	/************************************************************************
	��ʾ�����������������Ľ��
	*************************************************************************/
//	Mat mapx = Mat(image_size, CV_32FC1);
//	Mat mapy = Mat(image_size, CV_32FC1);
//	Mat R = Mat::eye(3, 3, CV_32F);
//	pEdit->AddString("�������ͼ��");
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
//		//��һ�ֲ���Ҫת������ķ�ʽ
//		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);
//		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
//		imshow("ԭʼͼ��", imageSource);
//		imshow("������ͼ��", newimage);
//		waitKey(500);//��ͣ0.5S		
//		StrStm.clear();
//		filePath.clear();
//		StrStm << i + 1;
//		StrStm >> imageFileName;
//		imageFileName += "_d.jpg";
//		imwrite(imageFileName, newimage);
//	}
//	pEdit->AddString("���������");
//}