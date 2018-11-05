#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include "solvepnp.h"
#include "calibrate.h"
#include "imageXY.h"
#include "LEDLocalizationDlg.h"//��ʱ�ã���listbox��ʾcalibrate�Ľ���
#include "resource.h"//��ʱ�ã����ÿؼ���

using namespace std;
double Cx_pre7, Cy_pre7, Cz_pre7, Cx_pre8, Cy_pre8, Cz_pre8, 
	   Cx_pre9, Cy_pre9, Cz_pre9, Cx_pre10, Cy_pre10, Cz_pre10,
	   Cx_pre11, Cy_pre11, Cz_pre11, Cx_pre12, Cy_pre12, Cz_pre12,
	   Cx_pre13, Cy_pre13, Cz_pre13, Cx_pre14, Cy_pre14, Cz_pre14, 
	   Cx_pre15, Cy_pre15, Cz_pre15;
double Cx_asymmetric4, Cy_asymmetric4, Cz_asymmetric4, 
	   Cx_asymmetric5, Cy_asymmetric5, Cz_asymmetric5,
	   Cx_asymmetric6, Cy_asymmetric6, Cz_asymmetric6, 
	   Cx_asymmetric7, Cy_asymmetric7, Cz_asymmetric7, 
	   Cx_asymmetric8, Cy_asymmetric8, Cz_asymmetric8,
	   Cx_asymmetric9, Cy_asymmetric9, Cz_asymmetric9,
	   Cx_asymmetric10, Cy_asymmetric10, Cz_asymmetric10,
       Cx_asymmetric11, Cy_asymmetric11, Cz_asymmetric11;

double Cx_asymmetric6_3, Cy_asymmetric6_3, Cz_asymmetric6_3,
	   Cx_asymmetric6_2, Cy_asymmetric6_2, Cz_asymmetric6_2,
	   Cx_asymmetric6_1, Cy_asymmetric6_1, Cz_asymmetric6_1,
	   Cx_asymmetric6_0, Cy_asymmetric6_0, Cz_asymmetric6_0;

double Cx_iter, Cy_iter, Cz_iter,
	   Cx_P3P, Cy_P3P, Cz_P3P;

double Cx_new15, Cy_new15, Cz_new15,
		Cx_LINE9, Cy_LINE9, Cz_LINE9,
		Cx_LINE7, Cy_LINE7, Cz_LINE7,
		Cx_L14, Cy_L14, Cz_L14,
		Cx_Z13, Cy_Z13, Cz_Z13,
		Cx_ANGLE7, Cy_ANGLE7, Cz_ANGLE7, Cx_ANGLE15, Cy_ANGLE15, Cz_ANGLE15,
		Cx_ANGLE9, Cy_ANGLE9, Cz_ANGLE9;
double thetax_new15, thetay_new15, thetaz_new15;

//���ռ����Z����ת
//������� x yΪ�ռ��ԭʼx y����
//thetazΪ�ռ����Z����ת���ٶȣ��Ƕ��Ʒ�Χ��-180��180
//outx outyΪ��ת��Ľ������
void codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
	double x1 = x;//����������һ�Σ���֤&x == &outx���������Ҳ�ܼ�����ȷ
	double y1 = y;
	double rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
}

//���ռ����Y����ת
//������� x zΪ�ռ��ԭʼx z����
//thetayΪ�ռ����Y����ת���ٶȣ��Ƕ��Ʒ�Χ��-180��180
//outx outzΪ��ת��Ľ������
void codeRotateByY(double x, double z, double thetay, double& outx, double& outz)
{
	double x1 = x;
	double z1 = z;
	double ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
}

//���ռ����X����ת
//������� y zΪ�ռ��ԭʼy z����
//thetaxΪ�ռ����X����ת���ٶȣ��Ƕ��ƣ���Χ��-180��180
//outy outzΪ��ת��Ľ������
void codeRotateByX(double y, double z, double thetax, double& outy, double& outz)
{
	double y1 = y;//����������һ�Σ���֤&y == &y���������Ҳ�ܼ�����ȷ
	double z1 = z;
	double rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}


//��������������ת������ϵ
//�������old_x��old_y��old_zΪ��תǰ�ռ�������
//vx��vy��vzΪ��ת������
//thetaΪ��ת�ǶȽǶ��ƣ���Χ��-180��180
//����ֵΪ��ת�������
cv::Point3f RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta)
{
	double r = theta * CV_PI / 180;
	double c = cos(r);
	double s = sin(r);
	double new_x = (vx*vx*(1 - c) + c) * old_x + (vx*vy*(1 - c) - vz * s) * old_y + (vx*vz*(1 - c) + vy * s) * old_z;
	double new_y = (vy*vx*(1 - c) + vz * s) * old_x + (vy*vy*(1 - c) + c) * old_y + (vy*vz*(1 - c) - vx * s) * old_z;
	double new_z = (vx*vz*(1 - c) - vy * s) * old_x + (vy*vz*(1 - c) + vx * s) * old_y + (vz*vz*(1 - c) + c) * old_z;
	return cv::Point3f(new_x, new_y, new_z);
}
//�ȽϺ����������Ԫ������Ҫ��vector�洢������һ��
bool compare_x(cv::Point2f a, cv::Point2f b)
{
	return a.x<b.x; //��������
}
bool compare_y(cv::Point2f a, cv::Point2f b)
{
	return a.y<b.y; //��������
}

void CvSlovePNP::Test()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);
	std::vector<cv::Point2f> Points2D;
	for (int i = 5; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
		m_str.Format("%f", Points2D[i-5].x);
		pEdit->AddString(m_str);
	}

}
void CvSlovePNP::SloveEPNP_ANGLE7()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_y);//��y����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 5, compare_x);//y����ֵ��С��5���㣬����x����ֵ��������
	std::sort(Points2D.begin() + 6, Points2D.begin() + 11, compare_x);
	std::sort(Points2D.begin() + 12, Points2D.begin() + 17, compare_x);
	std::sort(Points2D.begin() + 18, Points2D.begin() + 23, compare_x);
	

	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));		//P5
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));			//P6
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));			//P7
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));		//P9
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));			//P10
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));			//P13
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));		//P15
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));			//P19
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));		//P21
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

																	
	//ȡ�м�7����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 2);
	Points2D.erase(Points2D.begin() + 1, Points2D.begin() + 3);
	Points2D.erase(Points2D.begin() + 2, Points2D.begin() + 4);
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 4, Points2D.begin() + 6);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 7);
	Points2D.erase(Points2D.begin() + 6, Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 7, Points2D.end());
	

	Points3D.erase(Points3D.begin(), Points3D.begin() + 2);
	Points3D.erase(Points3D.begin() + 1, Points3D.begin() + 3);
	Points3D.erase(Points3D.begin() + 2, Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 4, Points3D.begin() + 6);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 7);
	Points3D.erase(Points3D.begin() + 6, Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 7, Points3D.end());

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_ANGLE7 - Cx), 2) + pow((Cy_ANGLE7 - Cy), 2) + pow((Cz_ANGLE7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_ANGLE7 = Cx;
	Cy_ANGLE7 = Cy;
	Cz_ANGLE7 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_ANGLE9()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_y);//��y����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 5, compare_x);//y����ֵ��С��5���㣬����x����ֵ��������
	std::sort(Points2D.begin() + 6, Points2D.begin() + 11, compare_x);
	std::sort(Points2D.begin() + 12, Points2D.begin() + 17, compare_x);
	std::sort(Points2D.begin() + 18, Points2D.begin() + 23, compare_x);


	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));		//P5
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));			//P6
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));			//P7
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));		//P9
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));			//P10
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));			//P13
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));		//P15
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));			//P19
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));		//P21
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25


																		//ȡ�м�7����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 2);
	Points2D.erase(Points2D.begin() + 1, Points2D.begin() + 3);
	Points2D.erase(Points2D.begin() + 2, Points2D.begin() + 4);
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 4, Points2D.begin() + 6);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 7);
	Points2D.erase(Points2D.begin() + 6, Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 7, Points2D.begin() + 9);


	Points3D.erase(Points3D.begin(), Points3D.begin() + 2);
	Points3D.erase(Points3D.begin() + 1, Points3D.begin() + 3);
	Points3D.erase(Points3D.begin() + 2, Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 4, Points3D.begin() + 6);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 7);
	Points3D.erase(Points3D.begin() + 6, Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 7, Points3D.begin() + 9);

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_ANGLE9 - Cx), 2) + pow((Cy_ANGLE9 - Cy), 2) + pow((Cz_ANGLE9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_ANGLE9 = Cx;
	Cy_ANGLE9 = Cy;
	Cz_ANGLE9 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_ANGLE15()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_y);//��y����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 5, compare_x);//y����ֵ��С��5���㣬����x����ֵ��������
	std::sort(Points2D.begin() + 6, Points2D.begin() + 11, compare_x);
	std::sort(Points2D.begin() + 12, Points2D.begin() + 17, compare_x);
	std::sort(Points2D.begin() + 18, Points2D.begin() + 23, compare_x);


	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));		//P5
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));			//P6
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));			//P7
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));		//P9
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));			//P10
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));			//P13
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));		//P15
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));			//P19
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));		//P21
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25


																		//ȡ�м�7����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 1);
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 4);
	Points2D.erase(Points2D.begin() + 4, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 7, Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 8, Points2D.begin() + 9);
	Points2D.erase(Points2D.begin() + 11, Points2D.begin() + 12);
	Points2D.erase(Points2D.begin() + 12, Points2D.begin() + 13);
	Points2D.erase(Points2D.begin() + 15, Points2D.end());


	Points3D.erase(Points3D.begin(), Points3D.begin() + 1);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 4, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 7, Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 8, Points3D.begin() + 9);
	Points3D.erase(Points3D.begin() + 11, Points3D.begin() + 12);
	Points3D.erase(Points3D.begin() + 12, Points3D.begin() + 13);
	Points3D.erase(Points3D.begin() + 15, Points3D.end());

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_ANGLE15 - Cx), 2) + pow((Cy_ANGLE15 - Cy), 2) + pow((Cz_ANGLE15 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_ANGLE15 = Cx;
	Cy_ANGLE15 = Cy;
	Cz_ANGLE15 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_NEW15()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 4, compare_y);//x����ֵ��С��4���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 4, Points2D.begin() + 8, compare_y);
	std::sort(Points2D.begin() + 8, Points2D.begin() + 17, compare_y);
	std::sort(Points2D.begin() + 17, Points2D.begin() + 21, compare_y);
	std::sort(Points2D.begin() + 21, Points2D.end(), compare_y);//x����ֵ����4���㣬����y����ֵ��������
	//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
	//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));		//P7
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));		//P13
	Points3D.push_back(cv::Point3f(1.2543, 296.5173,98.4156));		//P19

	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20

	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));		//P6
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));	//P9
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));//P15
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));	//P21
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));		//P10
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22

	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));			//P5
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23
	
	//Points3D.push_back(cv::Point3f(47.5789, 407.1668, 169.8168));//P1 ��ά����ĵ�λ�Ǻ���
	//Points3D.push_back(cv::Point3f(141.3161, 392.2496, 182.3110));		//P7
	//Points3D.push_back(cv::Point3f(240.0176, 378.1010, 192.4528));		//P13
	//Points3D.push_back(cv::Point3f(339.3242, 363.0495, 199.3860));		//P19

	//Points3D.push_back(cv::Point3f(40.2834, 345.5651, 119.2408));		//P2
	//Points3D.push_back(cv::Point3f(136.3951, 331.3753, 130.6281));		//P8
	//Points3D.push_back(cv::Point3f(234.6343, 317.7378, 140.4371));		//P14
	//Points3D.push_back(cv::Point3f(333.8219, 301.8001, 147.9250));		//P20

	//Points3D.push_back(cv::Point3f(33.4381, 283.6361, 68.9792));		//P3
	//Points3D.push_back(cv::Point3f(65.6843, 215.3452, 149.7186));		//P6
	//Points3D.push_back(cv::Point3f(131.9004, 270.1303, 79.1423));	//P9
	//Points3D.push_back(cv::Point3f(164.1084, 200.7185, 158.3707));		//P12
	//Points3D.push_back(cv::Point3f(229.9553, 256.4955, 88.7762));	//P15
	//Points3D.push_back(cv::Point3f(263.0007, 186.5913, 167.5965));		//P18
	//Points3D.push_back(cv::Point3f(328.2006, 240.5560, 96.7608));	//P21
	//Points3D.push_back(cv::Point3f(361.1669, 171.9028, 176.1964));		//P24
	//Points3D.push_back(cv::Point3f(410.5890, 164.8296, 180.7003));		//P25

	//Points3D.push_back(cv::Point3f(26.7349, 222.3241, 18.2446));			//P4
	//Points3D.push_back(cv::Point3f(126.5076, 209.9261, 27.0709));		//P10
	//Points3D.push_back(cv::Point3f(224.6109, 196.5607, 36.2336));		//P16
	//Points3D.push_back(cv::Point3f(322.3630, 180.1925, 44.9855));		//P22

	//Points3D.push_back(cv::Point3f(19.7389, 160.7924, -32.1913));			//P5
	//Points3D.push_back(cv::Point3f(122.0701, 149.2756, -24.5904));		//P11
	//Points3D.push_back(cv::Point3f(220.1413, 136.0962, -15.7854));		//P17
	//Points3D.push_back(cv::Point3f(316.9400, 119.3284, -6.7051));		//P23

	//15���㡪�����Ҹ�4�����м�7��
	Points2D.erase(Points2D.begin(), Points2D.begin() + 4);
	Points2D.erase(Points2D.begin()+11, Points2D.begin() + 13);
	Points2D.erase(Points2D.begin() + 15, Points2D.end());

	Points3D.erase(Points3D.begin(), Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 11, Points3D.begin() + 13);
	Points3D.erase(Points3D.begin() + 15, Points3D.end());

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_new15 - Cx), 2) + pow((Cy_new15 - Cy), 2) + pow((Cz_new15 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	m_str.Format("��ת�ǵı仯����x: %f,��y: %f,��z: %f", thetax_out - thetax_new15, thetay_out - thetay_new15, thetaz_out - thetaz_new15);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_new15 = Cx;
	Cy_new15 = Cy;
	Cz_new15 = Cz;
	thetax_new15 = thetax_out;
	thetay_new15 = thetay_out;
	thetaz_new15 = thetaz_out;
	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_LINE9()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 4, compare_y);//x����ֵ��С��4���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 4, Points2D.begin() + 8, compare_y);
	std::sort(Points2D.begin() + 8, Points2D.begin() + 17, compare_y);
	std::sort(Points2D.begin() + 17, Points2D.begin() + 21, compare_y);
	std::sort(Points2D.begin() + 21, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));		//P7
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));		//P13
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));		//P19

	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20

	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));		//P6
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));	//P9
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));//P15
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));	//P21
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));		//P10
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22

	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));			//P5
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23

																		//Points3D.push_back(cv::Point3f(47.5789, 407.1668, 169.8168));//P1 ��ά����ĵ�λ�Ǻ���
																		//Points3D.push_back(cv::Point3f(141.3161, 392.2496, 182.3110));		//P7
																		//Points3D.push_back(cv::Point3f(240.0176, 378.1010, 192.4528));		//P13
																		//Points3D.push_back(cv::Point3f(339.3242, 363.0495, 199.3860));		//P19

																		//Points3D.push_back(cv::Point3f(40.2834, 345.5651, 119.2408));		//P2
																		//Points3D.push_back(cv::Point3f(136.3951, 331.3753, 130.6281));		//P8
																		//Points3D.push_back(cv::Point3f(234.6343, 317.7378, 140.4371));		//P14
																		//Points3D.push_back(cv::Point3f(333.8219, 301.8001, 147.9250));		//P20

																		//Points3D.push_back(cv::Point3f(33.4381, 283.6361, 68.9792));		//P3
																		//Points3D.push_back(cv::Point3f(65.6843, 215.3452, 149.7186));		//P6
																		//Points3D.push_back(cv::Point3f(131.9004, 270.1303, 79.1423));	//P9
																		//Points3D.push_back(cv::Point3f(164.1084, 200.7185, 158.3707));		//P12
																		//Points3D.push_back(cv::Point3f(229.9553, 256.4955, 88.7762));	//P15
																		//Points3D.push_back(cv::Point3f(263.0007, 186.5913, 167.5965));		//P18
																		//Points3D.push_back(cv::Point3f(328.2006, 240.5560, 96.7608));	//P21
																		//Points3D.push_back(cv::Point3f(361.1669, 171.9028, 176.1964));		//P24
																		//Points3D.push_back(cv::Point3f(410.5890, 164.8296, 180.7003));		//P25

																		//Points3D.push_back(cv::Point3f(26.7349, 222.3241, 18.2446));			//P4
																		//Points3D.push_back(cv::Point3f(126.5076, 209.9261, 27.0709));		//P10
																		//Points3D.push_back(cv::Point3f(224.6109, 196.5607, 36.2336));		//P16
																		//Points3D.push_back(cv::Point3f(322.3630, 180.1925, 44.9855));		//P22

																		//Points3D.push_back(cv::Point3f(19.7389, 160.7924, -32.1913));			//P5
																		//Points3D.push_back(cv::Point3f(122.0701, 149.2756, -24.5904));		//P11
																		//Points3D.push_back(cv::Point3f(220.1413, 136.0962, -15.7854));		//P17
																		//Points3D.push_back(cv::Point3f(316.9400, 119.3284, -6.7051));		//P23

	//�м�9����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 9, Points2D.end());

	Points3D.erase(Points3D.begin(), Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 9, Points3D.end());
	
	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_LINE9 - Cx), 2) + pow((Cy_LINE9 - Cy), 2) + pow((Cz_LINE9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_LINE9 = Cx;
	Cy_LINE9 = Cy;
	Cz_LINE9 = Cz;
	
	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_LINE7()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 4, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 4, Points2D.begin() + 8, compare_y);
	std::sort(Points2D.begin() + 8, Points2D.begin() + 17, compare_y);
	std::sort(Points2D.begin() + 17, Points2D.begin() + 21, compare_y);
	std::sort(Points2D.begin() + 21, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));		//P7
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));		//P13
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));		//P19

	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20

	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));		//P6
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));	//P9
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));//P15
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));	//P21
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));		//P10
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22

	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));			//P5
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23

																		//Points3D.push_back(cv::Point3f(47.5789, 407.1668, 169.8168));//P1 ��ά����ĵ�λ�Ǻ���
																		//Points3D.push_back(cv::Point3f(141.3161, 392.2496, 182.3110));		//P7
																		//Points3D.push_back(cv::Point3f(240.0176, 378.1010, 192.4528));		//P13
																		//Points3D.push_back(cv::Point3f(339.3242, 363.0495, 199.3860));		//P19

																		//Points3D.push_back(cv::Point3f(40.2834, 345.5651, 119.2408));		//P2
																		//Points3D.push_back(cv::Point3f(136.3951, 331.3753, 130.6281));		//P8
																		//Points3D.push_back(cv::Point3f(234.6343, 317.7378, 140.4371));		//P14
																		//Points3D.push_back(cv::Point3f(333.8219, 301.8001, 147.9250));		//P20

																		//Points3D.push_back(cv::Point3f(33.4381, 283.6361, 68.9792));		//P3
																		//Points3D.push_back(cv::Point3f(65.6843, 215.3452, 149.7186));		//P6
																		//Points3D.push_back(cv::Point3f(131.9004, 270.1303, 79.1423));	//P9
																		//Points3D.push_back(cv::Point3f(164.1084, 200.7185, 158.3707));		//P12
																		//Points3D.push_back(cv::Point3f(229.9553, 256.4955, 88.7762));	//P15
																		//Points3D.push_back(cv::Point3f(263.0007, 186.5913, 167.5965));		//P18
																		//Points3D.push_back(cv::Point3f(328.2006, 240.5560, 96.7608));	//P21
																		//Points3D.push_back(cv::Point3f(361.1669, 171.9028, 176.1964));		//P24
																		//Points3D.push_back(cv::Point3f(410.5890, 164.8296, 180.7003));		//P25

																		//Points3D.push_back(cv::Point3f(26.7349, 222.3241, 18.2446));			//P4
																		//Points3D.push_back(cv::Point3f(126.5076, 209.9261, 27.0709));		//P10
																		//Points3D.push_back(cv::Point3f(224.6109, 196.5607, 36.2336));		//P16
																		//Points3D.push_back(cv::Point3f(322.3630, 180.1925, 44.9855));		//P22

																		//Points3D.push_back(cv::Point3f(19.7389, 160.7924, -32.1913));			//P5
																		//Points3D.push_back(cv::Point3f(122.0701, 149.2756, -24.5904));		//P11
																		//Points3D.push_back(cv::Point3f(220.1413, 136.0962, -15.7854));		//P17
																		//Points3D.push_back(cv::Point3f(316.9400, 119.3284, -6.7051));		//P23

	//�м�7����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 7, Points2D.end());

	Points3D.erase(Points3D.begin(), Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 7, Points3D.end());

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_LINE7 - Cx), 2) + pow((Cy_LINE7 - Cy), 2) + pow((Cz_LINE7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_LINE7 = Cx;
	Cy_LINE7 = Cy;
	Cz_LINE7 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_L14()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 4, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 4, Points2D.begin() + 8, compare_y);
	std::sort(Points2D.begin() + 8, Points2D.begin() + 17, compare_y);
	std::sort(Points2D.begin() + 17, Points2D.begin() + 21, compare_y);
	std::sort(Points2D.begin() + 21, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));		//P7
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));		//P13
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));		//P19

	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20

	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));		//P6
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));	//P9
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));//P15
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));	//P21
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));		//P10
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22

	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));			//P5
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23

																		//Points3D.push_back(cv::Point3f(47.5789, 407.1668, 169.8168));//P1 ��ά����ĵ�λ�Ǻ���
																		//Points3D.push_back(cv::Point3f(141.3161, 392.2496, 182.3110));		//P7
																		//Points3D.push_back(cv::Point3f(240.0176, 378.1010, 192.4528));		//P13
																		//Points3D.push_back(cv::Point3f(339.3242, 363.0495, 199.3860));		//P19

																		//Points3D.push_back(cv::Point3f(40.2834, 345.5651, 119.2408));		//P2
																		//Points3D.push_back(cv::Point3f(136.3951, 331.3753, 130.6281));		//P8
																		//Points3D.push_back(cv::Point3f(234.6343, 317.7378, 140.4371));		//P14
																		//Points3D.push_back(cv::Point3f(333.8219, 301.8001, 147.9250));		//P20

																		//Points3D.push_back(cv::Point3f(33.4381, 283.6361, 68.9792));		//P3
																		//Points3D.push_back(cv::Point3f(65.6843, 215.3452, 149.7186));		//P6
																		//Points3D.push_back(cv::Point3f(131.9004, 270.1303, 79.1423));	//P9
																		//Points3D.push_back(cv::Point3f(164.1084, 200.7185, 158.3707));		//P12
																		//Points3D.push_back(cv::Point3f(229.9553, 256.4955, 88.7762));	//P15
																		//Points3D.push_back(cv::Point3f(263.0007, 186.5913, 167.5965));		//P18
																		//Points3D.push_back(cv::Point3f(328.2006, 240.5560, 96.7608));	//P21
																		//Points3D.push_back(cv::Point3f(361.1669, 171.9028, 176.1964));		//P24
																		//Points3D.push_back(cv::Point3f(410.5890, 164.8296, 180.7003));		//P25

																		//Points3D.push_back(cv::Point3f(26.7349, 222.3241, 18.2446));			//P4
																		//Points3D.push_back(cv::Point3f(126.5076, 209.9261, 27.0709));		//P10
																		//Points3D.push_back(cv::Point3f(224.6109, 196.5607, 36.2336));		//P16
																		//Points3D.push_back(cv::Point3f(322.3630, 180.1925, 44.9855));		//P22

																		//Points3D.push_back(cv::Point3f(19.7389, 160.7924, -32.1913));			//P5
																		//Points3D.push_back(cv::Point3f(122.0701, 149.2756, -24.5904));		//P11
																		//Points3D.push_back(cv::Point3f(220.1413, 136.0962, -15.7854));		//P17
																		//Points3D.push_back(cv::Point3f(316.9400, 119.3284, -6.7051));		//P23


	//L��14����
	Points2D.erase(Points2D.begin() + 8, Points2D.begin() + 12);
	Points2D.erase(Points2D.begin() + 9, Points2D.begin() + 10);
	Points2D.erase(Points2D.begin() + 10, Points2D.begin() + 14);
	Points2D.erase(Points2D.begin() + 12, Points2D.begin() + 14);

	Points3D.erase(Points3D.begin() + 8, Points3D.begin() + 12);
	Points3D.erase(Points3D.begin() + 9, Points3D.begin() + 10);
	Points3D.erase(Points3D.begin() + 10, Points3D.begin() + 14);
	Points3D.erase(Points3D.begin() + 12, Points3D.begin() + 14);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_L14 - Cx), 2) + pow((Cy_L14 - Cy), 2) + pow((Cz_L14 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_L14 = Cx;
	Cy_L14 = Cy;
	Cz_L14 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}

void CvSlovePNP::SloveEPNP_Z13()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<25; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 4, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 4, Points2D.begin() + 8, compare_y);
	std::sort(Points2D.begin() + 8, Points2D.begin() + 17, compare_y);
	std::sort(Points2D.begin() + 17, Points2D.begin() + 21, compare_y);
	std::sort(Points2D.begin() + 21, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(0, 0, 101.8642));//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(1.7559, 95.6418, 98.0094));		//P7
	Points3D.push_back(cv::Point3f(2.8632, 195.8575, 97.2249));		//P13
	Points3D.push_back(cv::Point3f(1.2543, 296.5173, 98.4156));		//P19

	Points3D.push_back(cv::Point3f(-79.9799, -2.7329, 100.5745));		//P2
	Points3D.push_back(cv::Point3f(-78.2473, 95.0462, 98.4001));		//P8
	Points3D.push_back(cv::Point3f(-76.9879, 194.703, 98.1159));		//P14
	Points3D.push_back(cv::Point3f(-78.9254, 295.4227, 98.3087));		//P20

	Points3D.push_back(cv::Point3f(-159.9838, -4.947, 98.9121));		//P3
	Points3D.push_back(cv::Point3f(-158.652, 43.833, -0.29));		//P6
	Points3D.push_back(cv::Point3f(-158.3842, 94.9418, 98.4764));	//P9
	Points3D.push_back(cv::Point3f(-158.8815, 143.7132, -0.2839));		//P12
	Points3D.push_back(cv::Point3f(-157.2452, 194.4011, 98.2977));//P15
	Points3D.push_back(cv::Point3f(-158.335, 244.0331, -0.3264));		//P18
	Points3D.push_back(cv::Point3f(-158.9172, 294.2362, 97.96));	//P21
	Points3D.push_back(cv::Point3f(-158.6595, 343.6634, -0.3606));		//P24
	Points3D.push_back(cv::Point3f(-158.4649, 393.7914, -0.3087));		//P25

	Points3D.push_back(cv::Point3f(-239.8114, -7.152, 98.02));			//P4
	Points3D.push_back(cv::Point3f(-238.1499, 93.75, 99.5081));		//P10
	Points3D.push_back(cv::Point3f(-237.1045, 193.1765, 99.8652));		//P16
	Points3D.push_back(cv::Point3f(-238.6392, 292.6553, 98.5948));		//P22

	Points3D.push_back(cv::Point3f(-319.6314, -9.5875, 96.7155));			//P5
	Points3D.push_back(cv::Point3f(-317.9415, 93.6005, 100.1003));		//P11
	Points3D.push_back(cv::Point3f(-316.9851, 192.9369, 100.8416));		//P17
	Points3D.push_back(cv::Point3f(-318.6672, 291.5628, 98.9174));		//P23

																		//Points3D.push_back(cv::Point3f(47.5789, 407.1668, 169.8168));//P1 ��ά����ĵ�λ�Ǻ���
																		//Points3D.push_back(cv::Point3f(141.3161, 392.2496, 182.3110));		//P7
																		//Points3D.push_back(cv::Point3f(240.0176, 378.1010, 192.4528));		//P13
																		//Points3D.push_back(cv::Point3f(339.3242, 363.0495, 199.3860));		//P19

																		//Points3D.push_back(cv::Point3f(40.2834, 345.5651, 119.2408));		//P2
																		//Points3D.push_back(cv::Point3f(136.3951, 331.3753, 130.6281));		//P8
																		//Points3D.push_back(cv::Point3f(234.6343, 317.7378, 140.4371));		//P14
																		//Points3D.push_back(cv::Point3f(333.8219, 301.8001, 147.9250));		//P20

																		//Points3D.push_back(cv::Point3f(33.4381, 283.6361, 68.9792));		//P3
																		//Points3D.push_back(cv::Point3f(65.6843, 215.3452, 149.7186));		//P6
																		//Points3D.push_back(cv::Point3f(131.9004, 270.1303, 79.1423));	//P9
																		//Points3D.push_back(cv::Point3f(164.1084, 200.7185, 158.3707));		//P12
																		//Points3D.push_back(cv::Point3f(229.9553, 256.4955, 88.7762));	//P15
																		//Points3D.push_back(cv::Point3f(263.0007, 186.5913, 167.5965));		//P18
																		//Points3D.push_back(cv::Point3f(328.2006, 240.5560, 96.7608));	//P21
																		//Points3D.push_back(cv::Point3f(361.1669, 171.9028, 176.1964));		//P24
																		//Points3D.push_back(cv::Point3f(410.5890, 164.8296, 180.7003));		//P25

																		//Points3D.push_back(cv::Point3f(26.7349, 222.3241, 18.2446));			//P4
																		//Points3D.push_back(cv::Point3f(126.5076, 209.9261, 27.0709));		//P10
																		//Points3D.push_back(cv::Point3f(224.6109, 196.5607, 36.2336));		//P16
																		//Points3D.push_back(cv::Point3f(322.3630, 180.1925, 44.9855));		//P22

																		//Points3D.push_back(cv::Point3f(19.7389, 160.7924, -32.1913));			//P5
																		//Points3D.push_back(cv::Point3f(122.0701, 149.2756, -24.5904));		//P11
																		//Points3D.push_back(cv::Point3f(220.1413, 136.0962, -15.7854));		//P17
																		//Points3D.push_back(cv::Point3f(316.9400, 119.3284, -6.7051));		//P23

	//Z��13����
	Points2D.erase(Points2D.begin() + 1, Points2D.begin() + 3);
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 4);
	Points2D.erase(Points2D.begin() + 6, Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 7, Points2D.begin() + 9);
	Points2D.erase(Points2D.begin() + 8, Points2D.begin() + 10);
	Points2D.erase(Points2D.begin() + 10, Points2D.begin() + 11);
	Points2D.erase(Points2D.begin() + 12, Points2D.begin() + 14);

	Points3D.erase(Points3D.begin() + 1, Points3D.begin() + 3);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 6, Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 7, Points3D.begin() + 9);
	Points3D.erase(Points3D.begin() + 8, Points3D.begin() + 10);
	Points3D.erase(Points3D.begin() + 10, Points3D.begin() + 11);
	Points3D.erase(Points3D.begin() + 12, Points3D.begin() + 14);



	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_Z13 - Cx), 2) + pow((Cy_Z13 - Cy), 2) + pow((Cz_Z13 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_Z13 = Cx;
	Cy_Z13 = Cy;
	Cz_Z13 = Cz;
	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}


void CvSlovePNP::SloveP3P()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7


	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15
	//ѡȡP3,P6,7,P15��4���������ʮ������ΪP3P�����
	Points2D.erase(Points2D.begin(), Points2D.begin() + 2);
	Points2D.erase(Points2D.begin() + 1, Points2D.begin() + 3);
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 10);

	Points3D.erase(Points3D.begin(), Points3D.begin() + 2);
	Points3D.erase(Points3D.begin() + 1, Points3D.begin() + 3);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 10);

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

	//��ת��������ת����
	//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_P3P - Cx), 2) + pow((Cy_P3P - Cy), 2) + pow((Cz_P3P - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);

	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_P3P = Cx;
	Cy_P3P = Cy;
	Cz_P3P = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveIterative()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
	
	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15
	//ѡȡP2,P3,P14,P15��4����߿�ľ�����Ϊ�����������
	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 12);
	Points2D.erase(Points2D.begin(), Points2D.begin() + 1);
	Points2D.erase(Points2D.begin() + 2, Points2D.begin() + 3);

	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 12);
	Points3D.erase(Points3D.begin(), Points3D.begin() + 1);
	Points3D.erase(Points3D.begin() + 2, Points3D.begin() + 3);

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

	//��ת��������ת����
	//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_iter - Cx), 2) + pow((Cy_iter - Cy), 2) + pow((Cz_iter - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_iter = Cx;
	Cy_iter = Cy;
	Cz_iter = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

void CvSlovePNP::SloveEPNP7()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	
	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 7;//���Ʋ��ü�������PNP

	Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
	Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre7 - Cx), 2) + pow((Cy_pre7 - Cy), 2) + pow((Cz_pre7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre7 = Cx;
	Cy_pre7 = Cy;
	Cz_pre7 = Cz;


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/




	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);



	//test();
}

void CvSlovePNP::SloveEPNP8()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;
	
	
	for(int i = 0;i<15;i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin()+3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin()+3, Points2D.begin()+12, compare_y);
	std::sort(Points2D.begin()+12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
	//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
	//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
	//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
	//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 8;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

	//��ת��������ת����
	//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	
	/*************************************�˴�������������ת��END**********************************************/


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre8 - Cx),2)+ pow((Cy_pre8 - Cy), 2) + pow((Cz_pre8 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre8 = Cx;
	Cy_pre8 = Cy;
	Cz_pre8 = Cz;


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/




	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);



	//test();
}
void CvSlovePNP::SloveEPNP9()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 9;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre9 - Cx), 2) + pow((Cy_pre9 - Cy), 2) + pow((Cz_pre9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre9 = Cx;
	Cy_pre9 = Cy;
	Cz_pre9 = Cz;


	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/




	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);



	//test();
}
void CvSlovePNP::SloveEPNP10()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 10;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre10 - Cx), 2) + pow((Cy_pre10 - Cy), 2) + pow((Cz_pre10 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre10 = Cx;
	Cy_pre10 = Cy;
	Cz_pre10 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}

void CvSlovePNP::SloveEPNP11()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
	//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
	//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 11;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

	//��ת��������ת����
	//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre11 - Cx), 2) + pow((Cy_pre11 - Cy), 2) + pow((Cz_pre11 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre11 = Cx;
	Cy_pre11 = Cy;
	Cz_pre11 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}

void CvSlovePNP::SloveEPNP12()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 12;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre12 - Cx), 2) + pow((Cy_pre12 - Cy), 2) + pow((Cz_pre12 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre12 = Cx;
	Cy_pre12 = Cy;
	Cz_pre12 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP13()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 13;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre13 - Cx), 2) + pow((Cy_pre13 - Cy), 2) + pow((Cz_pre13 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre13 = Cx;
	Cy_pre13 = Cy;
	Cz_pre13 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP14()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 14;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre14 - Cx), 2) + pow((Cy_pre14 - Cy), 2) + pow((Cz_pre14 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre14 = Cx;
	Cy_pre14 = Cy;
	Cz_pre14 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP15()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	int number = 15;//���Ʋ��ü�������PNP
	if (number < 8)
	{
		Points2D.erase(Points2D.begin() + number, Points2D.end());
		Points3D.erase(Points3D.begin() + number, Points3D.end());
	}
	else
	{
		Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
		Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	}

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre15 - Cx), 2) + pow((Cy_pre15 - Cy), 2) + pow((Cz_pre15 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre15 = Cx;
	Cy_pre15 = Cy;
	Cz_pre15 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/

	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
}
void CvSlovePNP::SloveEPNP_asymmetric4()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 12);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 12);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric4 - Cx), 2) + pow((Cy_asymmetric4 - Cy), 2) + pow((Cz_asymmetric4 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric4 = Cx;
	Cy_asymmetric4 = Cy;
	Cz_asymmetric4 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric5()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������

	//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 4);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 11);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 4);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 11);
	

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

	//��ת��������ת����
	//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric5 - Cx), 2) + pow((Cy_asymmetric5 - Cy), 2) + pow((Cz_asymmetric5 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric5 = Cx;
	Cy_asymmetric5 = Cy;
	Cz_asymmetric5 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

void CvSlovePNP::SloveEPNP_asymmetric6()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0 , 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 10);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 10);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6 - Cx), 2) + pow((Cy_asymmetric6 - Cy), 2) + pow((Cz_asymmetric6 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6 = Cx;
	Cy_asymmetric6 = Cy;
	Cz_asymmetric6 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric7()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 6);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 9);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 6);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 9);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric7 - Cx), 2) + pow((Cy_asymmetric7 - Cy), 2) + pow((Cz_asymmetric7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric7 = Cx;
	Cy_asymmetric7 = Cy;
	Cz_asymmetric7 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric8()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 7);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 8);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 7);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 8);


	//m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	//pEdit->AddString(m_str);

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric8 - Cx), 2) + pow((Cy_asymmetric8 - Cy), 2) + pow((Cz_asymmetric8 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric8 = Cx;
	Cy_asymmetric8 = Cy;
	Cz_asymmetric8 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric9()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 8);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 7);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 8);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 7);

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric9 - Cx), 2) + pow((Cy_asymmetric9 - Cy), 2) + pow((Cz_asymmetric9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric9 = Cx;
	Cy_asymmetric9 = Cy;
	Cz_asymmetric9 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric10()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 9);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 6);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 9);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 6);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric10 - Cx), 2) + pow((Cy_asymmetric10 - Cy), 2) + pow((Cz_asymmetric10 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric10 = Cx;
	Cy_asymmetric10 = Cy;
	Cz_asymmetric10 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

void CvSlovePNP::SloveEPNP_asymmetric11()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 10);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 10);
	

	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric11 - Cx), 2) + pow((Cy_asymmetric11 - Cy), 2) + pow((Cz_asymmetric11 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric11 = Cx;
	Cy_asymmetric11 = Cy;
	Cz_asymmetric11 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

void CvSlovePNP::SloveEPNP_asymmetric6_3()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 5, Points2D.begin() + 10);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 5, Points3D.begin() + 10);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_3 - Cx), 2) + pow((Cy_asymmetric6_3 - Cy), 2) + pow((Cz_asymmetric6_3 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_3 = Cx;
	Cy_asymmetric6_3 = Cy;
	Cz_asymmetric6_3 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric6_2()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 6, Points2D.begin() + 11);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 6, Points3D.begin() + 11);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_2 - Cx), 2) + pow((Cy_asymmetric6_2 - Cy), 2) + pow((Cz_asymmetric6_2 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_2 = Cx;
	Cy_asymmetric6_2 = Cy;
	Cz_asymmetric6_2 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric6_1()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 7, Points2D.begin() + 12);
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 7, Points3D.begin() + 12);


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_1 - Cx), 2) + pow((Cy_asymmetric6_1 - Cy), 2) + pow((Cz_asymmetric6_1 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_1 = Cx;
	Cy_asymmetric6_1 = Cy;
	Cz_asymmetric6_1 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}
void CvSlovePNP::SloveEPNP_asymmetric6_0()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;

	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//��x����ֵ��С��������
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x����ֵ��С��3���㣬����y����ֵ��������
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x����ֵ����3���㣬����y����ֵ��������
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//ɾ��11������б�㣬4�㷽��
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//ɾ��9������б�㣬6�㷽��
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//ɾ��7������б�㣬8�㷽��
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//ɾ��5������б�㣬10�㷽��

																//��������������
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 ��ά����ĵ�λ�Ǻ���
	Points3D.push_back(cv::Point3f(-150, 0, 100));		//P2
	Points3D.push_back(cv::Point3f(-150, -150, 100));	//P3
	Points3D.push_back(cv::Point3f(0, 100, 0));			//P4
	Points3D.push_back(cv::Point3f(0, 40, 0));			//P5
	Points3D.push_back(cv::Point3f(0, -70, 0));			//P6
	Points3D.push_back(cv::Point3f(0, -195, 0));		//P7

	Points3D.push_back(cv::Point3f(0, -225, 0));		//P8
	Points3D.push_back(cv::Point3f(0, -255, 0));		//P19
	Points3D.push_back(cv::Point3f(0, -285, 0));		//P10
	Points3D.push_back(cv::Point3f(0, -315, 0));		//P11
	Points3D.push_back(cv::Point3f(0, -345, 0));		//P12

	Points3D.push_back(cv::Point3f(150, 150, 100));		//P13
	Points3D.push_back(cv::Point3f(150, 0, 100));		//P14
	Points3D.push_back(cv::Point3f(150, -150, 100));	//P15

	Points2D.erase(Points2D.begin() + 3, Points2D.begin() + 5);
	Points2D.erase(Points2D.begin() + 8, Points2D.end());
	Points3D.erase(Points3D.begin() + 3, Points3D.begin() + 5);
	Points3D.erase(Points3D.begin() + 8, Points3D.end());


	/*m_str.Format("2D�������: %d��3D�������: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//��ʼ���������
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//���ַ������
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//ʵ��������ƺ�ֻ����4��������������⣬5�����ǹ���4��ⲻ����ȷ�Ľ�
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//�÷�����������N��λ�˹���

																								//��ת��������ת����
																								//��ȡ��ת����
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rvec, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];

	/*************************************�˴�������������ת��**********************************************/
	//������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
	//��ת˳��Ϊz��y��x
	//ԭ������ӣ�
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "�����������ת�ǣ�" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("�����������ת�� x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************�˴�������������ת��END**********************************************/

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��**********************************************/
	/* ��ԭʼ����ϵ������תz��y��x������ת�󣬻�����������ϵ��ȫƽ�У���������ת������OcOw�������ת */
	/* ��������֪��������������ϵ��ȫƽ��ʱ��OcOw��ֵ */
	/* ��ˣ�ԭʼ����ϵÿ����ת��ɺ󣬶�����OcOw����һ�η�����ת�����տ��Եõ���������ϵ��ȫƽ��ʱ��OcOw */
	/* ����������-1������������ϵ����������� */
	/***********************************************************************************/

	//���ƽ�ƾ��󣬱�ʾ���������ϵԭ�㣬��������(x,y,z)�ߣ��͵�����������ϵԭ��
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z ΪΨһ���������ԭʼ����ϵ�µ�����ֵ
	//Ҳ��������OcOw���������ϵ�µ�ֵ
	double x = tx, y = ty, z = tz;

	//�������η�����ת
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//����������������ϵ�µ�λ������
	//������OcOw����������ϵ�µ�ֵ
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6 - Cx), 2) + pow((Cy_asymmetric6 - Cy), 2) + pow((Cz_asymmetric6 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "������������꣺" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("������������� x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("��⦤d:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6 = Cx;
	Cy_asymmetric6 = Cy;
	Cz_asymmetric6 = Cz;

	/*************************************�˴�������������ϵԭ��Oc����������ϵ�е�λ��END**********************************************/
	//��ͶӰ����λ�˽��Ƿ���ȷ
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

//�����ã��������
void test()
{
	//double x = 1, y = 2, z = 3;
	////codeRotateByZ(x, y, -90, x, y);
	////codeRotateByY(x, z, -90, x, z);
	////codeRotateByX(y, z, 90, y, z);
	////cout << endl << "   (1,2,3) -> (" << x << ',' << y << ',' << z << ")" << endl << endl;

	//if (1 == 1)
	//{
	//	double x1 = 0, y1 = 1, z1 = 2;

	//	vector<cv::Point3f> rotateAxis;//��ת����˳��
	//	rotateAxis.push_back(cv::Point3f(1, 0, 0));//��x��
	//	rotateAxis.push_back(cv::Point3f(0, 1, 0));//��y��
	//	rotateAxis.push_back(cv::Point3f(0, 0, 1));//��z��

	//	vector<double> theta;//��ת�ĽǶ�˳��
	//	theta.push_back(90);
	//	theta.push_back(-90);
	//	theta.push_back(-180);
	//	cv::Point3f p1 = RotateByVector(x1, y1, z1, rotateAxis[0].x, rotateAxis[0].y, rotateAxis[0].z, theta[0]);
	//	rotateAxis[1] = RotateByVector(rotateAxis[1].x, rotateAxis[1].y, rotateAxis[1].z, rotateAxis[0].x, rotateAxis[0].y, rotateAxis[0].z, theta[0]);
	//	rotateAxis[2] = RotateByVector(rotateAxis[2].x, rotateAxis[2].y, rotateAxis[2].z, rotateAxis[0].x, rotateAxis[0].y, rotateAxis[0].z, theta[0]);
	//	cv::Point3f p2 = RotateByVector(p1.x, p1.y, p1.z, rotateAxis[1].x, rotateAxis[1].y, rotateAxis[1].z, theta[1]);
	//	rotateAxis[2] = RotateByVector(rotateAxis[2].x, rotateAxis[2].y, rotateAxis[2].z, rotateAxis[1].x, rotateAxis[1].y, rotateAxis[1].z, theta[1]);
	//	cv::Point3f p3 = RotateByVector(p2.x, p2.y, p2.z, rotateAxis[2].x, rotateAxis[2].y, rotateAxis[2].z, theta[2]);
	//}

	//if (1 == 1)
	//{
	//	double x1 = 0, y1 = 1, z1 = 2;

	//	codeRotateByZ(x1, y1, -180, x1, y1);
	//	codeRotateByY(x1, z1, -90, x1, z1);
	//	codeRotateByX(y1, z1, 90, y1, z1);
	//	cout << x1;
	//}

	////cv::Point3f np = RotateByVector(x1, y1, z1, 1, 0, 0, 90);
	//////codeRotateByX(y1, z1, -90, y1, z1);
	//////codeRotateByZ(x1, y1, -90, x1, y1);
	//////codeRotateByY(x1, z1, -90, x1, z1);
	////codeRotateByX(y1, z1, 90, y1, z1);
	//////codeRotateByZ(x1, y1, -90, x1, y1);
}