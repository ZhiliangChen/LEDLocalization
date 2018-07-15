#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include "solvepnp.h"
#include "calibrate.h"
#include "imageXY.h"
#include "LEDLocalizationDlg.h"//临时用，在listbox显示calibrate的进度
#include "resource.h"//临时用，引用控件名

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



//将空间点绕Z轴旋转
//输入参数 x y为空间点原始x y坐标
//thetaz为空间点绕Z轴旋转多少度，角度制范围在-180到180
//outx outy为旋转后的结果坐标
void codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
	double x1 = x;//将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
	double y1 = y;
	double rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
}

//将空间点绕Y轴旋转
//输入参数 x z为空间点原始x z坐标
//thetay为空间点绕Y轴旋转多少度，角度制范围在-180到180
//outx outz为旋转后的结果坐标
void codeRotateByY(double x, double z, double thetay, double& outx, double& outz)
{
	double x1 = x;
	double z1 = z;
	double ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
}

//将空间点绕X轴旋转
//输入参数 y z为空间点原始y z坐标
//thetax为空间点绕X轴旋转多少度，角度制，范围在-180到180
//outy outz为旋转后的结果坐标
void codeRotateByX(double y, double z, double thetax, double& outy, double& outz)
{
	double y1 = y;//将变量拷贝一次，保证&y == &y这种情况下也能计算正确
	double z1 = z;
	double rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}


//点绕任意向量旋转，右手系
//输入参数old_x，old_y，old_z为旋转前空间点的坐标
//vx，vy，vz为旋转轴向量
//theta为旋转角度角度制，范围在-180到180
//返回值为旋转后坐标点
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
//比较函数，这里的元素类型要与vector存储的类型一致
bool compare_x(cv::Point2f a, cv::Point2f b)
{
	return a.x<b.x; //升序排列
}
bool compare_y(cv::Point2f a, cv::Point2f b)
{
	return a.y<b.y; //升序排列
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
void CvSlovePNP::SloveEPNP7()
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);

	std::vector<cv::Point2f> Points2D;
	std::vector<cv::Point2f> temp;


	for (int i = 0; i<15; i++)
	{
		Points2D.push_back(detectKeyPoint[i].pt);
	}
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 7;//控制采用几点个求解PNP

	Points2D.erase(Points2D.begin() + (number - 3), Points2D.begin() + 12);
	Points3D.erase(Points3D.begin() + (number - 3), Points3D.begin() + 12);
	

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre7 - Cx), 2) + pow((Cy_pre7 - Cy), 2) + pow((Cz_pre7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre7 = Cx;
	Cy_pre7 = Cy;
	Cz_pre7 = Cz;


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/




	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin()+3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin()+3, Points2D.begin()+12, compare_y);
	std::sort(Points2D.begin()+12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
	//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
	//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
	//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
	//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

	//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 8;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

	//旋转向量变旋转矩阵
	//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	
	/*************************************此处计算出相机的旋转角END**********************************************/


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre8 - Cx),2)+ pow((Cy_pre8 - Cy), 2) + pow((Cz_pre8 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre8 = Cx;
	Cy_pre8 = Cy;
	Cz_pre8 = Cz;


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/




	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 9;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre9 - Cx), 2) + pow((Cy_pre9 - Cy), 2) + pow((Cz_pre9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre9 = Cx;
	Cy_pre9 = Cy;
	Cz_pre9 = Cz;


	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/




	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 10;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre10 - Cx), 2) + pow((Cy_pre10 - Cy), 2) + pow((Cz_pre10 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre10 = Cx;
	Cy_pre10 = Cy;
	Cz_pre10 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
	//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
	//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 11;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

	//旋转向量变旋转矩阵
	//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre11 - Cx), 2) + pow((Cy_pre11 - Cy), 2) + pow((Cz_pre11 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre11 = Cx;
	Cy_pre11 = Cy;
	Cz_pre11 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 12;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre12 - Cx), 2) + pow((Cy_pre12 - Cy), 2) + pow((Cz_pre12 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre12 = Cx;
	Cy_pre12 = Cy;
	Cz_pre12 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 13;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre13 - Cx), 2) + pow((Cy_pre13 - Cy), 2) + pow((Cz_pre13 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre13 = Cx;
	Cy_pre13 = Cy;
	Cz_pre13 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 14;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre14 - Cx), 2) + pow((Cy_pre14 - Cy), 2) + pow((Cz_pre14 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre14 = Cx;
	Cy_pre14 = Cy;
	Cz_pre14 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	int number = 15;//控制采用几点个求解PNP
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);
	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);

	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_pre15 - Cx), 2) + pow((Cy_pre15 - Cy), 2) + pow((Cz_pre15 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_pre15 = Cx;
	Cy_pre15 = Cy;
	Cz_pre15 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric4 - Cx), 2) + pow((Cy_asymmetric4 - Cy), 2) + pow((Cz_asymmetric4 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric4 = Cx;
	Cy_asymmetric4 = Cy;
	Cz_asymmetric4 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列

	//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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
	

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

	//旋转向量变旋转矩阵
	//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric5 - Cx), 2) + pow((Cy_asymmetric5 - Cy), 2) + pow((Cz_asymmetric5 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric5 = Cx;
	Cy_asymmetric5 = Cy;
	Cz_asymmetric5 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6 - Cx), 2) + pow((Cy_asymmetric6 - Cy), 2) + pow((Cz_asymmetric6 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6 = Cx;
	Cy_asymmetric6 = Cy;
	Cz_asymmetric6 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric7 - Cx), 2) + pow((Cy_asymmetric7 - Cy), 2) + pow((Cz_asymmetric7 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric7 = Cx;
	Cy_asymmetric7 = Cy;
	Cz_asymmetric7 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	//m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	//pEdit->AddString(m_str);

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric8 - Cx), 2) + pow((Cy_asymmetric8 - Cy), 2) + pow((Cz_asymmetric8 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric8 = Cx;
	Cy_asymmetric8 = Cy;
	Cz_asymmetric8 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric9 - Cx), 2) + pow((Cy_asymmetric9 - Cy), 2) + pow((Cz_asymmetric9 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric9 = Cx;
	Cy_asymmetric9 = Cy;
	Cz_asymmetric9 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric10 - Cx), 2) + pow((Cy_asymmetric10 - Cy), 2) + pow((Cz_asymmetric10 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric10 = Cx;
	Cy_asymmetric10 = Cy;
	Cz_asymmetric10 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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
	

	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric11 - Cx), 2) + pow((Cy_asymmetric11 - Cy), 2) + pow((Cz_asymmetric11 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric11 = Cx;
	Cy_asymmetric11 = Cy;
	Cz_asymmetric11 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_3 - Cx), 2) + pow((Cy_asymmetric6_3 - Cy), 2) + pow((Cz_asymmetric6_3 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_3 = Cx;
	Cy_asymmetric6_3 = Cy;
	Cz_asymmetric6_3 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_2 - Cx), 2) + pow((Cy_asymmetric6_2 - Cy), 2) + pow((Cz_asymmetric6_2 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_2 = Cx;
	Cy_asymmetric6_2 = Cy;
	Cz_asymmetric6_2 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6_1 - Cx), 2) + pow((Cy_asymmetric6_1 - Cy), 2) + pow((Cz_asymmetric6_1 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6_1 = Cx;
	Cy_asymmetric6_1 = Cy;
	Cz_asymmetric6_1 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
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
	std::sort(Points2D.begin(), Points2D.end(), compare_x);//按x坐标值大小升序排列
	std::sort(Points2D.begin(), Points2D.begin() + 3, compare_y);//x坐标值最小的3个点，按照y坐标值升序排列
	std::sort(Points2D.begin() + 3, Points2D.begin() + 12, compare_y);
	std::sort(Points2D.begin() + 12, Points2D.end(), compare_y);//x坐标值最大的3个点，按照y坐标值升序排列
																//Points2D.erase(Points2D.begin() + 4, Points2D.end());//删除11个光标判别点，4点方案
																//Points2D.erase(Points2D.begin() + 6, Points2D.end());//删除9个光标判别点，6点方案
																//Points2D.erase(Points2D.begin() + 8, Points2D.end());//删除7个光标判别点，8点方案
																//Points2D.erase(Points2D.begin()+7, Points2D.begin() + 12);//删除5个光标判别点，10点方案

																//特征点世界坐标
	std::vector<cv::Point3f> Points3D;
	Points3D.push_back(cv::Point3f(-150, 150, 100));	//P1 三维坐标的单位是毫米
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


	/*m_str.Format("2D数组个数: %d，3D数组个数: %d", Points2D.size(), Points3D.size());
	pEdit->AddString(m_str);*/

	//初始化输出矩阵
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	//三种方法求解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);	//实测迭代法似乎只能用4个共面特征点求解，5个点或非共面4点解不出正确的解
	//solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4
	solvePnP(Points3D, Points2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);			//该方法可以用于N点位姿估计

																								//旋转向量变旋转矩阵
																								//提取旋转矩阵
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

	/*************************************此处计算出相机的旋转角**********************************************/
	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x
	//原理见帖子：
	double thetaz = atan2(r21, r11) / CV_PI * 180;
	double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
	double thetax = atan2(r32, r33) / CV_PI * 180;
	double thetaz_out = -thetaz;
	double thetay_out = -thetay;
	double thetax_out = -thetax;

	//ofstream fout("D:\\pnp_theta.txt");
	//fout << -1 * thetax << endl << -1 * thetay << endl << -1 * thetaz << endl;
	////cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
	//fout.close();

	m_str.Format("相机的三轴旋转角 x: %f, y: %f, z: %f", thetax_out, thetay_out, thetaz_out);
	pEdit->AddString(m_str);

	/*************************************此处计算出相机的旋转角END**********************************************/

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
	/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
	/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
	/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
	/* 该向量乘以-1就是世界坐标系下相机的坐标 */
	/***********************************************************************************/

	//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
	double tx = tvec.ptr<double>(0)[0];
	double ty = tvec.ptr<double>(0)[1];
	double tz = tvec.ptr<double>(0)[2];

	//x y z 为唯一向量在相机原始坐标系下的向量值
	//也就是向量OcOw在相机坐标系下的值
	double x = tx, y = ty, z = tz;

	//进行三次反向旋转
	codeRotateByZ(x, y, -1 * thetaz, x, y);
	codeRotateByY(x, z, -1 * thetay, x, z);
	codeRotateByX(y, z, -1 * thetax, y, z);


	//获得相机在世界坐标系下的位置坐标
	//即向量OcOw在世界坐标系下的值
	double Cx = x * -1;
	double Cy = y * -1;
	double Cz = z * -1;
	displacement = sqrt(pow((Cx_asymmetric6 - Cx), 2) + pow((Cy_asymmetric6 - Cy), 2) + pow((Cz_asymmetric6 - Cz), 2));
	//ofstream fout2("D:\\pnp_t.txt");
	//fout2 << Cx << std::endl << Cy << endl << Cz << endl;
	////cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
	//fout2.close();

	m_str.Format("相机的世界坐标 x: %lf, y: %lf, z: %lf", Cx, Cy, Cz);
	pEdit->AddString(m_str);

	m_str.Format("求解Δd:  %lf", displacement);
	pEdit->AddString(m_str);
	pEdit->SetCurSel(pEdit->GetCount() - 1);
	Cx_asymmetric6 = Cx;
	Cy_asymmetric6 = Cy;
	Cz_asymmetric6 = Cz;

	/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/
	//重投影测试位姿解是否正确
	std::vector<cv::Point2f> projectedPoints;
	Points3D.push_back(cv::Point3f(0, 100, 105));
	cv::projectPoints(Points3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

}

//测试用，不必理会
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

	//	vector<cv::Point3f> rotateAxis;//旋转的轴顺序
	//	rotateAxis.push_back(cv::Point3f(1, 0, 0));//先x轴
	//	rotateAxis.push_back(cv::Point3f(0, 1, 0));//先y轴
	//	rotateAxis.push_back(cv::Point3f(0, 0, 1));//再z轴

	//	vector<double> theta;//旋转的角度顺序
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