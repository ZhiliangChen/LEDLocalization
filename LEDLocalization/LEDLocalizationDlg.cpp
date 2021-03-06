
// LEDLocalizationDlg.cpp: 实现文件
//
#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include "LEDLocalization.h"
#include "LEDLocalizationDlg.h"
#include "afxdialogex.h"
#include <string>
#include "imageXY.h"
#include "calibrate.h"
#include "solvepnp.h"

#include <stdio.h>
#include "ros.h"
#include <geometry_msgs/Pose2D.h>
#include <windows.h>
#include "time.h"
using std::string;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
CvImageXY m_imageXY;
CWnd* g_pWnd;
CvCalibrate m_calibrate;
CvSlovePNP m_solvepnp;
IplImage*  m_pImg;


ros::NodeHandle nh;
geometry_msgs::Pose2D Pose2D_msg;
ros::Publisher pose2d_pub("agvpose", &Pose2D_msg);

int Thread_flag = 1;

// CLEDLocalizationDlg 对话框
CLEDLocalizationDlg::CLEDLocalizationDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_LEDLOCALIZATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	m_hFactory = NULL;
	m_hCam = NULL;
	m_hThread = NULL;
	m_BoardSize.width = 9;
	m_BoardSize.height = 5;
	m_SquareSize = 1.f;
	m_AspectRatio = 1.f;
	m_Flags = 0;
	m_ImageCount = 10;

	m_pImg = NULL;

	m_pUndistortMapX = 0;
	m_pUndistortMapY = 0;
	m_pImagePointsBuf = 0;
	m_pImagePointsSeq = 0;
	m_pStorage = 0;
	m_bSaveSettings = false;
	
	m_bSaveImages = false;
	
	m_bCameraOpen = false;
	m_bAcquisitionRunning = false;

	InitializeCriticalSection(&m_CriticalSection);
}

void CLEDLocalizationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LISTMSG, m_listmsg);
}

BEGIN_MESSAGE_MAP(CLEDLocalizationDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()

	ON_WM_DESTROY()
	ON_WM_TIMER()

	ON_BN_CLICKED(IDC_START, OnStart)
	ON_BN_CLICKED(IDC_IMAGEXY, OnImageXY)
	ON_BN_CLICKED(IDC_SOLVEPNP, OnSolvepnp)
	ON_BN_CLICKED(IDC_STOP, OnStop)
	ON_BN_CLICKED(IDC_CALIBRATE, OnCalibrate)
	ON_BN_CLICKED(IDC_TEST, OnTest)


	ON_BN_CLICKED(IDC_AGVCONNECT, OnAGVCONNECT)
END_MESSAGE_MAP()


// CLEDLocalizationDlg 消息处理程序
BOOL CLEDLocalizationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	g_pWnd = this;//全局变量获取窗口指针
	BOOL retval;//返回值

	// Open factory & camera，软件开启的同时打开相机？？？
	retval = OpenFactoryAndCamera();
	if (retval)
	{
		m_test.Format(CString((char*)m_sCameraId));// Display camera ID
		m_listmsg.AddString(m_test);
		
		InitializeControls();   // Initialize Controls
		m_bCameraOpen = true;
	}
	else
	{
		m_test.Format("error");
		m_listmsg.AddString(m_test);
	}

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CLEDLocalizationDlg::OnDestroy()
{
	CDialog::OnDestroy();

	// Stop acquisition
	OnStop();

	// Close factory & camera
	CloseFactoryAndCamera();
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CLEDLocalizationDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CLEDLocalizationDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}
void CLEDLocalizationDlg::OnTimer(UINT_PTR nIDEvent)
{
	// Update the GUI with latest processing time value
	if (nIDEvent == 1)
	{
		//// Update UI when calibrated flag changed
		//if (m_bCalibratedChanged)
		//{
		//	m_bCalibratedChanged = false;

		//}
		//if (m_bImagesSaved)
		//{
		//	m_bImagesSaved = false;
		//	AfxMessageBox(_T("Images saved to disk!"), MB_OK | MB_ICONINFORMATION);
		//}
		//if (m_bSettingsSaved)
		//{
		//	m_bSettingsSaved = false;
		//	AfxMessageBox(_T("Settings saved to disk!"), MB_OK | MB_ICONINFORMATION);
		//}
	}

	if (nIDEvent == 2)
	{
		clock_t start, finish;
		start = clock();
		m_imageXY.BlobDetector_AGV();
		//Sleep(200);
		m_solvepnp.SloveEPNP_AGV15();
		
		finish = clock();
		m_test.Format("%f seconds", (double)(finish - start)/CLOCKS_PER_SEC);
		m_listmsg.AddString(m_test);
		m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);

	}
	if (nIDEvent == 3)
	{
		/*geometry_msgs::Pose2D Pose2D_msg;
		ros::Publisher pose2d_pub("agvpose", &Pose2D_msg);
		nh.advertise(pose2d_pub);*/
		//Pose2D_msg.x = Cx_AGV;
		//Pose2D_msg.y = Cz_AGV;
		//Pose2D_msg.theta = thetay_out_AGV;
		//pose2d_pub.publish(&Pose2D_msg);
		//nh.spinOnce();
		////Cx = Cx_AGV;

		//m_test.Format("finish sending");
		//m_listmsg.AddString(m_test);
		//m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);

	}
	CDialog::OnTimer(nIDEvent);
}

// OpenFactoryAndCamera
BOOL CLEDLocalizationDlg::OpenFactoryAndCamera()
{
	J_STATUS_TYPE   retval;
	uint32_t        iSize;
	uint32_t        iNumDev;
	bool8_t         bHasChange;

	// Open factory
	retval = J_Factory_Open((int8_t*)"", &m_hFactory);
	if (retval != J_ST_SUCCESS)
	{
		AfxMessageBox(CString("Could not open factory!"));
		return FALSE;
	}
	TRACE("Opening factory succeeded\n");

	// Update camera list
	retval = J_Factory_UpdateCameraList(m_hFactory, &bHasChange);
	if (retval != J_ST_SUCCESS)
	{
		AfxMessageBox(CString("Could not update camera list!"), MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	TRACE("Updating camera list succeeded\n");

	// Get the number of Cameras
	retval = J_Factory_GetNumOfCameras(m_hFactory, &iNumDev);
	if (retval != J_ST_SUCCESS)
	{
		AfxMessageBox(CString("Could not get the number of cameras!"), MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	if (iNumDev == 0)
	{
		AfxMessageBox(CString("There is no camera!"), MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	TRACE("%d cameras were found\n", iNumDev);

	// Get camera ID
	iSize = (uint32_t)sizeof(m_sCameraId);
	retval = J_Factory_GetCameraIDByIndex(m_hFactory, 0, m_sCameraId, &iSize);
	if (retval != J_ST_SUCCESS)
	{
		AfxMessageBox(CString("Could not get the camera ID!"), MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	TRACE("Camera ID: %s\n", m_sCameraId);

	// Open camera
	retval = J_Camera_Open(m_hFactory, m_sCameraId, &m_hCam);
	if (retval != J_ST_SUCCESS)
	{
		AfxMessageBox(CString("Could not open the camera!"), MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}
	TRACE("Opening camera succeeded\n");

	return TRUE;
}
//--------------------------------------------------------------------------------------------------
// CloseFactoryAndCamera
//--------------------------------------------------------------------------------------------------
void CLEDLocalizationDlg::CloseFactoryAndCamera()
{
	if (m_hCam)
	{
		// Close camera
		J_Camera_Close(m_hCam);
		m_hCam = NULL;
		TRACE("Closed camera\n");
	}

	if (m_hFactory)
	{
		// Close factory
		J_Factory_Close(m_hFactory);
		m_hFactory = NULL;
		TRACE("Closed factory\n");
	}
}

// InitializeControls
//--------------------------------------------------------------------------------------------------
void CLEDLocalizationDlg::InitializeControls()
{
	J_STATUS_TYPE   retval;
	NODE_HANDLE hNode;

	retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_WIDTH, 2560);//(2560,2048)平均2.5，（1920,1536）平均1.3
	retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_HEIGHT, 2048);//（1280,1024）平均0.5s
	retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_GAIN, 10);//增益增大会提高感光度，使图片变亮，副作用是增加噪点
	retval = J_Camera_SetValueInt64(m_hCam, NODE_NAME_EXPOSURE, 10);//增大光圈，减小曝光时间
}
	

// StreamCBFunc
//--------------------------------------------------------------------------------------------------
void CLEDLocalizationDlg::StreamCBFunc(J_tIMAGE_INFO * pAqImageInfo)
{
	CString valueString;

	// Skip images if they start to queue up in order to avoid lag caused by the queue size.
	if (pAqImageInfo->iAwaitDelivery > 2)
		return;

	// We only want to create the OpenCV Image object once and we want to get the correct size from the Acquisition Info structure
	if (m_pImg == NULL)
	{
		// Create the Image:
		// We assume this is a 8-bit monochrome image in this sample
		m_pImg = cvCreateImage(cvSize(pAqImageInfo->iSizeX, pAqImageInfo->iSizeY), IPL_DEPTH_8U, 1);

		// Create Undistort maps
		//m_pUndistortMapX = cvCreateImage(cvSize(pAqImageInfo->iSizeX, pAqImageInfo->iSizeY), IPL_DEPTH_32F, 1);
		//m_pUndistortMapY = cvCreateImage(cvSize(pAqImageInfo->iSizeX, pAqImageInfo->iSizeY), IPL_DEPTH_32F, 1);

		m_ImgSize = cvGetSize(m_pImg);
	}

	
	// Copy the data from the Acquisition engine image buffer into the OpenCV Image obejct
	memcpy(m_pImg->imageData, pAqImageInfo->pImageBuffer, m_pImg->imageSize);
	
	//视频显示用
	//cvShowImage("circles", m_pImg);
	
	//if (m_bCalibrated && m_bSaveSettings)
	//{
	//	// save camera parameters in any case, to catch Inf's/NaN's
	//	SaveCameraParams("Settings.yml",
	//		m_ImageCount,
	//		m_ImgSize,
	//		m_BoardSize,
	//		m_SquareSize,
	//		m_AspectRatio,
	//		m_Flags,
	//		&m_CameraMatrix,
	//		&m_DistCoeffsMatrix,
	//		m_pExtrParamsMatrix,
	//		m_pImagePointsSeq,
	//		m_pReprojErrsMatrix,
	//		m_AvgReprojRrr);
	//	m_bSaveSettings = false;
	//	m_bSettingsSaved = true;
	//}

}


void CLEDLocalizationDlg::OnStart()
{
	J_STATUS_TYPE   retval;
	int64_t int64Val;
	int64_t pixelFormat;

	SIZE	ViewSize;
	POINT	TopLeft;

	// Get Width from the camera
	retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_WIDTH, &int64Val);
	ViewSize.cx = (LONG)int64Val;     // Set window size cx

									  // Get Height from the camera
	retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_HEIGHT, &int64Val);
	ViewSize.cy = (LONG)int64Val;     // Set window size cy

									  // Get pixelformat from the camera
	retval = J_Camera_GetValueInt64(m_hCam, NODE_NAME_PIXELFORMAT, &int64Val);
	pixelFormat = int64Val;

	// Calculate number of bits (not bytes) per pixel using macro
	int bpp = J_BitsPerPixel(pixelFormat);

	// Set window position
	TopLeft.x = 100;
	TopLeft.y = 50;

	// Open stream
	retval = J_Image_OpenStream(m_hCam, 0, reinterpret_cast<J_IMG_CALLBACK_OBJECT>(this), reinterpret_cast<J_IMG_CALLBACK_FUNCTION>(&CLEDLocalizationDlg::StreamCBFunc), &m_hThread, (ViewSize.cx*ViewSize.cy*bpp) / 8);
	if (retval != J_ST_SUCCESS) {
		AfxMessageBox(CString("Could not open stream!"), MB_OK | MB_ICONEXCLAMATION);
		return;
	}
	TRACE("Opening stream succeeded\n");

	// Start Acquision
	retval = J_Camera_ExecuteCommand(m_hCam, NODE_NAME_ACQSTART);

	m_bAcquisitionRunning = true;
	//视频显示用
	//SetTimer(1, 500, NULL);
	//cvNamedWindow("circles", 0);

	m_listmsg.AddString("USB camera start acquision");
	//	DWORD nRet=MV_Usb2SetThreadAffinityMask(m_hMVC1000,2);
	//选中listbox中的最后一行
	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
}

void CLEDLocalizationDlg::OnStop()
{
	J_STATUS_TYPE retval;

	// Stop Acquision
	if (m_hCam) {
		retval = J_Camera_ExecuteCommand(m_hCam, NODE_NAME_ACQSTOP);
	}

	if (m_hThread)
	{
		// Close stream
		retval = J_Image_CloseStream(m_hThread);
		m_hThread = NULL;
		TRACE("Closed stream\n");
	}

	cvDestroyAllWindows();//销毁所有HighGUI窗口

	if (m_pImg != NULL)
	{
		cvReleaseImage(&m_pImg);
		m_pImg = NULL;
	}

	if (m_pStorage)
	{
		cvReleaseMemStorage(&m_pStorage);
		m_pStorage = 0;
	}

	m_bAcquisitionRunning = false;
	//EnableControls();
	//KillTimer(1);
	KillTimer(2);

	
	m_listmsg.AddString("USB camera stop capture");
	//选中listbox中的最后一行
	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
}





void CLEDLocalizationDlg::OnImageXY()
{

	int i;
	float point_num;
	
	//m_imageXY.ShowImage();
	//m_imageXY.BlobDetector_NEW();
	m_imageXY.BlobDetector_static();
	//m_imageXY.BlobDetector_test();
	Sleep(110);//？有必要吗
	
	//std::vector<cv::KeyPoint>().swap(detectKeyPoint);//清空detectKeyPoint
	//point_num = detectKeyPoint.size();//实时输出size，看看是不是都是25个点
	//for (i = 0; i < 25; i++)
	//{
	//	m_test.Format("%f;%f", detectKeyPoint[i].pt.x, detectKeyPoint[i].pt.y);
	//	//m_test.Format("%f;%f", point_num, detectKeyPoint[i].pt.y);
	//	m_listmsg.AddString(m_test);
	//}
	////选中listbox中的最后一行
	//m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);

}
void CLEDLocalizationDlg::OnSolvepnp()
{

	/*m_solvepnp.SloveEPNP_NEW15();
	m_solvepnp.SloveEPNP_LINE9();
	m_solvepnp.SloveEPNP_LINE7();
	m_solvepnp.SloveEPNP_L14();
	m_solvepnp.SloveEPNP_Z13();*/
	//m_solvepnp.SloveEPNP_ANGLE7();
	//m_solvepnp.SloveEPNP_ANGLE9();
	m_solvepnp.SloveEPNP_ANGLE15();
	//m_solvepnp.SloveEPNP_AGV15();
}


void CLEDLocalizationDlg::OnCalibrate()
{
	m_calibrate.Calibrate();
}
void CLEDLocalizationDlg::OnTest()
{
	AfxBeginThread(Thread2, this);
	//Sleep(1000);
	//AfxBeginThread(Thread1, this, THREAD_PRIORITY_IDLE);
	
	//m_imageXY.Test();
	//SetTimer(2, 100, NULL);
	//Sleep(1000);
	//SetTimer(3, 200, NULL);
}

void CLEDLocalizationDlg::OnAGVCONNECT()
{
	
	Thread_flag = 0;
	//clock_t start, finish;
	//start = clock();

	//
	//m_imageXY.BlobDetector_static();
	//m_solvepnp.SloveEPNP_ANGLE15();
	//finish = clock();
	//m_test.Format("%f seconds", (double)(finish - start) / CLOCKS_PER_SEC);
	//m_listmsg.AddString(m_test);
	//m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);

}
UINT CLEDLocalizationDlg::Thread1(void *param)
{
	//ros::NodeHandle nh;
	char *ros_master = "192.168.31.200"; // EAI ROS IP
	nh.initNode(ros_master);

	//m_test.Format("Connecting to server at %s\n", ros_master);
	//m_listmsg.AddString(m_test);
	//m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);

	//ros::Publisher pose2d_pub("agvpose", &Pose2D_msg);
	nh.advertise(pose2d_pub);
	while(Thread_flag != 0)
	{
	//	if (Thread_flag = 1)
	//	{
			Pose2D_msg.x = Cx_AGV;
			Pose2D_msg.y = Cz_AGV;
			Pose2D_msg.theta = thetay_out_AGV;
			pose2d_pub.publish(&Pose2D_msg);
			nh.spinOnce();
	//	}
		/*if (Thread_flag = 2)
		{
			Pose2D_msg.x = 9999;
			Pose2D_msg.y = 9999;
			Pose2D_msg.theta = 0;
			pose2d_pub.publish(&Pose2D_msg);
			nh.spinOnce();
			Thread_flag = 1;
		}*/
		
		//Cx = Cx_AGV;

		/*m_test.Format("finish sending");
		m_listmsg.AddString(m_test);
		m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);*/
		Sleep(100);
	}
	return 0;
}

UINT CLEDLocalizationDlg::Thread2(void *param)
{
	CListBox *pEdit = (CListBox*)g_pWnd->GetDlgItem(IDC_LISTMSG);
	CString m_listmsg;
	float point_num;

	while (Thread_flag != 0)
	{
		clock_t start, finish;
		start = clock();

		m_imageXY.BlobDetector_AGV();
		Sleep(100);//是不是sleep100时间太短，来不及detect?
		point_num = detectKeyPoint.size();//实时输出size，看看是不是都是25个点
		m_solvepnp.SloveEPNP_AGV15();
		//Sleep(50);
		finish = clock();
		m_listmsg.Format("%f seconds", (double)(finish - start) / CLOCKS_PER_SEC);
		pEdit->AddString(m_listmsg);

		m_listmsg.Format("keypoint 个数 %f ", point_num);
		pEdit->AddString(m_listmsg);

		pEdit->SetCurSel(pEdit->GetCount() - 1);
	}
	return 0;
}

//=====================================以下是采用MVC-1000F相机所用的代码======================================================
//#include "stdafx.h"
//#include "LEDLocalization.h"
//#include "LEDLocalizationDlg.h"
//#include "afxdialogex.h"
//#include <string>
//#include <highgui.h>
//#include "imageXY.h"
//#include "calibrate.h"
//#include "solvepnp.h"
//
//
//#ifdef _DEBUG
//#define new DEBUG_NEW
//#endif
//CvImageXY m_imageXY;
//CWnd* g_pWnd;
//CvCalibrate m_calibrate;
//CvSlovePNP m_solvepnp;
//
//// CLEDLocalizationDlg 对话框
//CLEDLocalizationDlg::CLEDLocalizationDlg(CWnd* pParent /*=nullptr*/)
//	: CDialogEx(IDD_LEDLOCALIZATION_DIALOG, pParent)
//{
//	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
//}
//
//void CLEDLocalizationDlg::DoDataExchange(CDataExchange* pDX)
//{
//	CDialogEx::DoDataExchange(pDX);
//	DDX_Control(pDX, IDC_LISTMSG, m_listmsg);
//}
//
//BEGIN_MESSAGE_MAP(CLEDLocalizationDlg, CDialogEx)
//	ON_WM_PAINT()
//	ON_WM_QUERYDRAGICON()
//
//	ON_BN_CLICKED(IDC_START,OnStart)
//	ON_BN_CLICKED(IDC_IMAGEXY, OnImageXY)
//	ON_BN_CLICKED(IDC_SOLVEPNP, OnSolvepnp)
//	ON_BN_CLICKED(IDC_STOP, OnStop)
//	ON_BN_CLICKED(IDC_CALIBRATE, OnCalibrate)
//	ON_BN_CLICKED(IDC_TEST, OnTest)
//	
//
//END_MESSAGE_MAP()
//
//
//// CLEDLocalizationDlg 消息处理程序
//BOOL CLEDLocalizationDlg::OnInitDialog()
//{
//	CDialogEx::OnInitDialog();
//
//	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
//	//  执行此操作
//	SetIcon(m_hIcon, TRUE);			// 设置大图标
//	SetIcon(m_hIcon, FALSE);		// 设置小图标
//
//	// TODO: 在此添加额外的初始化代码
//	g_pWnd = this;//全局变量获取窗口指针
//	DWORD RGBDataSize = 1280 * 1024 * 3;
//	m_pRGBData = (BYTE*)malloc(RGBDataSize * sizeof(BYTE));
//	m_RGBData = (BYTE*)malloc(RGBDataSize * sizeof(BYTE));//全局变量
//	memset(m_pRGBData, 0, RGBDataSize);
//	memset(m_RGBData, 0, RGBDataSize);//全局变量
//	DWORD RawDataSize = 1280 * 1024;
//	m_pRawData = (BYTE*)malloc(RawDataSize * sizeof(BYTE));
//	memset(m_pRawData, 0, RawDataSize);
//	m_bRawSave = FALSE;
//	m_bRawToRGB = FALSE;
//	m_hMVC1000 = NULL;
//	InitImageParam();
//	m_nOpMode = 0;//采集模式，0为连续模式；1为外触发模式
//	//手动初始化相机内外参数，以后改自动,暂时失败
//	//cameraMatrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 摄像机内参数矩阵 */
//	//distCoeffs = cv::Mat(5, 1, CV_32FC1, cv::Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
//	//cameraMatrix.at<double>(0, 0) = 1152.195197178761;
//	//cameraMatrix.at<double>(0, 1) = 0;
//	//cameraMatrix.at<double>(0, 2) = 611.574705796087;
//	//cameraMatrix.at<double>(1, 0) = 0;
//	//cameraMatrix.at<double>(1, 1) = 1152.841197789385;
//	//cameraMatrix.at<double>(1, 2) = 502.5554719939521;
//	//cameraMatrix.at<double>(2, 0) = 0;
//	//cameraMatrix.at<double>(2, 1) = 0;
//	//cameraMatrix.at<double>(2, 2) = 1;
//	//distCoeffs.at<double>(0, 0) = -0.2132339464479368;
//	//distCoeffs.at<double>(1, 0) = 0.3584186449258908;
//	//distCoeffs.at<double>(2, 0) = -0.00121915745714334;
//	//distCoeffs.at<double>(3, 0) = -0.001944591250376644;
//	//distCoeffs.at<double>(4, 0) = -0.07942466741526423;
//	//CV_MAT_ELEM(cameraMatrix, double, 0, 0) = 1152.195197178761;
//
//
//
//	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
//}
//
//// 如果向对话框添加最小化按钮，则需要下面的代码
////  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
////  这将由框架自动完成。
//
//void CLEDLocalizationDlg::OnPaint()
//{
//	if (IsIconic())
//	{
//		CPaintDC dc(this); // 用于绘制的设备上下文
//
//		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);
//
//		// 使图标在工作区矩形中居中
//		int cxIcon = GetSystemMetrics(SM_CXICON);
//		int cyIcon = GetSystemMetrics(SM_CYICON);
//		CRect rect;
//		GetClientRect(&rect);
//		int x = (rect.Width() - cxIcon + 1) / 2;
//		int y = (rect.Height() - cyIcon + 1) / 2;
//
//		// 绘制图标
//		dc.DrawIcon(x, y, m_hIcon);
//	}
//	else
//	{
//		CDialogEx::OnPaint();
//	}
//}
//
////当用户拖动最小化窗口时系统调用此函数取得光标
////显示。
//HCURSOR CLEDLocalizationDlg::OnQueryDragIcon()
//{
//	return static_cast<HCURSOR>(m_hIcon);
//}
//
////Initial video parameter
//void CLEDLocalizationDlg::InitImageParam()
//{
//	memset(&m_CapInfo, 0, sizeof(CapInfoStruct));
//	m_CapInfo.Buffer = m_pRawData;
//
//	m_CapInfo.Width = 1280;
//	m_CapInfo.Height = 1024;
//	m_CapInfo.HorizontalOffset = 0;
//	m_CapInfo.VerticalOffset = 0;
//	m_CapInfo.Exposure = 200;//2.6m最佳曝光是200，0.8m最佳曝光20，1.7m曝光70，2.3m曝光100，3m曝光300
//	m_CapInfo.Gain[0] = 10;
//	m_CapInfo.Gain[1] = 10;
//	m_CapInfo.Gain[2] = 10;
//	m_CapInfo.Control = 0;
//	memset(m_CapInfo.Reserved, 0, 8);
//	m_CapInfo.Reserved[0] = 2; /*Reserved[0] 设置显示方式
//		0  GDI显示方式1，效率较高，但缩放效果不如GDI显示方式1
//		1  GDI显示方式2，效率不高，但缩放效果好1
//		2  GDI显示方式3，效率高，但不支持缩放
//		3  DirectX显示方式，效率较高，缩放效果较好，需要操作系统已经安装了DirectX9.0或更好版本。
//*/
//
//}
//
//void CALLBACK RawCallBack(LPVOID lpParam, LPVOID lpUser)
//{
//	BYTE *pDataBuffer = (BYTE*)lpParam;
//	CLEDLocalizationDlg *pDemoDlg = (CLEDLocalizationDlg*)lpUser;
//
//	if (pDemoDlg->m_bRawSave)
//	{
//		errno_t err;
//		FILE * fp;
//		err = fopen_s(&fp,"RawData.raw", "wb+");
//		if (err == 0)
//			fwrite(pDataBuffer, sizeof(BYTE), pDemoDlg->m_CapInfo.Width*pDemoDlg->m_CapInfo.Height, fp);
//		fclose(fp);
//		pDemoDlg->m_bRawSave = FALSE;
//	}
//	if (pDemoDlg->m_bRawToRGB) 
//	{
//		std::string   str = "RawToRGBData.bmp";
//		LPCTSTR   lpstr = (LPCTSTR)str.c_str();//获取字符串首地址指针
//		MV_Usb2ConvertRawToRgb(pDemoDlg->m_hMVC1000, pDataBuffer, pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height, pDemoDlg->m_pRGBData);
//		//pDemoDlg->SaveRGBAsBmp(pDemoDlg->m_pRGBData, "RawToRGBData.bmp", pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height);
//		//是否可以取消bmp存储，影响运行速度？
//		//MV_Usb2SaveFrameAsBmp(pDemoDlg->m_hMVC1000, &pDemoDlg->m_CapInfo, pDemoDlg->m_pRGBData, lpstr);
//		
//		m_RGBData = pDemoDlg->m_pRGBData;
//		pDemoDlg->m_bRawToRGB = FALSE;
//	}
//}
//
//
//void CLEDLocalizationDlg::OnStart()
//{
//	//相机编号
//	int nIndex = 0;
//	//对设备初始化，查找并打开设备，并返回设备句柄
//	int rt = MV_Usb2Init("MVC-F", &nIndex, &m_CapInfo, &m_hMVC1000);
//	if (ResSuccess != rt)
//	{
//		MessageBox("Can not open USB camera!","error", MB_ICONERROR | MB_SYSTEMMODAL | MB_SETFOREGROUND);
//		MV_Usb2Uninit(&m_hMVC1000);
//		m_hMVC1000 = NULL;
//		return;
//	}
//	MV_Usb2SetOpMode(m_hMVC1000, m_nOpMode, FALSE);//设置模式
//	MV_Usb2SetRawCallBack(m_hMVC1000, RawCallBack, this);
//	//start capture
//	MV_Usb2StartCapture(m_hMVC1000, TRUE);
//	//Save raw
//	m_bRawSave = TRUE;
//
//	m_listmsg.AddString("USB camera start capture");
//	//	DWORD nRet=MV_Usb2SetThreadAffinityMask(m_hMVC1000,2);
//	//选中listbox中的最后一行
//	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
//}
//
//void CLEDLocalizationDlg::OnImageXY()
//{
//	
//		int i;
//		m_bRawToRGB = TRUE;
//		//需要sleep100ms以上，存储图片
//		Sleep(110);
//		//m_imageXY.ShowImage();
//		m_imageXY.BlobDetector();
//		Sleep(110);
//		for (i = 0; i < 15; i++)
//		{
//			m_test.Format("%f;%f", detectKeyPoint[i].pt.x, detectKeyPoint[i].pt.y);
//			m_listmsg.AddString(m_test);
//		}
//		//选中listbox中的最后一行
//		m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
//	
//}
//void CLEDLocalizationDlg::OnSolvepnp()
//{
//	
//	m_solvepnp.SloveEPNP11();
//	
//}
//void CLEDLocalizationDlg::OnStop()
//{
//	MV_Usb2StartCapture(m_hMVC1000, FALSE);
//	//uninit camera
//	if (m_hMVC1000 != NULL) {
//		MV_Usb2Uninit(&m_hMVC1000);
//		m_hMVC1000 = NULL;
//
//	}
//	//销毁窗口
//	//cvDestroyWindow("circles");
//	cvDestroyAllWindows();//销毁所有HighGUI窗口
//
//	m_listmsg.AddString("USB camera stop capture");
//	//选中listbox中的最后一行
//	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
//}
//
//void CLEDLocalizationDlg::OnCalibrate()
//{
//	m_calibrate.Calibrate();
//}
//void CLEDLocalizationDlg::OnTest()
//{
//	//m_solvepnp.SloveEPNP_asymmetric6();
//
//}


