
// LEDLocalizationDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "LEDLocalization.h"
#include "LEDLocalizationDlg.h"
#include "afxdialogex.h"
#include <string>
#include <highgui.h>
#include "imageXY.h"
#include "calibrate.h"
#include "solvepnp.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif
CvImageXY m_imageXY;
CWnd* g_pWnd;
CvCalibrate m_calibrate;
CvSlovePNP m_solvepnp;

// CLEDLocalizationDlg 对话框
CLEDLocalizationDlg::CLEDLocalizationDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_LEDLOCALIZATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CLEDLocalizationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LISTMSG, m_listmsg);
}

BEGIN_MESSAGE_MAP(CLEDLocalizationDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()

	ON_BN_CLICKED(IDC_START,OnStart)
	ON_BN_CLICKED(IDC_IMAGEXY, OnImageXY)
	ON_BN_CLICKED(IDC_SOLVEPNP, OnSolvepnp)
	ON_BN_CLICKED(IDC_STOP, OnStop)
	ON_BN_CLICKED(IDC_CALIBRATE, OnCalibrate)
	ON_BN_CLICKED(IDC_TEST, OnTest)
	

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
	DWORD RGBDataSize = 1280 * 1024 * 3;
	m_pRGBData = (BYTE*)malloc(RGBDataSize * sizeof(BYTE));
	m_RGBData = (BYTE*)malloc(RGBDataSize * sizeof(BYTE));//全局变量
	memset(m_pRGBData, 0, RGBDataSize);
	memset(m_RGBData, 0, RGBDataSize);//全局变量
	DWORD RawDataSize = 1280 * 1024;
	m_pRawData = (BYTE*)malloc(RawDataSize * sizeof(BYTE));
	memset(m_pRawData, 0, RawDataSize);
	m_bRawSave = FALSE;
	m_bRawToRGB = FALSE;
	m_hMVC1000 = NULL;
	InitImageParam();
	m_nOpMode = 0;//采集模式，0为连续模式；1为外触发模式
	//手动初始化相机内外参数，以后改自动,暂时失败
	//cameraMatrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 摄像机内参数矩阵 */
	//distCoeffs = cv::Mat(5, 1, CV_32FC1, cv::Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	//cameraMatrix.at<double>(0, 0) = 1152.195197178761;
	//cameraMatrix.at<double>(0, 1) = 0;
	//cameraMatrix.at<double>(0, 2) = 611.574705796087;
	//cameraMatrix.at<double>(1, 0) = 0;
	//cameraMatrix.at<double>(1, 1) = 1152.841197789385;
	//cameraMatrix.at<double>(1, 2) = 502.5554719939521;
	//cameraMatrix.at<double>(2, 0) = 0;
	//cameraMatrix.at<double>(2, 1) = 0;
	//cameraMatrix.at<double>(2, 2) = 1;
	//distCoeffs.at<double>(0, 0) = -0.2132339464479368;
	//distCoeffs.at<double>(1, 0) = 0.3584186449258908;
	//distCoeffs.at<double>(2, 0) = -0.00121915745714334;
	//distCoeffs.at<double>(3, 0) = -0.001944591250376644;
	//distCoeffs.at<double>(4, 0) = -0.07942466741526423;
	//CV_MAT_ELEM(cameraMatrix, double, 0, 0) = 1152.195197178761;



	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
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

//Initial video parameter
void CLEDLocalizationDlg::InitImageParam()
{
	memset(&m_CapInfo, 0, sizeof(CapInfoStruct));
	m_CapInfo.Buffer = m_pRawData;

	m_CapInfo.Width = 1280;
	m_CapInfo.Height = 1024;
	m_CapInfo.HorizontalOffset = 0;
	m_CapInfo.VerticalOffset = 0;
	m_CapInfo.Exposure = 200;//2.6m最佳曝光是200，0.8m最佳曝光20，1.7m曝光70，2.3m曝光100，3m曝光300
	m_CapInfo.Gain[0] = 10;
	m_CapInfo.Gain[1] = 10;
	m_CapInfo.Gain[2] = 10;
	m_CapInfo.Control = 0;
	memset(m_CapInfo.Reserved, 0, 8);
	m_CapInfo.Reserved[0] = 2; /*Reserved[0] 设置显示方式
		0  GDI显示方式1，效率较高，但缩放效果不如GDI显示方式1
		1  GDI显示方式2，效率不高，但缩放效果好1
		2  GDI显示方式3，效率高，但不支持缩放
		3  DirectX显示方式，效率较高，缩放效果较好，需要操作系统已经安装了DirectX9.0或更好版本。
*/

}

void CALLBACK RawCallBack(LPVOID lpParam, LPVOID lpUser)
{
	BYTE *pDataBuffer = (BYTE*)lpParam;
	CLEDLocalizationDlg *pDemoDlg = (CLEDLocalizationDlg*)lpUser;

	if (pDemoDlg->m_bRawSave)
	{
		errno_t err;
		FILE * fp;
		err = fopen_s(&fp,"RawData.raw", "wb+");
		if (err == 0)
			fwrite(pDataBuffer, sizeof(BYTE), pDemoDlg->m_CapInfo.Width*pDemoDlg->m_CapInfo.Height, fp);
		fclose(fp);
		pDemoDlg->m_bRawSave = FALSE;
	}
	if (pDemoDlg->m_bRawToRGB) 
	{
		std::string   str = "RawToRGBData.bmp";
		LPCTSTR   lpstr = (LPCTSTR)str.c_str();//获取字符串首地址指针
		MV_Usb2ConvertRawToRgb(pDemoDlg->m_hMVC1000, pDataBuffer, pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height, pDemoDlg->m_pRGBData);
		//pDemoDlg->SaveRGBAsBmp(pDemoDlg->m_pRGBData, "RawToRGBData.bmp", pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height);
		//是否可以取消bmp存储，影响运行速度？
		//MV_Usb2SaveFrameAsBmp(pDemoDlg->m_hMVC1000, &pDemoDlg->m_CapInfo, pDemoDlg->m_pRGBData, lpstr);
		
		m_RGBData = pDemoDlg->m_pRGBData;
		pDemoDlg->m_bRawToRGB = FALSE;
	}
}


void CLEDLocalizationDlg::OnStart()
{
	//相机编号
	int nIndex = 0;
	//对设备初始化，查找并打开设备，并返回设备句柄
	int rt = MV_Usb2Init("MVC-F", &nIndex, &m_CapInfo, &m_hMVC1000);
	if (ResSuccess != rt)
	{
		MessageBox("Can not open USB camera!","error", MB_ICONERROR | MB_SYSTEMMODAL | MB_SETFOREGROUND);
		MV_Usb2Uninit(&m_hMVC1000);
		m_hMVC1000 = NULL;
		return;
	}
	MV_Usb2SetOpMode(m_hMVC1000, m_nOpMode, FALSE);//设置模式
	MV_Usb2SetRawCallBack(m_hMVC1000, RawCallBack, this);
	//start capture
	MV_Usb2StartCapture(m_hMVC1000, TRUE);
	//Save raw
	m_bRawSave = TRUE;

	m_listmsg.AddString("USB camera start capture");
	//	DWORD nRet=MV_Usb2SetThreadAffinityMask(m_hMVC1000,2);
	//选中listbox中的最后一行
	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
}

void CLEDLocalizationDlg::OnImageXY()
{
	
		int i;
		m_bRawToRGB = TRUE;
		//需要sleep100ms以上，存储图片
		Sleep(110);
		//m_imageXY.ShowImage();
		m_imageXY.BlobDetector();
		Sleep(110);
		for (i = 0; i < 15; i++)
		{
			m_test.Format("%f;%f", detectKeyPoint[i].pt.x, detectKeyPoint[i].pt.y);
			m_listmsg.AddString(m_test);
		}
		//选中listbox中的最后一行
		m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
	
}
void CLEDLocalizationDlg::OnSolvepnp()
{
	
	m_solvepnp.SloveEPNP10();
	
}
void CLEDLocalizationDlg::OnStop()
{
	MV_Usb2StartCapture(m_hMVC1000, FALSE);
	//uninit camera
	if (m_hMVC1000 != NULL) {
		MV_Usb2Uninit(&m_hMVC1000);
		m_hMVC1000 = NULL;

	}
	//销毁窗口
	//cvDestroyWindow("circles");
	cvDestroyAllWindows();//销毁所有HighGUI窗口

	m_listmsg.AddString("USB camera stop capture");
	//选中listbox中的最后一行
	m_listmsg.SetCurSel(m_listmsg.GetCount() - 1);
}

void CLEDLocalizationDlg::OnCalibrate()
{
	m_calibrate.Calibrate();
}
void CLEDLocalizationDlg::OnTest()
{
	m_solvepnp.SloveEPNP_asymmetric6();

}