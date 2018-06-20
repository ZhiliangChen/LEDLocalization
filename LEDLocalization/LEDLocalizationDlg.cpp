
// LEDLocalizationDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "LEDLocalization.h"
#include "LEDLocalizationDlg.h"
#include "afxdialogex.h"
#include <string>
#include <highgui.h>
#include "imageXY.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
CvImageXY m_imageXY;


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
	m_nOpMode = 0;//采集模式，0为连续模式；1为外触发模式

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
	if (pDemoDlg->m_bRawToRGB) {

		std::string   str = "RawToRGBData.bmp";
		LPCTSTR   lpstr = (LPCTSTR)str.c_str();//获取字符串首地址指针
		MV_Usb2ConvertRawToRgb(pDemoDlg->m_hMVC1000, pDataBuffer, pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height, pDemoDlg->m_pRGBData);
		//pDemoDlg->SaveRGBAsBmp(pDemoDlg->m_pRGBData, "RawToRGBData.bmp", pDemoDlg->m_CapInfo.Width, pDemoDlg->m_CapInfo.Height);
		//是否可以取消bmp存储，影响运行速度？
		MV_Usb2SaveFrameAsBmp(pDemoDlg->m_hMVC1000, &pDemoDlg->m_CapInfo, pDemoDlg->m_pRGBData, lpstr);
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
	num_CurSel = m_listmsg.GetCount();
	m_listmsg.SetCurSel(num_CurSel - 1);
}

void CLEDLocalizationDlg::OnImageXY()
{
	m_bRawToRGB = TRUE;
	m_imageXY.ShowImage();
}
void CLEDLocalizationDlg::OnSolvepnp()
{

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
	num_CurSel = m_listmsg.GetCount();
	m_listmsg.SetCurSel(num_CurSel - 1);
}
