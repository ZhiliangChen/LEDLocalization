
// LEDLocalizationDlg.h: 头文件
//

#pragma once
#include <Jai_Factory.h>
#include <cv.h>
#include <stdio.h>
#include "ros.h"
#include <geometry_msgs/Pose2D.h>

#define NODE_NAME_WIDTH         (int8_t*)"Width"
#define NODE_NAME_HEIGHT        (int8_t*)"Height"
#define NODE_NAME_PIXELFORMAT   (int8_t*)"PixelFormat"
#define NODE_NAME_GAIN          (int8_t*)"GainRaw"
#define NODE_NAME_ACQSTART      (int8_t*)"AcquisitionStart"
#define NODE_NAME_ACQSTOP       (int8_t*)"AcquisitionStop"
#define NODE_NAME_EXPOSURE      (int8_t*)"ExposureTimeRaw"

// CLEDLocalizationDlg 对话框
class CLEDLocalizationDlg : public CDialogEx
{
	// 构造
public:
	CLEDLocalizationDlg(CWnd* pParent = nullptr);	// 标准构造函数

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_LEDLOCALIZATION_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

// 实现
public:

	FACTORY_HANDLE  m_hFactory;     // Factory Handle
	CAM_HANDLE      m_hCam;         // Camera Handle
	THRD_HANDLE     m_hThread;      // Acquisition Thread Handle
	int8_t          m_sCameraId[J_CAMERA_ID_SIZE];    // Camera ID
	
	bool            m_bCameraOpen;
	bool            m_bAcquisitionRunning;
	CvSize          m_BoardSize;
	CvSize          m_ImgSize;
	CvPoint2D32f*   m_pImagePointsBuf;
	CvSeq*          m_pImagePointsSeq;

	int             m_ImageCount;
	float           m_SquareSize;
	float           m_AspectRatio;
	int             m_Flags;
	int             m_CaptureDelay;
	bool            m_bSaveImages;
	//bool            m_bImagesSaved;
	bool            m_bSaveSettings;
	//bool            m_bSettingsSaved;
	CRITICAL_SECTION m_CriticalSection;             // Critical section used for protecting the measured time value so it can be displayed
	CvMemStorage*   m_pStorage;
	IplImage*       m_pUndistortMapX;
	IplImage*       m_pUndistortMapY;

	BOOL OpenFactoryAndCamera();
	void CloseFactoryAndCamera();
	void StreamCBFunc(J_tIMAGE_INFO * pAqImageInfo);
	void InitializeControls();
	//void EnableControls();

// 对话框数据

	CString m_test;
	CListBox m_listmsg;
	int      num_CurSel;

//多线程
	static UINT Thread1(void *param);
	static UINT Thread2(void *param);
														
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:	
	afx_msg void OnDestroy();
	afx_msg void OnTimer(UINT_PTR nIDEvent);

	afx_msg void OnAGVCONNECT();
	afx_msg void OnStart();
	afx_msg void OnImageXY();
	afx_msg void OnSolvepnp();
	afx_msg void OnStop();
	afx_msg void OnCalibrate();
	afx_msg void OnTest();
};
extern CWnd* g_pWnd;
extern IplImage*  m_pImg;        // OpenCV Images
extern ros::NodeHandle nh;
extern geometry_msgs::Pose2D Pose2D_msg;
extern int Thread_flag;
//extern ros::Publisher pose2d_pub;




//=================================以下是采用MVC-1000F相机所用的头文件===========================
//// CLEDLocalizationDlg 对话框
//class CLEDLocalizationDlg : public CDialogEx
//{
//// 构造
//public:
//	CLEDLocalizationDlg(CWnd* pParent = nullptr);	// 标准构造函数
//	
//	void InitImageParam();//初始化相机参数
//
//	HANDLE m_hMVC1000;
//	struct CapInfoStruct m_CapInfo;		//视频属性
//	int		m_nOpMode;
//	BOOL	m_bRawSave;
//	BOOL	m_bRawToRGB;
//	BYTE   *m_pRGBData;					//24bitRGB数据指针
//	BYTE   *m_pRawData;					//用于存放RawData数据
//
//// 对话框数据
//#ifdef AFX_DESIGN_TIME
//	enum { IDD = IDD_LEDLOCALIZATION_DIALOG };
//#endif
//	CString m_test;
//	CListBox m_listmsg;
//	int      num_CurSel;
//
//	protected:
//	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持
//
//
//// 实现
//protected:
//	HICON m_hIcon;
//
//	// 生成的消息映射函数
//	virtual BOOL OnInitDialog();
//	afx_msg void OnPaint();
//	afx_msg HCURSOR OnQueryDragIcon();
//	DECLARE_MESSAGE_MAP()
//public:
//	afx_msg void OnStart();
//	afx_msg void OnImageXY();
//	afx_msg void OnSolvepnp();
//	afx_msg void OnStop();
//	afx_msg void OnCalibrate();
//	afx_msg void OnTest();
//};
//extern CWnd* g_pWnd;