
// LEDLocalizationDlg.h: 头文件
//

#pragma once


// CLEDLocalizationDlg 对话框
class CLEDLocalizationDlg : public CDialogEx
{
// 构造
public:
	CLEDLocalizationDlg(CWnd* pParent = nullptr);	// 标准构造函数
	void InitImageParam();//初始化相机参数

	HANDLE m_hMVC1000;
	struct CapInfoStruct m_CapInfo;		//视频属性
	int		m_nOpMode;
	BOOL	m_bRawSave;
	BOOL	m_bRawToRGB;
	BYTE   *m_pRGBData;					//24bitRGB数据指针
	BYTE   *m_pRawData;					//用于存放RawData数据



// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_LEDLOCALIZATION_DIALOG };
#endif
	CString m_test;
	CListBox m_listmsg;
	int      num_CurSel;




	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnStart();
	afx_msg void OnImageXY();
	afx_msg void OnSolvepnp();
	afx_msg void OnStop();
	afx_msg void OnCalibrate();
	afx_msg void OnTest();
};
extern CWnd* g_pWnd;