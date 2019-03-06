#pragma once
class CvSlovePNP
{

public:
	void SloveEPNP_NEW15();
	void SloveEPNP_Z13();
	void SloveEPNP_L14();
	void SloveEPNP_LINE9();
	void SloveEPNP_LINE7();
	void SloveEPNP_ANGLE7();
	void SloveEPNP_ANGLE9();
	void SloveEPNP_ANGLE15();

	void SloveIterative();
	void SloveP3P();
	void SloveEPNP7();
	void SloveEPNP8();
	void SloveEPNP9();
	void SloveEPNP10();
	void SloveEPNP11();
	void SloveEPNP12();
	void SloveEPNP13();
	void SloveEPNP14();
	void SloveEPNP15();
	void SloveEPNP_asymmetric4();
	void SloveEPNP_asymmetric5();
	void SloveEPNP_asymmetric6();
	void SloveEPNP_asymmetric7();
	void SloveEPNP_asymmetric8();
	void SloveEPNP_asymmetric9();
	void SloveEPNP_asymmetric10();
	void SloveEPNP_asymmetric11();

	void SloveEPNP_asymmetric6_3();
	void SloveEPNP_asymmetric6_2();
	void SloveEPNP_asymmetric6_1();
	void SloveEPNP_asymmetric6_0();
	void SloveEPNP_AGV15();
	void Test();
	double displacement;

	CString m_str;
};
extern double thetay_out_AGV, Cx_AGV, Cz_AGV, Cy_AGV;