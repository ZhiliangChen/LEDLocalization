// rostest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <stdio.h>
#include "ros.h"
#include <geometry_msgs/Pose2D.h>
#include <windows.h>
using std::string;

int _tmain (int argc, _TCHAR * argv[])
{
	// init
	printf("rostest:\n");
	ros::NodeHandle nh;
	char *ros_master = "192.168.31.200"; // EAI ROS IP
	printf ("Connecting to server at %s\n", ros_master);
	nh.initNode (ros_master);	
	geometry_msgs::Pose2D Pose2D_msg;
	ros::Publisher pose2d_pub("agvpose", &Pose2D_msg);
	nh.advertise (pose2d_pub);
	printf ("Sending Pose2D message\n");

	// loop
	while (1)
	{
		Pose2D_msg.x = 5.1;
		Pose2D_msg.y = 0;
		Pose2D_msg.theta = 0;
		pose2d_pub.publish(&Pose2D_msg);
		nh.spinOnce ();
		Sleep (100);
	}
	printf ("All done!\n");
	return 0;
}

