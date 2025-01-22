#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg  import Float32
from sensor_msgs.msg import LaserScan

class MultiRobotController:
    def __init__(self):
        rospy.init_node('multi_robot_controller',anonymous=True)
        self.robot_names=rospy.get_param('/robot_names',[])
        self.subscribers=[]
        self.publishers={}
        
        self.linear_true_max=rospy.get_param('linear_true_max',0.0)
        self.angular_true_max=rospy.get_param('angular_true_max',0.0)
        self.linear_false=rospy.get_param('linear_false',0.0)
        self.angular_false=rospy.get_param('angular_false',0.0)

        
        
        for robot_name in self.robot_names:
            topic_name=f"/{robot_name}/laser"
            self.subscribers.append(rospy.Subscriber(topic_name,LaserScan,self.callback,robot_name))
            self.publishers[robot_name]=rospy.Publisher(f"/{robot_name}/velocity_controller/cmd_vel",Twist,queue_size=10)
            
        rospy.spin()
    
    def callback(self,data,robot_name):
        if data.intensities[8]>rospy.get_param(f"intensities/{robot_name}",10.0):
            self.send_velocity_2(robot_name)
        else:
            self.send_velocity_3(robot_name)
    
    
    def send_velocity_2(self,robot_name):
        cmd=Twist()
        cmd.linear.x=self.linear_true_max
        cmd.angular.z=self.angular_true_max
        self.publishers[robot_name].publish(cmd)  
        
    def send_velocity_3(self,robot_name):
        cmd=Twist()
        cmd.linear.x=self.linear_false
        cmd.angular.z=self.angular_false
        self.publishers[robot_name].publish(cmd)
        
if __name__=='__main__':
    try:
        MultiRobotController()
    except rospy.ROSInterruptException:
        pass
        


