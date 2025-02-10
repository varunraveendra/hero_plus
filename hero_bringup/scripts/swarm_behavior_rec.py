#!/usr/bin/env python3

import rospy
from hero_common.msg import predictions
import numpy as np

class SwarmBehavior:
    def __init__(self):
        rospy.init_node('swarm_behavior', anonymous=True)
        self.robot_names = rospy.get_param('/robot_names', [])
        self.subscribers = []
        self.publishers = rospy.Publisher('/prediction/swarm',predictions,queue_size=10)
        self.reading={}
        self.robot_nums=enumerate(self.robot_names)
        self.reading_mat=np.zeros((3,len(self.robot_names)))

        c=0

        for robot_name in self.robot_names:
            topic_name = f"/prediction/{robot_name}"
            self.reading[robot_name]=c
            self.subscribers.append(rospy.Subscriber(topic_name,predictions, self.callback, robot_name))
            c+=1

        rospy.spin()

    def callback(self, data, robot_name):
        self.reading_mat[0,self.reading[robot_name]]=data.x
        self.reading_mat[1,self.reading[robot_name]]=data.y
        self.reading_mat[2,self.reading[robot_name]]=data.z
        self.swarm_callback()

    def swarm_callback(self):
        pred=predictions()
        pred.x=0.0
        pred.y=0.0
        pred.z=0.0
        p=np.sum(self.reading_mat,axis=1)
        pred.x=p[0]
        pred.y=p[1]
        pred.z=p[2]

        self.publishers.publish(pred)



if __name__ == '__main__':
    try:
        SwarmBehavior()
    except rospy.ROSInterruptException:
        pass



