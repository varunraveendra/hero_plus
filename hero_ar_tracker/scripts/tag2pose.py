#!/usr/bin/env python3

import rospy
import math

from apriltag_ros.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry

import tf2_ros
from geometry_msgs.msg import TransformStamped

from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_inverse,
)


def publish_static_transform():
    static_transform_stamped = TransformStamped()
    static_transform_stamped.header.stamp = rospy.Time.now()
    static_transform_stamped.header.frame_id = "world"
    static_transform_stamped.child_frame_id = "usb_cam"

    # Define the transformation
    static_transform_stamped.transform.translation.x = 0.0  # Adjust these values as needed
    static_transform_stamped.transform.translation.y = 0.0
    static_transform_stamped.transform.translation.z = 0.0
    static_transform_stamped.transform.rotation.x = 0.0
    static_transform_stamped.transform.rotation.y = 0.0
    static_transform_stamped.transform.rotation.z = 0.0
    static_transform_stamped.transform.rotation.w = 1.0

    static_broadcaster.sendTransform(static_transform_stamped)


def normalize(q):
    dx2 = q.x*q.x
    dy2 = q.y*q.y
    dz2 = q.z*q.z
    dw2 = q.w*q.w
    norm = math.sqrt(dw2 + dz2 + dy2 + dx2) #+ 10e-9
    if abs(norm) < 1e-8:
        norm = 1
        q[3] = 1
    q.x /= norm
    q.y /= norm
    q.z /= norm
    q.w /= norm
    return q 
    

def tag_callback(data):
    for detection in data.detections:
        tag_id = detection.id[0]
        tag_pose = detection.pose.pose.pose

        if tag_pose.position.z > 2.30 or tag_pose.position.z < 2.10:
            continue

        # Check if publisher for this tag ID already exists
        if tag_id not in publishers:
            topic_name = f"/hero_{tag_id}/odom2"
            publishers[tag_id] = rospy.Publisher(topic_name, Odometry, queue_size=1)

        # Populate and publish PoseStamped message
        pose_msg = Odometry()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        pose_msg.child_frame_id = f"hero_{tag_id}/base_link"
        # print(tag_pose)

        # Assuming detection.pose contains the tag's pose information
        pose_msg.pose.pose = tag_pose
        pose_msg.pose.pose.position.y = -pose_msg.pose.pose.position.y
        pose_msg.pose.pose.position.z = 0

        q_rot = quaternion_from_euler(math.pi, 0, 0)
        q_new = quaternion_multiply(
            q_rot,
            [
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
            ],
        )
        pose_msg.pose.pose.orientation.x = 0
        pose_msg.pose.pose.orientation.y = 0
        pose_msg.pose.pose.orientation.z = q_new[2]
        pose_msg.pose.pose.orientation.w = q_new[3]
        pose_msg.pose.pose.orientation = normalize(pose_msg.pose.pose.orientation)
        # pose_msg.pose.pose.orientation.x = -pose_msg.pose.pose.orientation.x
        

        publishers[tag_id].publish(pose_msg)
        publish_static_transform()

      


if __name__ == "__main__":
    rospy.init_node("tag2pose")

    # Initialize publishers dictionary
    publishers = {}

    # Initialize tf2_ros StaticTransformBroadcaster
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Publish the static transform once
    publish_static_transform()

    # Subscriber to listen for tag detections
    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, tag_callback)

    rospy.spin()
