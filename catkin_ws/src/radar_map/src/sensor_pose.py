#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import rospy

from sensor_msgs.msg import PointCloud2
import tf

class SensorPose:
    def __init__(self):
        self.sub = rospy.Subscriber('cloud_in', PointCloud2, self.callback)
        pose_file = rospy.get_param('~pose_file', '/path/to/default/pose/file')

        gt = pd.read_csv(pose_file)
        self.gt = gt[["stamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]


        self.br = tf.TransformBroadcaster()

    def callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        
        while self.gt.iloc[0]["stamp"] < stamp:
            pose = self.gt.iloc[0]
            self.br.sendTransform(
                (pose["tx"], pose["ty"], pose["tz"]),
                (pose["qx"], pose["qy"], pose["qz"], pose["qw"]),
                rospy.Time.from_sec(pose["stamp"]),
                msg.header.frame_id,
                "map"
            )
            self.pose = pose
            self.gt = self.gt.iloc[1:]

    def get_pose(self):
        return self.pose
    

if __name__ == '__main__':
    rospy.init_node('sensor_pose')
    sp = SensorPose()
    rospy.spin()