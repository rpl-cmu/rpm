#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import least_squares

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import tf

class SensorPose:
    def __init__(self):
        self.cloud_topic = rospy.get_param('~cloud_topic', '/radar_points')
        pose_file = rospy.get_param('~pose_file', '/path/to/default/pose/file')
        self.save_file = rospy.get_param('~save_file', '/path/to/default/save/file.pkl')
        self.doppler_thres = rospy.get_param('~doppler_thres', 0.1)

        self.sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.callback)

        gt = pd.read_csv(pose_file)
        self.gt = gt[["stamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]
        self.br = tf.TransformBroadcaster()

        self.map_cache = None
        self.sub_map = rospy.Subscriber('projected_map', OccupancyGrid, self.map_callback)
        self.pub_pc = rospy.Publisher('cloud_out', PointCloud2, queue_size=1)
        

    def map_callback(self, msg):
        self.map_cache = msg

    def bodyframe_vel_estimate(self,
        radar_points: np.ndarray,
        vs_init: np.ndarray = np.array([1, 0, 0]),
        threshold=0.5,
    ):
        doppler_v = radar_points[:, -1]
        norm_r = np.linalg.norm(radar_points[:, :3], axis=1)
        unit_r = -radar_points[:, :3] / norm_r[:, None]

        # cauchy loss doppler residual
        lstsq_func = lambda x, A, b: A @ x - b
        res_log = least_squares(
            lstsq_func, vs_init, loss="cauchy", f_scale=1, args=(unit_r, doppler_v)
        )
        vs = res_log.x

        res = unit_r @ vs - doppler_v
        mask = np.abs(res) < threshold

        return vs, radar_points[mask]


    def callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        
        if 'radar' in self.cloud_topic and self.doppler_thres > 0:
            pc = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 5)
            v, pc = self.bodyframe_vel_estimate(
                pc, vs_init=np.array([0, 1, 0]), threshold=self.doppler_thres
            )             
            msg.data = pc.tobytes()
            msg.width = pc.shape[0]

        
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

        self.pub_pc.publish(msg)
    
    def on_shutdown(self):
        rospy.loginfo("save map")
        if self.map_cache is not None:
            time = self.map_cache.info.map_load_time.to_sec()
            data = self.map_cache.data
            resolution = self.map_cache.info.resolution
            w = self.map_cache.info.width
            h = self.map_cache.info.height
            t = self.map_cache.info.origin.position
            r = self.map_cache.info.origin.orientation
            map_data = {
                'time': time,
                'data': np.asarray(data).reshape((h, w)),
                'resolution': resolution,
                'width': w,
                'height': h,
                't':[t.x, t.y, t.z],
                'r':[r.x, r.y, r.z, r.w], 
            }
            # Save the map data to a file
            

            with open(self.save_file, 'wb') as f:
                pickle.dump(map_data, f)
            rospy.loginfo("Map saved successfully.")
        else:
            rospy.logwarn("No map data to save.")
    

if __name__ == '__main__':
    rospy.init_node('sensor_pose')
    sp = SensorPose()
    rospy.on_shutdown(sp.on_shutdown)
    rospy.spin()