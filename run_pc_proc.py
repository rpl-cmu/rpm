import os
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob
from os.path import join as pjoin

import rospy
import rosbag
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

import jax
import jax.numpy as jnp

from mmwcas.dataset import CascadeADCDataset, RadarParam
from mmwcas.mmwcas import RadarPCProc

argparser = argparse.ArgumentParser(description="process radar pointcloud")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
args = argparser.parse_args()


data_dir = args.folder
radar_all = []
dataset, pc_proc = {}, {}
for r in ["radar0", "radar1"]:
    if pjoin(data_dir, r) in glob(pjoin(data_dir, "*")):
        dataset.update({r: CascadeADCDataset(pjoin(data_dir, r))})
        pc_proc.update({r: RadarPCProc(dataset[r].param)})
        for i in range(len(dataset[r])):
            radar_all.append([dataset[r].stamps[i], r, i])
radar_all = sorted(radar_all, key=lambda x: x[0])

fields = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("snr", 12, PointField.FLOAT32, 1),
    PointField("doppler", 16, PointField.FLOAT32, 1),
]


os.makedirs(pjoin(data_dir, "bags_pcd"), exist_ok=True)
bags = sorted(glob(pjoin(data_dir, "bags/*.bag")))
start_time = rosbag.Bag(bags[0]).get_start_time()
end_time = rosbag.Bag(bags[-1]).get_end_time()
pbar = tqdm(total=np.floor((end_time - start_time) * 1e3) / 1e3)
for bfile in bags:
    bag = rosbag.Bag(bfile)
    output_bag = rosbag.Bag(bfile.replace("bags", "bags_pcd"), "w")
    for topic, msg, t in bag.read_messages():
        while len(radar_all) > 0 and radar_all[0][0] < msg.header.stamp.to_sec():
            stamp, radar, idx = radar_all.pop(0)

            signal_cube = dataset[radar][idx]
            pcd = pc_proc[radar](signal_cube)
            
            header = Header()
            header.frame_id = radar
            header.stamp = rospy.Time.from_sec(stamp)
            ros_point = point_cloud2.create_cloud(header, fields, pcd.tolist())
            output_bag.write(f"/{radar}/points", ros_point, ros_point.header.stamp)

            pbar.set_description(f"num pcd: {pcd.shape[0]:5d}")
            pbar.n = np.floor((max(0, stamp - start_time)) * 1e3) / 1e3
            pbar.refresh()

        output_bag.write(topic, msg, msg.header.stamp)
    bag.close()
    output_bag.close()
