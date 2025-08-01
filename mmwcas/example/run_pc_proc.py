import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from glob import glob
from typing import cast
from os.path import join as pjoin

from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.interfaces import ConnectionExtRosbag1
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
    sensor_msgs__msg__PointField as PointField,
    std_msgs__msg__Header as Header,
)

from mmwcas.dataset import CascadeADCDataset
from mmwcas.process import RadarPCProcCU

argparser = argparse.ArgumentParser(description="process radar pointcloud")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
args = argparser.parse_args()

data_dir = args.folder
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The specified folder does not exist: {data_dir}")

radar_all = []
dataset, pc_proc = {}, {}
for r in ["radar0", "radar1"]:
    if pjoin(data_dir, r) in glob(pjoin(data_dir, "*")):
        dataset.update({r: CascadeADCDataset(pjoin(data_dir, r))})
        pc_proc.update({r: RadarPCProcCU(dataset[r].param)})
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
point_step = 20
typestore = get_typestore(Stores.ROS1_NOETIC)

os.makedirs(pjoin(data_dir, "bags_w_radar"), exist_ok=True)
bags = sorted(glob(pjoin(data_dir, "bags/*.bag")))
output_bag_path = [bfile.replace("bags", "bags_w_radar") for bfile in bags]
for bfile in output_bag_path:
    if os.path.exists(bfile):
        os.remove(bfile)

with AnyReader([Path(p) for p in bags]) as reader:
    start_time, end_time = reader.start_time, reader.end_time

pbar = tqdm(total=np.floor((end_time - start_time) / 1e6) / 1e3)
for b_in, b_out in zip(bags, output_bag_path):
    with Writer(b_out) as writer, AnyReader([Path(b_in)]) as reader:
        conn_map = {}
        for conn in reader.connections:
            ext = cast(ConnectionExtRosbag1, conn.ext)
            conn_map[conn.topic] = writer.add_connection(
                conn.topic,
                conn.msgtype,
                typestore=typestore,
                callerid=ext.callerid,
                latching=ext.latching,
            )
        for radar in pc_proc.keys():
            conn_map[radar] = writer.add_connection(
                f"/{radar}/points", PointCloud2.__msgtype__, typestore=typestore
            )

        for conn, timestamp, rawdata in reader.messages():

            while len(radar_all) > 0 and radar_all[0][0] < timestamp / 1e9:
                stamp, radar, idx = radar_all.pop(0)
                signal_cube = dataset[radar][idx]
                pcd = pc_proc[radar](signal_cube)
                if pcd is not None:
                    pcd = pcd.astype(np.float32)
                    pc_byte = pcd.reshape(-1).view(np.uint8)

                    header = Header(
                        seq=idx,
                        stamp=Time(sec=int(stamp), nanosec=int(stamp * 1e9) % 10**9),
                        frame_id=radar,
                    )
                    msg = PointCloud2(
                        header=header,
                        height=1,
                        width=pcd.shape[0],
                        fields=fields,
                        is_bigendian=False,
                        point_step=point_step,
                        row_step=point_step * pcd.shape[0],
                        data=pc_byte,
                        is_dense=True,
                    )

                    writer.write(
                        conn_map[radar],
                        int(stamp * 1e9),
                        typestore.serialize_ros1(msg, msg.__msgtype__),
                    )

                    pbar.set_description(f"num pcd:{pcd.shape[0]:4d}")
                pbar.n = np.floor((max(0, timestamp - start_time)) / 1e6) / 1e3
                pbar.refresh()

            writer.write(conn_map[conn.topic], timestamp, rawdata)
