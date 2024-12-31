import time
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from os.path import join as pjoin

from mmwcas.dataset import PointDataset
from evaluation import map_pc, chamfer_distance
from utils import bodyframe_vel_estimate
from sar import RadarMap

argparser = argparse.ArgumentParser(description="map radar pc")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
argparser.add_argument("--radar_map", type=str, help="load radar map", required=True)
argparser.add_argument(
    "--dist_thresh", type=float, default=10.0, help="Distance threshold in meters"
)
argparser.add_argument(
    "--z_thresh", type=float, default=0.2, help="Height (z) threshold in meters"
)
argparser.add_argument(
    "--snr_thresh", type=float, default=80, help="Signal-to-noise ratio threshold"
)
argparser.add_argument(
    "--doppler_thresh",
    type=float,
    default=0.5,
    help="Doppler velocity threshold in m/s",
)

args = argparser.parse_args()

dataset = PointDataset(args.folder + "/radar0")
pc_all = []

for pc, pose in tqdm(dataset):
    v, pc = bodyframe_vel_estimate(
        pc, vs_init=np.array([0, 1, 0]), threshold=args.doppler_thresh
    )
    snr = pc[:, 3]
    pc = pc[:, :3]
    z = pc[:, 2]
    d = np.linalg.norm(pc, axis=1)
    mask = np.logical_and(snr > args.snr_thresh, d < args.dist_thresh),
        
    pc = pc[mask]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[:, :3]))
    pcd.transform(pose)
    pc_all.append(np.asarray(pcd.points))
pc_all = np.concatenate(pc_all, axis=0)
pc_all[:, 2] = 0

map_sar = RadarMap.load(args.radar_map)
radar_pc_2D = map_pc(map_sar.grid, map_sar.resolution, pc_all, args.radar_map)

lidar_pcd = o3d.io.read_point_cloud(pjoin(args.folder, "config/map.pcd"))
lidar_pc = np.asarray(lidar_pcd.points)
lidar_2D = map_pc(
    map_sar.grid, map_sar.resolution, lidar_pc, args.radar_map, "map_lidar.png"
)

cd = chamfer_distance(lidar_2D, radar_pc_2D)
print(f"Chamfer distance: {cd}")

