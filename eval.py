
import argparse
import numpy as np
import cv2
from os.path import join as pjoin

import open3d as o3d
from sar import RadarMap
from evaluation import map_pc, chamfer_distance
from utils import bodyframe_vel_estimate
from mmwcas.dataset import PointDataset

argparser = argparse.ArgumentParser(description="evaluate")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
argparser.add_argument("--radar_map", type=str, help="load radar map", required=True)
argparser.add_argument(
    "--dist_thresh", type=float, default=10.0, help="Distance threshold in meters"
)
argparser.add_argument(
    "--z_thresh", type=float, default=0.3, help="Height (z) threshold in meters"
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

lidar_pcd = o3d.io.read_point_cloud(pjoin(args.folder, "config/map.pcd"))
lidar_pc = np.asarray(lidar_pcd.points)
lidar_pc = lidar_pc[np.abs(lidar_pc[:, -1]) < args.z_thresh]

map_sar = RadarMap.load(args.radar_map)
lidar, lidar_img = map_pc(
    map_sar.grid, map_sar.resolution, lidar_pc, args.radar_map, "map_lidar.png"
)

map_abs = np.abs(map_sar.complex)
thresh = np.percentile(map_abs, 97)
grid = map_sar.grid
sar_pc = grid[map_abs > thresh]
sar_pc, sar_image = map_pc(grid, map_sar.resolution, sar_pc, args.radar_map, "map_sar.png")

cd = chamfer_distance(lidar[:,:2], sar_pc[:,:2])
print(f"SAR Chamfer distance: {cd}")


dataset = PointDataset(args.folder + "/radar0")
pc_all = []

for pc, pose in dataset:
    v, pc = bodyframe_vel_estimate(
        pc, vs_init=np.array([0, 1, 0]), threshold=args.doppler_thresh
    )
    snr = pc[:, 3]
    pc = pc[:, :3]
    z = pc[:, 2]
    d = np.linalg.norm(pc, axis=1)
    mask = np.logical_and(snr > args.snr_thresh, d < args.dist_thresh)
    mask = np.logical_and(mask, np.abs(z) < args.z_thresh)
        
    pc = pc[mask]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[:, :3]))
    pcd.transform(pose)
    pc_all.append(np.asarray(pcd.points))
pc_all = np.concatenate(pc_all, axis=0)

mimo_pc, mimo_image = map_pc(map_sar.grid, map_sar.resolution, pc_all, args.radar_map)
cd = chamfer_distance(lidar[:,:2], mimo_pc[:,:2])
print(f"PC Chamfer distance: {cd}")

map_overlap = np.stack([lidar_img[...,0], sar_image[...,0], mimo_image[...,0]], axis=2)
cv2.imwrite(f"{args.radar_map}/map_overlap.png", map_overlap)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(mimo_pc)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.run()
 