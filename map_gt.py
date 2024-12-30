import time
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

import open3d as o3d
from sar import RadarMap
from evaluation import map_pc

argparser = argparse.ArgumentParser(description="map radar pc")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
argparser.add_argument("--radar_map", type=str, help="load radar map", required=True)
argparser.add_argument(
    "--dist_thresh", type=float, default=10.0, help="Distance threshold in meters"
)
argparser.add_argument(
    "--z_thresh", type=float, default=0.3, help="Height (z) threshold in meters"
)


args = argparser.parse_args()

pcd_path = pjoin(args.folder, "config/map.pcd")
pcd = o3d.io.read_point_cloud(pcd_path)
pts = np.asarray(pcd.points)
z = pts[:,-1]
pts = pts[np.abs(z)<args.z_thresh]

map_sar = RadarMap.load(args.radar_map)
map_pc(map_sar.grid, map_sar.resolution, pts, args.radar_map, 'map_lidar.png')

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.run()
 