import numpy as np
import argparse
from os.path import join as pjoin
from sar import ChirpPoseDataset
import open3d as o3d
import pytransform3d.visualizer as pv


argparser = argparse.ArgumentParser(description="vis trajectory")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)

args = argparser.parse_args()
data_dir = args.folder
seq_name = data_dir.split("/")[-1]
r = "radar0"
rx, tx = 0, 2

dataset = ChirpPoseDataset(pjoin(data_dir, r), rx, tx)
print("len", len(dataset))

fig = pv.figure()
fig.plot_basis(R=np.eye(3), p=[0, 0, 0])

i_start, i_end = 0, 40
poses = dataset.chirp_poses[i_start:i_end, :, tx, :, :]

poses = poses.reshape(-1, 4, 4)
print("poses", poses.shape)

trajectory = pv.Trajectory(poses, s=0.5, n_frames=len(poses))
trajectory.add_artist(fig)

fig.view_init()
fig.show()
