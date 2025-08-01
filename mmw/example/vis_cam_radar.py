import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import argparse
import imageio

from rosbags.highlevel import AnyReader

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as pjoin

import jax
import jax.numpy as jnp

from mmwcas.dataset import CascadeADCDataset, RadarParam
from mmwcas.process import (
    RangeProc,
    RangeDopplerCFARProc,
    RangeAzimuthProc,
    RangeAzimuthCFARProc,
)

import warnings
warnings.filterwarnings("ignore")


argparser = argparse.ArgumentParser(description="Radar data visualization")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
argparser.add_argument(
    "--method", type=str, default="range_azimuth_polar", help="method to process siganl"
)
args = argparser.parse_args()

data_dir = args.folder
colormap = plt.get_cmap("hot")
radars, idx, cache, process = {}, {}, {}, {}
os.makedirs("videos", exist_ok=True)
data_name = Path(args.folder).name
writer = imageio.get_writer(f"videos/{data_name}_{args.method}.mp4", fps=20)

for r in ["radar0", "radar1"]:
    if pjoin(data_dir, r) in glob(pjoin(data_dir, "*")):
        radars.update({r: CascadeADCDataset(pjoin(data_dir, r))})
        idx.update({r: 0})

        if args.method == "range_img":
            process.update({r: jax.jit(RangeProc(radars[r].param))})
        elif args.method == "range_doppler":
            process.update({r: jax.jit(RangeDopplerCFARProc(radars[r].param))})
        elif args.method == "range_azimuth" or args.method == "range_azimuth_polar":
            process.update({r: jax.jit(RangeAzimuthProc(radars[r].param))})
        elif (
            args.method == "range_azimuth" or args.method == "range_azimuth_polar_cfar"
        ):
            process.update({r: jax.jit(RangeAzimuthCFARProc(radars[r].param))})


def to_img(radar_img, title, size=(256, 1024), color=(255, 255, 255)):
    radar_img = (np.flipud(radar_img) * 255).astype(np.uint8)
    radar_img = cv2.resize(radar_img, size, interpolation=cv2.INTER_NEAREST)
    radar_img = cv2.putText(
        radar_img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    return radar_img


def range_img(signal_cube, sensor, title):
    range_img = process[sensor](signal_cube)
    radar_img = colormap(np.asarray(range_img))[:, :, :3]
    return to_img(radar_img, title)


def range_doppler(signal_cube, sensor, title):
    obj_mask, rd_image = process[sensor](signal_cube)
    radar_img = colormap(np.asarray(rd_image))[:, :, :3]
    # radar_img[np.asarray(obj_mask)] = [0, 0.99, 0]
    return to_img(radar_img, title)


def range_azimuth(signal_cube, sensor, title):
    radar_img = process[sensor](signal_cube)
    radar_img = colormap(np.asarray(radar_img))[:, :, :3]
    return to_img(np.fliplr(radar_img), title, size=(1024, 1024))


def range_azimuth_polar(signal_cube, sensor, title):
    radar_img = process[sensor](signal_cube)

    fig = plt.figure(figsize=(10, 5))
    canvas = fig.canvas
    ax = fig.gca()
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()

    ax.pcolormesh(
        process[sensor].y_axis.T,
        process[sensor].x_axis.T,
        radar_img.T,
        cmap=colormap,
        shading="nearest",
    )
    canvas.draw()
    radar_img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8") #type: ignore
    radar_img = radar_img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    radar_img = cv2.resize(radar_img, (2048, 1024), interpolation=cv2.INTER_NEAREST)
    radar_img = cv2.putText(
        radar_img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    return radar_img


def range_azimuth_polar_cfar(signal_cube, sensor, title):
    obj_mask, radar_img = process[sensor](signal_cube)
    radar_img = colormap(np.asarray(radar_img))[:, :, :3]
    radar_img[np.asarray(obj_mask)] = [0, 0.99, 0]

    fig = plt.figure(figsize=(10, 5))
    canvas = fig.canvas
    ax = fig.gca()
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()

    ax.pcolormesh(
        process[sensor].y_axis.T,
        process[sensor].x_axis.T,
        np.moveaxis(radar_img, (0, 1), (1, 0)),
        shading="nearest",
    )
    canvas.draw()
    radar_img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8") #type: ignore
    radar_img = radar_img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    radar_img = cv2.resize(radar_img, (2048, 1024), interpolation=cv2.INTER_NEAREST)
    radar_img = cv2.putText(
        radar_img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    return radar_img


stamps = [v.stamps for v in radars.values()][0]
start_time, end_time = stamps[0], stamps[-1]
pbar = tqdm(total=np.floor((end_time - start_time) * 1e3) / 1e3)
pbar.set_postfix_str("Processing")

bags = sorted(glob(pjoin(data_dir, "bags/*.bag")))
topic = "/camera/left/image_color"
with AnyReader([Path(p) for p in bags]) as reader:
    connections = [x for x in reader.connections if x.topic == topic]
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        timestamp = timestamp / 1e9
        if timestamp > end_time:
            break
        pbar.n = np.floor((timestamp - start_time) * 1e3) / 1e3
        pbar.refresh()

        msg = reader.deserialize(rawdata, connection.msgtype)
        image = np.asarray(msg.data, dtype=np.uint8) #type: ignore
        image = image.reshape((msg.height, msg.width, 3)) #type: ignore
        image = np.flip(image, (2))[:, 4:-4, :]

        for r in radars.keys():
            stamp_r = radars[r].stamps
            while idx[r] < len(radars[r]) and stamp_r[idx[r]] < timestamp:
                cache[r] = None
                idx[r] += 1
            if idx[r] >= len(radars[r]):
                exit(0)

            if cache[r] is None:
                signal_cube = radars[r][idx[r]]
                cache[r] = locals()[args.method](signal_cube, r, f"{r}_{args.method}")

            if r == "radar1":
                image = np.concatenate([image, cache[r]], axis=1)
            elif r == "radar0":
                image = np.concatenate([cache[r], image], axis=1)

        writer.append_data(image)