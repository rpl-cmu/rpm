# openMMWCAS

data loader and signal processing library for TI Cascade mmWave Radar

## Install

recommend using mamba(conda) to install all dependencies

```
conda env create --file conda.yaml
```

then install mmwcas package

```
pip install . --verbose
```

## Dataset

each sequence contain following files

```
Sequence
 ┣ 📂bags
 ┃ ┗ 📜bag_*.bag
 ┣ 📂config
 ┃ ┣ 📜T_radarN_epson.yaml
 ┃ ┣ 📜map.pcd
 ┃ ┣ 📜radarN.mmwave.json
 ┃ ┗ 📜radarN_calib.mat
 ┣ 📂pose
 ┃ ┣ 📜pose_imu_lio.txt
 ┃ ┗ 📜pose_radarN.pkl
 ┣ 📂radarN
 ┃ ┣ 📜*_data.bin
 ┃ ┗ 📜*_idx.bin
 ┗ 📂stamp
   ┗ 📜*.txt
```

## Run examples

visualize processed radar image with camera image \
FOLDER: path to your sequence folder \
METHOD: [range_img, range_doppler, range_azimuth, range_azimuth_polar]

```
python example/vis_cam_radar.py --folder FOLDER [--method METHOD]
```
