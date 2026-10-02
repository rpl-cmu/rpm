# RPM: synthetic aperture and probability mapping for mmWave radar

## Prepare environment using [uv](https://docs.astral.sh/uv/getting-started/installation/)

```
uv sync
```

## Download Dataset

The dataset is hosted on [google drive](https://drive.google.com/drive/folders/16VlVu-fHJ3t3CoSsfjtCBawzcBG7iJYH?usp=drive_link) and is downloaded into `data/rpm` using [gdown](https://github.com/wkentaro/gdown). Download one or more traces by name:

```
uv run download-dataset nsh_short z_shape1
```

Available traces: `c_corridor`, `garden`, `nsh`, `nsh_a`, `nsh_a1`, `nsh_b`, `nsh_short`, `parking`, `square1`, `square2`, `tepper`, `wean`, `z_shape1`, `z_shape2`.

To download all 14 traces (~1.1TB, so make sure you have enough disk space):

```
uv run download-dataset --all
```

Rerunning the same command skips files that are already downloaded and resumes interrupted ones. Use `--output` to download somewhere other than `data/rpm`.

## Run mapping

```
uv run map_sar.py --folder data/rpm/nsh_short --save-video --name test
```

You can find the result folder under exps/map_sar

<img src="example.gif" alt="example" width="500">
