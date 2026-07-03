# RPM: synthetic aperture and probability mapping for mmWave radar

## Prepare environment using [uv](https://docs.astral.sh/uv/getting-started/installation/)

```
uv sync
```

## Download Dataset

Download traces from [google drive](https://drive.google.com/drive/folders/16VlVu-fHJ3t3CoSsfjtCBawzcBG7iJYH?usp=drive_link) into `data/rpm` using [gdown](https://github.com/wkentaro/gdown). The dataset has 14 traces totaling ~1.1TB, so make sure you have enough disk space:

```
uv run gdown --folder https://drive.google.com/drive/folders/16VlVu-fHJ3t3CoSsfjtCBawzcBG7iJYH -O data/
```

To grab a single trace, open the folder in a browser and download that subfolder individually.

## Run mapping

```
uv run map_sar.py --folder data/rpm/north_short --save-video --name test
```

You can find the result folder under exps/map_sar

<img src="example.gif" alt="example" width="500">
