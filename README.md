# probability mapping for mmWave radar

## Prepare environment

create conda env

```
conda env create --file conda.yaml
```

install mmwcas for dataloading and basic signal processing

```
pip install ./mmwcas --verbose
```

## Download Dataset

download any sequence from the [google drive]() (link will reveal authors)

## Run mapping

```
python map_sar.py --folder ~/data/sftp/nsh_short --save-video --name test
```

<img src="example.gif" alt="example" width="500">
