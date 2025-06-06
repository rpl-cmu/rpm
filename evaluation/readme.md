To run experiments:

```
python experiment_dispatcher.py --lidar_map_path data/collected_data/map_lidar_sanitized --radar_map_path data/collected_data/map_sar data/collected_data/map_ra data/collected_data/map_cfar
```
and this also writes outputs to a local `output.pkl`.

To print this in latex table formats, rerun `experiment_dispatcher.py` with `--existing_output`.
