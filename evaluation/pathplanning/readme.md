## Todo lists
1. Implement basic sampling methods - done
2. Map Inflation
3. Other sampling methods
4. Speed up astar - start with jit
5. Multi-threaded - since presumably this will be slow if we have many start-end goal pairs
6. Tool to get oriented bounding boxes for start/goal sampling - Claude wrote this in one go - Done
7. Voronoi and other path planning


## To run path planning experiments:

```bash
python main.py --gt-file ../data/exps/lidar_cic.pkl --pred-file ../data/exps/map_sar/cic_mimo/prob.pkl
```