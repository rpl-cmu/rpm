## Todo lists
1. Implement basic sampling methods - done
2. Map Inflation - done
3. Other sampling methods - room is done
4. Speed up astar - start with jit - not doable
5. Multi-threaded - since presumably this will be slow if we have many start-end goal pairs - done
6. Tool to get oriented bounding boxes for start/goal sampling - Claude wrote this in one go - Done
7. Other path planning
    1. Voronoi Graph
    2. RRT
    3. Artificial potential field
8. Evaluation


## To run path planning experiments:

```bash
python main.py --gt-file ../data/exps/lidar_cic.pkl --pred-file ../data/exps/map_sar/cic_mimo/prob.pkl
```

## Evaluation Criteria
1. Between start and end point with known path - basically MapEx TU
    1. Rate at which it succeeded at finding the path
    2. Rate where path are also valid on lidar map - or Distribution of percentage of invalid paths.
2. Qualitative analysis: distribution of failed paths
    1. Corner cases - uncertain cases where it can be handled by local planner
    2. Fundamentally unfeasible paths - goes through a wall / glass etc
    3. Strong reflection from metal pillars blocking off path
