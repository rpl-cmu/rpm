import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math
from typing import Any


def rrt_path_planning(
    map_array: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cache: Any
):
    """
    RRT (Rapidly-exploring Random Tree) path planning algorithm

    Parameters:
    -----------
    map_array : np.ndarray
        Binary occupancy grid (0: free space, 1: obstacle)
    start : tuple
        Start position (y, x)
    goal : tuple
        Goal position (y, x)
    max_iterations : int
        Maximum number of iterations
    step_size : float
        Distance of each step
    goal_sample_rate : float
        Probability of sampling the goal point

    Returns:
    --------
    path : list
        List of positions (y, x) from start to goal, or None if no path found
    """
    max_iterations = 5000
    step_size = 10
    goal_sample_rate = 0.05

    start_tuple = (start[0], start[1])
    goal_tuple = (goal[0], goal[1])

    # Initialize tree with start node
    tree = {}  # Dictionary to store the tree {child: parent}
    tree[start_tuple] = None

    # Check if start or goal is in an obstacle
    if map_array[start[0], start[1]] == 1 or map_array[goal[0], goal[1]] == 1:
        return np.zeros((0, 2), dtype=np.int32)

    for i in range(max_iterations):
        # Sample a random point
        if random.random() < goal_sample_rate:
            random_point = goal
        else:
            # Random sampling within the map boundaries
            while True:
                random_y = random.randint(0, map_array.shape[0] - 1)
                random_x = random.randint(0, map_array.shape[1] - 1)
                random_point = (random_y, random_x)
                if (random_point not in tree):
                    break

        # Find the nearest node in the tree
        nearest_node = find_nearest_node(tree, random_point)

        if ((nearest_node[0] == random_point[0]) and (nearest_node[1] == random_point[1])):
            continue

        # Create a new node in the direction of random_point
        new_node = steer(nearest_node, random_point, step_size)

        if ((new_node[0] == nearest_node[0]) and (new_node[1] == nearest_node[1])):
            continue

        # Check if the path is collision-free
        if is_collision_free(map_array, nearest_node, new_node):
            # print("Collision free")
            new_node_tuple = (new_node[0], new_node[1])
            # Add the new node to the tree
            tree[new_node_tuple] = (nearest_node[0], nearest_node[1])

            # Check if we can connect to the goal
            if distance(new_node, goal) <= step_size:
                if is_collision_free(map_array, new_node, goal):
                    if (goal_tuple not in tree):
                        tree[goal_tuple] = new_node_tuple
                    # Path found, extract it

                    retPath = np.asarray(extract_path(tree, start_tuple, goal_tuple))
                    result = np.empty_like(retPath)
                    result[:, 0] = retPath[:, 1]
                    result[:, 1] = retPath[:, 0]
                    return result

    return np.zeros((0, 2), dtype=np.int32)


def find_nearest_node(tree, point):
    """Find the node in the tree closest to the given point"""
    return min(tree.keys(), key=lambda node: distance(node, point))


def distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def steer(from_node, to_node, step_size):
    """Create a new node in the direction of to_node from from_node with distance step_size"""
    dist = distance(from_node, to_node)

    if dist <= step_size:
        return to_node

    # Calculate the direction vector
    dy = to_node[0] - from_node[0]
    dx = to_node[1] - from_node[1]

    # Normalize and scale by step_size
    magnitude = math.sqrt(dx**2 + dy**2)
    dy = dy / magnitude * step_size
    dx = dx / magnitude * step_size

    # Create new node
    new_y = int(from_node[0] + dy)
    new_x = int(from_node[1] + dx)

    return (new_y, new_x)


def is_collision_free(map_array, node1, node2):
    """Check if the path between node1 and node2 is collision-free"""
    # Bresenham's line algorithm to check for collisions along the path
    y1, x1 = node1
    y2, x2 = node2
    
    # Bresenham's line algorithm to find all cells the line passes through
    cells = get_line_segment_cells(x1, y1, x2, y2)

    y_lim, x_lim = map_array.shape

    # Check if any cell satisfies the predicate
    for x, y in cells:
        # Check if coordinates are within map boundaries
        if 0 < y < y_lim and 0 < x < x_lim:
            if map_array[y, x] > 0.4:
                return False
    
    return True

def get_line_segment_cells(x1, y1, x2, y2):
    """
    Implements Bresenham's line algorithm to get all cells a line segment passes through
    
    Args:
        x1 (float): Starting x coordinate
        y1 (float): Starting y coordinate
        x2 (float): Ending x coordinate
        y2 (float): Ending y coordinate
        
    Returns:
        list: List of cell coordinates the line passes through
    """
    cells = []
    
    # Convert to integers (since we're dealing with grid cells)
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    max_num = np.ceil(np.sqrt(dx**2 + dy**2)).astype(np.int32)
    
    while True:
        cells.append((int(x1), int(y1)))
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            if x1 == x2:
                break
            err -= dy
            x1 += sx
        if e2 < dx:
            if y1 == y2:
                break
            err += dx
            y1 += sy
    
    return cells

def extract_path(tree, start, goal):
    """Extract the path from start to goal from the tree"""
    path = [goal]
    node = goal

    while node != start:
        if (node == tree[node]):
            print("Somehow")
            print(f"goal: {goal}, node: {node}")
            return np.zeros((0, 2))
        node = tree[node]
        path.append(node)

    return path[::-1]  # Reverse to get path from start to goal

