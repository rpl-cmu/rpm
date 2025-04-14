import numpy as np
import heapq
from enum import Enum
from typing import Callable


def aStar(map: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    # Check if start or goal positions are valid
    if (
        not (0 <= start[0] < map.shape[0] and 0 <= start[1] < map.shape[1])
        or not (0 <= goal[0] < map.shape[0] and 0 <= goal[1] < map.shape[1])
    ):
        return []
    
    directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], 
                           [-1, 1], [1, 1], [1, -1], [-1, -1]])
    
    movement_cost = np.array([1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414])

    # Initialize data structures
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)

    # Dictionary to store cost from start to each node
    g_score = {start_tuple: 0}

    # Dictionary to store estimated total cost from start to goal through each node
    f_score = {start_tuple: np.linalg.norm(start - goal)}

    # Priority queue for open set
    open_set = [(f_score[start_tuple], 0, start_tuple)]  # (f_score, counter, position)
    heapq.heapify(open_set)

    # Counter to break ties in priority queue
    counter = 1

    # Dictionary to store parent nodes for reconstructing the path
    came_from = {}

    # Set to track nodes in the open set for faster lookup
    open_set_hash = {start_tuple}

    # Main loop
    while open_set:
        # Get node with lowest f_score
        _, _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        # If we reached the goal, reconstruct and return the path
        if current == goal_tuple:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_tuple)
            path.reverse()
            retPath = np.asarray(path)
            return np.hstack((retPath[:, 1].reshape((-1, 1)), retPath[:, 0].reshape((-1, 1))))

        # Explore neighbors
        current_pos = np.array(current)

        for i, direction in enumerate(directions):
            neighbor = current_pos + direction
            neighbor_tuple = tuple(neighbor)

            # Check if neighbor is within bounds
            if not (
                0 <= neighbor[0] < map.shape[0] and 0 <= neighbor[1] < map.shape[1]
            ):
                continue

            # Check if neighbor is obstacle-free
            if map[neighbor_tuple] >= 0.5:
                continue

            # Calculate tentative g_score for this neighbor
            tentative_g_score = g_score[current] + movement_cost[i]

            # If this path to neighbor is better than any previous one
            if (
                neighbor_tuple not in g_score
                or tentative_g_score < g_score[neighbor_tuple]
            ):
                # Update path and scores
                came_from[neighbor_tuple] = current
                g_score[neighbor_tuple] = tentative_g_score
                h_score = np.linalg.norm(neighbor - goal)
                f_score[neighbor_tuple] = tentative_g_score + h_score

                if neighbor_tuple not in open_set_hash:
                    counter += 1
                    heapq.heappush(
                        open_set, (f_score[neighbor_tuple], counter, neighbor_tuple)
                    )
                    open_set_hash.add(neighbor_tuple)

    # If we get here, no path was found
    return []


def voronoi(map: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    # TODO
    return np.zeros((0, 2))


class PlannerType(Enum):
    ASTAR = 0
    VORONOI = 1


def plannerFactory(
    planner_type: PlannerType,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Factory method for planner to be used in benchmark
    """
    if planner_type == PlannerType.ASTAR:
        return aStar
    elif planner_type == PlannerType.VORONOI:
        return voronoi
    else:
        raise NotImplementedError()


# Example usage:
if __name__ == "__main__":
    # Create a simple map (0 = free space, 1 = obstacle)
    test_map = np.zeros((10, 10), dtype=np.int8)

    # Add some obstacles
    test_map[2:8, 5] = 1  # Vertical wall
    test_map[5, 0:5] = 1  # Horizontal wall

    # Define start and end points
    start_point = np.array([1, 1])
    end_point = np.array([8, 8])

    # Find path
    path = aStar(test_map, start_point, end_point)

    # Print result
    if path:
        print("Path found:")
        for point in path:
            print(point)

        # Visualize the path
        vis_map = test_map.copy().astype(str)
        vis_map[vis_map == "0"] = "."
        vis_map[vis_map == "1"] = "#"

        # Mark path
        for y, x in path:
            vis_map[y, x] = "o"

        # Mark start and end
        vis_map[tuple(start_point)] = "S"
        vis_map[tuple(end_point)] = "E"

        # Print the map
        for row in vis_map:
            print("".join(row))
    else:
        print("No path found")
