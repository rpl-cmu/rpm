import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx

def voronoi(map: np.ndarray, start: np.ndarray, goal: np.ndarray, cache) -> np.ndarray:
    return find_path(cache, [start[1], start[0]], [goal[1], goal[0]])


def generate_voronoi_from_probability_map(prob_map):
    """
    Generate a Voronoi graph from a 2D probability map for path planning.

    Parameters:
    -----------
    prob_map : numpy.ndarray
        2D array representing probability of obstacles (higher value = more likely to be obstacle)
    threshold : float
        Probability threshold above which a cell is considered an obstacle
    min_distance : int
        Minimum distance between obstacle centers (used for peak finding)

    Returns:
    --------
    G : networkx.Graph
        Graph representation of the Voronoi diagram
    skeleton : numpy.ndarray
        Binary image of the Voronoi skeleton
    """
    # Create binary obstacle map from probability map
    threshold=0.4
    min_distance=0.1
    obstacle_map = prob_map > threshold

    # Calculate distance transform (distance to nearest obstacle)
    dist_transform = distance_transform_edt(~obstacle_map)

    # Method 1: Skeletonization approach (works best for dense obstacle maps)
    # Normalize distance transform for visualization
    dist_norm = dist_transform / np.max(dist_transform)

    # Create a skeleton of the distance transform
    # This gives us the Voronoi edges (equidistant from obstacles)
    skeleton = skeletonize(dist_norm > min_distance)  # Adjust threshold as needed

    # Create a graph from the skeleton
    G = nx.Graph()

    # Find edge pixels in the skeleton
    edge_pixels = np.argwhere(skeleton)

    # Add nodes for each pixel in the skeleton
    for i, (y, x) in enumerate(edge_pixels):
        G.add_node(i, pos=(x, y), clearance=dist_transform[y, x])

    # Connect neighboring pixels (8-connectivity)
    for i, (y1, x1) in enumerate(edge_pixels):
        for j, (y2, x2) in enumerate(edge_pixels):
            if i != j:
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < 1.5:  # Connect if neighbors (including diagonals)
                    # Edge weight is inverse of clearance to prefer high-clearance paths
                    weight = 1.0 / (
                        (dist_transform[y1, x1] + dist_transform[y2, x2]) / 2.0
                    )
                    G.add_edge(
                        i,
                        j,
                        weight=weight,
                        clearance=(dist_transform[y1, x1] + dist_transform[y2, x2])
                        / 2.0,
                    )

    return G


def find_path(G, start_point, end_point):
    """
    Find the shortest path on the Voronoi graph between two points.

    Parameters:
    -----------
    G : networkx.Graph
        Voronoi graph
    start_point : tuple
        (x, y) coordinates of start point
    end_point : tuple
        (x, y) coordinates of end point

    Returns:
    --------
    path : list
        List of node indices representing the path
    """
    # Find nearest nodes to start and end points
    pos = nx.get_node_attributes(G, "pos")

    start_node = min(
        G.nodes(),
        key=lambda n: np.sqrt(
            (pos[n][0] - start_point[0]) ** 2 + (pos[n][1] - start_point[1]) ** 2
        ),
    )
    end_node = min(
        G.nodes(),
        key=lambda n: np.sqrt(
            (pos[n][0] - end_point[0]) ** 2 + (pos[n][1] - end_point[1]) ** 2
        ),
    )

    # Find the shortest path (using edge weights that favor high clearance)
    try:
        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        return np.asarray(path)
    except nx.NetworkXNoPath:
        print("No path found between the specified points.")
        return np.zeros((0, 2))
