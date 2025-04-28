import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt

def voronoi(map: np.ndarray, start: np.ndarray, goal: np.ndarray, cache) -> np.ndarray:
    node_list = find_path(cache, [start[1], start[0]], [goal[1], goal[0]])
    if (len(node_list) == 0):
        return np.zeros((0, 2))
    positions = [cache.nodes[node]['pos'] for node in node_list]
    return np.asarray(positions)

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
        return nx.shortest_path(G, start_node, end_node)
    except nx.NetworkXNoPath:
        print("No path found between the specified points.")
        return list()

def plot_results(prob_map, G, path=None):
    """
    Plot the results of Voronoi path planning.

    Parameters:
    -----------
    prob_map : numpy.ndarray
        Original probability map
    G : networkx.Graph
        Graph representation of Voronoi diagram
    skeleton : numpy.ndarray
        Binary skeleton image
    path : list, optional
        List of node indices representing a path
    """
    fig, axs = plt.subplots(3, 1, figsize=(6, 18))

    # Plot probability map
    axs[0].imshow(prob_map, cmap="gray")
    axs[0].set_title("Probability Map")

    # Plot distance transform with skeleton overlay
    dist_transform = distance_transform_edt(~(prob_map > 0.5))
    dist_norm = dist_transform / np.max(dist_transform)
    axs[1].imshow(dist_norm, cmap="viridis")
    # axs[1].contour(skeleton, [0.5], colors="red", linewidths=1)
    axs[1].set_title("Distance Transform with Voronoi Skeleton")
    axs[1].axis("off")

    # Plot graph
    axs[2].imshow(prob_map, cmap="gray", alpha=0.5)

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # Calculate node colors based on clearance
    clearance = np.array([G.nodes[i]["clearance"] for i in G.nodes()])
    norm_clearance = clearance / clearance.max()

    # Draw nodes with color indicating clearance
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=5,
        node_color=norm_clearance,
        cmap="plasma",
        ax=axs[2],
        alpha=0.7,
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=axs[2])

    # If a path is provided, draw it
    if path is not None:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, width=2, edge_color="red", ax=axs[2]
        )

        # Mark start and end
        start_pos = pos[path[0]]
        end_pos = pos[path[-1]]
        axs[2].plot(start_pos[0], start_pos[1], "go", markersize=10)
        axs[2].plot(end_pos[0], end_pos[1], "ro", markersize=10)

    axs[2].set_title("Voronoi Graph")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import pickle as pkl
    pred_map = pkl.load(
        open(
            "/home/alex/GitHub_dev/radar_mapping/roboSAR/evaluation/data/exps/map_sar/cic_mimo/prob.pkl",
            "rb",
        )
    )
    pred_data = pred_map["data"].astype(np.float32)
    pred_data[pred_data < 0.4] = 0
    pred_data[pred_data >= 0.4] = 1

    sub_map = pred_data[0:150, 0:400]

    graph = generate_voronoi_from_probability_map(sub_map)

    start_pos = (70, 153)
    goal_pos = (60, 230)

    path = voronoi(None, start_pos, goal_pos, graph)
   
    plot_results(sub_map, graph, path)
