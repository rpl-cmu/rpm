import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from scipy.spatial import Voronoi
import networkx as nx
import pickle as pkl
from pathlib import Path
import tyro


def generate_voronoi_from_probability_map(prob_map, threshold=0.4, min_distance=0.1):
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

    return G, skeleton


def plot_results(prob_map, G, skeleton, path=None):
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
    axs[0].axis("off")

    # Plot distance transform with skeleton overlay
    dist_transform = distance_transform_edt(~(prob_map > 0.5))
    dist_norm = dist_transform / np.max(dist_transform)
    axs[1].imshow(dist_norm, cmap="viridis")
    axs[1].contour(skeleton, [0.5], colors="red", linewidths=1)
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
    if path:
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
        return path
    except nx.NetworkXNoPath:
        print("No path found between the specified points.")
        return None


# Example usage with a simple probability map
def example_usage():
    # Create a 100x100 probability map (higher value = more likely obstacle)
    size = 100
    prob_map = np.zeros((size, size))

    # Add some obstacles (higher probability regions)
    for i in range(10):
        x = np.random.randint(10, size - 10)
        y = np.random.randint(10, size - 10)
        r = np.random.randint(3, 10)

        # Create circular obstacles with Gaussian probability distribution
        y_grid, x_grid = np.ogrid[-y : size - y, -x : size - x]
        mask = (x_grid**2 + y_grid**2) <= r**2
        # a = x_grid[mask]

        t = np.exp(-((x_grid**2 + y_grid**2) / (2 * r**2)))

        prob_map[mask] = np.maximum(prob_map[mask], t[mask] * 0.8 + 0.2)

    # Add some walls
    prob_map[20:80, 30:33] = 0.9  # Vertical wall with a small gap
    prob_map[20:50, 70:73] = 0.9  # Another vertical wall

    # Generate Voronoi graph from probability map
    G, skeleton = generate_voronoi_from_probability_map(prob_map, threshold=0.5)

    # Define start and end points
    start_point = (10, 10)
    end_point = (90, 90)

    # Find path
    path = find_path(G, start_point, end_point)

    # Plot results
    plot_results(prob_map, G, skeleton, path)


# For custom probability map input:
def process_custom_map(your_prob_map):
    """
    Process a user-provided probability map.

    Parameters:
    -----------
    your_prob_map : numpy.ndarray
        2D array representing your probability map

    Returns:
    --------
    None (displays results)
    """
    # Generate Voronoi graph
    G, skeleton = generate_voronoi_from_probability_map(your_prob_map)

    # Define start and end points (adjust as needed)
    print("Finished pre-processing maps")

    starts = np.array([277, 147])
    goals = np.array([277, 127])

    # start_point = (10, 10)
    # end_point = (your_prob_map.shape[1] - 10, your_prob_map.shape[0] - 10)

    # Find path
    path = find_path(G, starts, goals)

    # Plot results
    plot_results(your_prob_map, G, skeleton, path)

    return G, skeleton, path

def main(map_file: Path):
    pred_map = pkl.load(open(map_file, "rb"))
    prob_map = pred_map["data"].astype(np.float32)
    # Generate Voronoi graph from probability map
    process_custom_map(prob_map)
    # G, skeleton = generate_voronoi_from_probability_map(
    #     prob_map, threshold=0.45, min_distance=0.2
    # )

    # # Plot results
    # plot_results(prob_map, G, skeleton)


if __name__ == "__main__":
    # example_usage()
    tyro.cli(main)



# Example of how to use with your own probability map:
"""
# Load your probability map
your_prob_map = np.load('your_map.npy')  # or read from image, etc.

# Process it
G, skeleton, path = process_custom_map(your_prob_map)
"""
