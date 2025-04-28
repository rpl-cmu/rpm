from typing import Any
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.ndimage as ndimage
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter


def create_repulsion_field(map_array):
    filtered = gaussian_filter(1 - map_array, sigma=3)
    sx = ndimage.sobel(filtered, axis=0)
    sy = ndimage.sobel(filtered, axis=1)
    return sx, sy, np.sqrt(sx * sx + sy * sy)


def bilinear_interpolation_2d(input_array, point):
    """
    Perform bilinear interpolation on a 2D numpy array at a single query point.

    Parameters:
    -----------
    input_array : numpy.ndarray
        The input 2D array from which to interpolate values.
    point : tuple or list or numpy.ndarray
        The (x, y) position at which to interpolate.

    Returns:
    --------
    float
        Interpolated value at the given position.
    """
    # Convert point to numpy array if it's not already
    point = np.asarray(point)
    x, y = point

    # Get the dimensions of the input array
    height, width = input_array.shape

    # Ensure the point is within the bounds of the array
    if x < 0 or x >= height - 1 or y < 0 or y >= width - 1:
        print(point)
        print(input_array.shape)
        raise ValueError("Point coordinates must be within the array bounds")

    # Get the four surrounding points
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    # Calculate the interpolation weights
    wx = x - x0
    wy = y - y0

    # Perform the bilinear interpolation
    value = (
        (1 - wx) * (1 - wy) * input_array[x0, y0]
        + wx * (1 - wy) * input_array[x0, y1]
        + (1 - wx) * wy * input_array[x1, y0]
        + wx * wy * input_array[x1, y1]
    )

    return value


def potential_field_planning(map_array, start, goal, cache: Any):
    # Convert inputs to numpy arrays for easier computation
    current_pos = np.array(start, dtype=float)
    goal_pos = np.array(goal, dtype=float)

    # Get map dimensions
    height, width = map_array.shape

    # Precompute repulsion field
    repulsion_field_x, repulsion_field_y, _ = create_repulsion_field(map_array)

    repulsion_gain = 10

    def ode(t, pos):
        diff = goal_pos - pos
        norm = np.linalg.norm(diff)
        if np.linalg.norm(diff) < 1:
            return np.zeros((2,))
        diff = diff / norm

        grad = diff
        if (
            pos[0] >= 0
            and np.ceil(pos[0]) < height
            and pos[1] >= 0
            and np.ceil(pos[1]) < width
        ):
            grad[0] += repulsion_gain * bilinear_interpolation_2d(
                repulsion_field_x, pos
            )
            grad[1] += repulsion_gain * bilinear_interpolation_2d(
                repulsion_field_y, pos
            )

        print(np.linalg.norm(grad))

        if np.linalg.norm(grad) < 0.1:
            grad[0] = -repulsion_gain * bilinear_interpolation_2d(
                repulsion_field_y, pos
            )
            grad[1] = repulsion_gain * bilinear_interpolation_2d(
                repulsion_field_x, pos
            )

        return grad

    path = list()
    for j in range(2000):
        new_pos = ode(0, current_pos) + current_pos
        path.append(new_pos)
        current_pos = new_pos

    result = np.asarray(path)
    retPath = np.empty_like(result)
    retPath[:, 0] = result[:, 1]
    retPath[:, 1] = result[:, 0]
    return retPath


# Example usage
def test_potential_field():
    # Create a simple map with obstacles
    map_size = 50
    test_map = np.zeros((map_size, map_size))

    # Add some obstacles
    for i in range(10, 20):
        test_map[i, 25] = 1
    for i in range(30, 40):
        test_map[i, 15] = 1

    # Define start and goal positions
    start_pos = (20, 3)
    goal_pos = (10, 80)

    # Visualize just the repulsion field

    pred_map = pkl.load(
        open(
            "/home/alex/GitHub_dev/radar_mapping/roboSAR/evaluation/data/exps/map_sar/cic_mimo/prob.pkl",
            "rb",
        )
    )
    pred_data = pred_map["data"].astype(np.float32)
    pred_data[pred_data < 0.4] = 0
    pred_data[pred_data >= 0.4] = 1

    sub_map = pred_data[50:100, 150:250]
    result_path = potential_field_planning(sub_map, start_pos, goal_pos)
    repulsion_field_x, repulsion_field_y, _ = create_repulsion_field(sub_map)

    # height, width = sub_map.shape
    # print(f"sub map size: {sub_map.shape}")
    # Y, X = np.mgrid[0:height, 0:width]

    plt.quiver(repulsion_field_y, repulsion_field_x)
    plt.plot(result_path[:, 1], result_path[:, 0], "-o")
    plt.imshow(sub_map, origin="lower")
    plt.show()


if __name__ == "__main__":
    test_potential_field()

