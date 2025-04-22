import numpy as np


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.

    Parameters:
    -----------
    point : numpy.ndarray
        (x, y) point to check.
    polygon : numpy.ndarray
        Array of polygon vertices as (x, y) coordinates.

    Returns:
    --------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_polygon_area(vertices):
    """
    Calculate the area of a polygon using the Shoelace formula.

    Parameters:
    -----------
    vertices : numpy.ndarray
        Array of shape (n, 2) containing the vertices of the polygon.

    Returns:
    --------
    float
        Area of the polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Using the Shoelace formula
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def triangulate_polygon(vertices):
    """
    Triangulate a polygon using ear clipping algorithm.
    This is a simple implementation that works for convex polygons.
    For complex polygons, consider using libraries like Shapely.

    Parameters:
    -----------
    vertices : numpy.ndarray
        Array of shape (n, 2) containing the vertices of the polygon.

    Returns:
    --------
    list
        List of triangles, where each triangle is a numpy array of shape (3, 2).
    """
    # For a quadrilateral, we can use a simple triangulation
    if len(vertices) == 4:
        return [
            np.array([vertices[0], vertices[1], vertices[2]]),
            np.array([vertices[0], vertices[2], vertices[3]]),
        ]

    # For a triangle, return as is
    if len(vertices) == 3:
        return [vertices]

    # For more complex polygons, use ear clipping
    # This is a simplified version that works for convex polygons
    triangles = []
    remaining_vertices = vertices.copy()

    while len(remaining_vertices) > 3:
        # Find an ear
        n = len(remaining_vertices)
        for i in range(n):
            # Vertices of the potential ear
            prev_idx = (i - 1) % n
            curr_idx = i
            next_idx = (i + 1) % n

            # Form a triangle
            triangle = np.array(
                [
                    remaining_vertices[prev_idx],
                    remaining_vertices[curr_idx],
                    remaining_vertices[next_idx],
                ]
            )

            # Check if this ear is valid
            # For convex polygons, all ears are valid
            triangles.append(triangle)

            # Remove the middle vertex of the ear
            remaining_vertices = np.delete(remaining_vertices, curr_idx, axis=0)
            break

    # Add the final triangle
    triangles.append(remaining_vertices)

    return triangles
