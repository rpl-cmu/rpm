import numpy as np

from .geometryUtils import *

def sample_points_from_different_polygons(points, polygon_list):
    """
    For each input point, sample a random point from a different polygon
    than the one containing the original point.
    
    Parameters:
    -----------
    points : list or numpy.ndarray
        List of (x, y) points to find corresponding points for.
    polygon_list : list of lists
        List of polygons, where each polygon is a list of (x, y) coordinate vertices.
        Each polygon should have at least 3 vertices.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (len(points), 2) containing the corresponding points,
        each sampled from a different polygon than the one containing the original point.
    """
    # Convert inputs to numpy arrays
    points = np.array(points)
    polygons = [np.array(poly) for poly in polygon_list]
    
    # Make sure there are at least 2 polygons
    if len(polygons) < 2:
        raise ValueError("At least 2 polygons are required to sample from different boxes")
    
    # Calculate areas for each polygon
    polygon_areas = []
    for poly in polygons:
        area = calculate_polygon_area(poly)
        polygon_areas.append(area)
    
    # Determine which polygon contains each point
    point_polygon_indices = []
    for point in points:
        contained_in = []
        for i, poly in enumerate(polygons):
            if point_in_polygon(point, poly):
                contained_in.append(i)
        
        if not contained_in:
            # If the point is not in any polygon, we'll consider it as in none
            contained_in = [-1]
        
        point_polygon_indices.append(contained_in)
    
    # Sample corresponding points
    corresponding_points = np.zeros_like(points)
    
    for i, point in enumerate(points):
        # Get polygons that contain the current point
        contained_in = point_polygon_indices[i]
        
        # Get indices of other polygons
        other_poly_indices = [j for j in range(len(polygons)) if j not in contained_in]
        
        # If all polygons contain the point (rare edge case), pick any polygon
        if not other_poly_indices:
            other_poly_indices = list(range(len(polygons)))
        
        # Calculate probabilities for the other polygons (proportional to their areas)
        other_areas = [polygon_areas[j] for j in other_poly_indices]
        other_probs = [area / sum(other_areas) for area in other_areas]
        
        # Choose a different polygon based on area proportions
        different_poly_idx = np.random.choice(other_poly_indices, p=other_probs)
        
        # Sample a point from the different polygon
        different_poly = polygons[different_poly_idx]
        corresponding_points[i] = sample_from_single_polygon(different_poly, 1)[0]

    retVal = np.empty_like(corresponding_points)
    retVal[:, 0] = corresponding_points[:, 1]
    retVal[:, 1] = corresponding_points[:, 0]
    
    return retVal

def sample_from_polygons(polygon_list, num_samples=1000):
    """
    Generate uniformly distributed random points within multiple polygons.
    
    Parameters:
    -----------
    polygon_list : list of lists
        List of polygons, where each polygon is a list of (x, y) coordinate vertices.
        Each polygon should have at least 3 vertices.
    num_samples : int, optional
        Total number of points to generate across all polygons. Default is 1000.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (num_samples, 2) containing the sampled points.
    """
    # Convert polygons to numpy arrays if not already
    polygons = [np.array(poly) for poly in polygon_list]
    
    # Calculate areas for each polygon
    polygon_areas = []
    for poly in polygons:
        # For polygons with more than 3 vertices, we triangulate
        area = calculate_polygon_area(poly)
        polygon_areas.append(area)
    
    # Total area of all polygons
    total_area = sum(polygon_areas)
    
    # Sampling probabilities based on polygon areas
    polygon_probs = [area / total_area for area in polygon_areas]
    
    # Number of points to sample from each polygon
    # We use multinomial to get an integer distribution of points
    points_per_polygon = np.random.multinomial(num_samples, polygon_probs)
    
    # Sample points from each polygon
    all_points = []
    for i, poly in enumerate(polygons):
        if points_per_polygon[i] > 0:
            points = sample_from_single_polygon(poly, points_per_polygon[i])
            all_points.append(points)
    
    # Combine all sampled points
    all_points = np.vstack(all_points)
    
    # Shuffle the points to remove any ordering bias
    np.random.shuffle(all_points)

    retVal = np.empty_like(all_points)
    retVal[:, 0] = all_points[:, 1]
    retVal[:, 1] = all_points[:, 0]
    
    return retVal.astype(np.int32)

def sample_from_single_polygon(vertices, num_samples):
    """
    Generate uniformly distributed random points within a single polygon.
    
    Parameters:
    -----------
    vertices : numpy.ndarray
        Array of shape (n, 2) containing the vertices of the polygon.
    num_samples : int
        Number of points to generate.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (num_samples, 2) containing the sampled points.
    """
    # Triangulate the polygon
    triangles = triangulate_polygon(vertices)
    
    # Calculate areas of each triangle
    areas = []
    for tri in triangles:
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = 0.5 * np.abs(np.cross(v1, v2))
        areas.append(area)
    
    # Normalize areas to get probability distribution
    total_area = sum(areas)
    probabilities = [area / total_area for area in areas]
    
    # Initialize array to store sampled points
    points = np.zeros((num_samples, 2))
    
    # Sample points
    for i in range(num_samples):
        # Choose a triangle with probability proportional to its area
        tri_idx = np.random.choice(len(triangles), p=probabilities)
        selected_triangle = triangles[tri_idx]
        
        # Generate barycentric coordinates
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Ensure the point is within the triangle
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
            
        # Convert barycentric coordinates to Cartesian
        point = (1 - r1 - r2) * selected_triangle[0] + r1 * selected_triangle[1] + r2 * selected_triangle[2]
        
        # Store the point
        points[i] = point
    
    return points
