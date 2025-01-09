# measure_grasp.py

import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
import cv2
import pyzed.sl as sl
from ultralytics import SAM
from sklearn.decomposition import PCA
from skimage import measure, morphology, graph
from sklearn.cluster import DBSCAN

# Import the base class from its module
from .MeasureAnything import MeasureAnything

class MeasureGrasp(MeasureAnything):
    def __init__(self, zed, stride, thin_and_long=False, window=30, image_file=None):
        # Initialize the base class
        super().__init__(zed, stride, thin_and_long, window, image_file)
        # Additional initialization for grasp functionality can be added here if needed
    
    def update_calibration_params(self, depth, intrinsics, distortion=[0, 0, 0, 0, 0]):
        self.depth = depth
        self.fx, self.fy = intrinsics[1, 1], intrinsics[0, 0]
        self.cx, self.cy = intrinsics[0, 2], intrinsics[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = distortion[0:5]

    def grasp_stability_score(self, line_segment_coordinates, w1=0.5, w2=0.5, top_k=7):
        """
        Identify and return the top K best line segments based on stability scores while ensuring
        that no two selected line segments are within a specified spatial window to maintain separation.
        Additionally, exclude segments from the top 10% and bottom 10% indices of the line segments.
        """
        if self.centroid is None:
            raise ValueError("Centroid not defined. Ensure that the skeleton has been built before calculating stability scores.")

        num_segments = len(line_segment_coordinates)
        if num_segments == 0:
            return np.array([])  # Return an empty array if there are no segments

        # Calculate exclusion indices (top 10% and bottom 10%)
        exclusion_ratio = 0.1
        exclusion_count = int(np.ceil(exclusion_ratio * num_segments))
        exclude_indices = set(range(0, exclusion_count)) | set(range(num_segments - exclusion_count, num_segments))
        
        # Initialize arrays to store metrics
        avg_distances = np.zeros(num_segments)
        lengths = np.zeros(num_segments)
        midpoints = np.zeros((num_segments, 2))  # To store midpoints for distance calculations

        # Compute average distance to centroid and length for each segment
        for i, (y1, x1, y2, x2) in enumerate(line_segment_coordinates):
            # Compute distances from endpoints to centroid
            dist1 = np.linalg.norm([x1 - self.centroid[1], y1 - self.centroid[0]])
            dist2 = np.linalg.norm([x2 - self.centroid[1], y2 - self.centroid[0]])
            avg_distances[i] = (dist1 + dist2) / 2.0

            # Compute length of the segment
            lengths[i] = np.linalg.norm([x2 - x1, y2 - y1])

            # Compute midpoint
            midpoints[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        # Handle cases where all distances or lengths are the same to avoid division by zero
        if np.max(avg_distances) - np.min(avg_distances) == 0:
            avg_distances_norm = np.ones(num_segments)
        else:
            avg_distances_norm = (avg_distances - np.min(avg_distances)) / (np.max(avg_distances) - np.min(avg_distances))

        if np.max(lengths) - np.min(lengths) == 0:
            lengths_norm = np.ones(num_segments)
        else:
            lengths_norm = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))

        # Compute base scores (higher scores indicate better stability)
        # Invert normalized metrics because lower distance and shorter length are preferable
        scores = w1 * (1 - avg_distances_norm) + w2 * (1 - lengths_norm)

        # Determine window size for spatial separation (5 times the stride in pixels)
        min_separation = 5 * self.stride  # in pixels

        # Identify local minima in the lengths array within the window
        local_minima = np.zeros(num_segments, dtype=bool)
        window_size = 5 * self.stride
        if window_size < 1:
            window_size = 1  # Ensure at least a window size of 1

        for i in range(num_segments):
            # Define the window boundaries
            start = max(0, i - window_size)
            end = min(num_segments, i + window_size + 1)

            # Extract the window for comparison
            window = lengths[start:end]

            # Current segment's length
            current_length = lengths[i]

            # Check if the current segment's length is the minimum in the window
            if current_length == np.min(window):
                local_minima[i] = True

        # Define the reward to be added for local minima
        reward = 0.5  # Adjust this value as needed

        # Add the reward to segments that are local minima
        scores[local_minima] += reward

        # Sort the segments by score in descending order
        sorted_indices = np.argsort(scores)[::-1]

        # Initialize a list to store selected segment indices
        selected_indices = []

        # Initialize an array to keep track of selected midpoints
        selected_midpoints = []

        for idx in sorted_indices:
            # Exclude segments from the top and bottom 10%
            if idx in exclude_indices:
                continue  # Skip excluded segments

            current_midpoint = midpoints[idx]

            if not selected_midpoints:
                # Select the first valid segment
                selected_indices.append(idx)
                selected_midpoints.append(current_midpoint)
            else:
                # Compute distances from current segment's midpoint to all selected segments' midpoints
                distances = cdist([current_midpoint], selected_midpoints, metric='euclidean').flatten()

                # Check if all distances are greater than or equal to min_separation
                if np.all(distances >= min_separation):
                    selected_indices.append(idx)
                    selected_midpoints.append(current_midpoint)

            if len(selected_indices) == top_k:
                break  # Stop if we've selected enough segments

        # Extract the top segments based on the selected indices
        top_segments = line_segment_coordinates[selected_indices]

        return top_segments, selected_indices
    
    def depth_values_to_3d_points(self, x, y, depth, intrinsics):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x_3d = (x - cx) * depth / fx
        y_3d = (y - cy) * depth / fy
        z_3d = depth
        return [x_3d, y_3d, z_3d]

    def convert_grasp_to_3d(self, grasp, depth_values, rgb_intrinsics):
        """
        Convert 2D grasp coordinates and depth values to 3D grasp points using camera intrinsics.
        """
        grasp_pair = []

        # Iterate over grasps [y1, x1, y2, x2] and depth values
        for (y1, x1, y2, x2), (d1, d2) in zip(grasp, depth_values):
            # Convert depth values to 3D points using inherited method
            p1_3d = self.depth_values_to_3d_points(x1, y1, d1, rgb_intrinsics)
            p2_3d = self.depth_values_to_3d_points(x2, y2, d2, rgb_intrinsics)
            grasp_pair.append((p1_3d, p2_3d))
        
        return grasp_pair

    def create_gripper_lines(self, grasp_3d, line_color):
        """
        Creates gripper lines based on 3D grasp points, forming a C-like shape aligned along the z-axis.
        Each gripper line is assigned a specific color based on the grasp pair's rank.
        """
        gripper_points = []
        gripper_lines = []
        gripper_line_colors = []
        current_index = 0

        # Define fixed direction along z-axis
        direction = np.array([0, 0, 1])

        # Define gripper dimensions
        gripper_length = 0.01  # meters
        gripper_depth = 0.02   # meters
        thickness = 0.0002     # meters

        # Define number of parallel lines per finger to simulate thickness
        num_parallel_lines = 20

        # Generate offset values for parallel lines
        offsets = np.linspace(-thickness, thickness, num_parallel_lines)

        for grasp_pair in grasp_3d:
            p1, p2 = grasp_pair
            p1 = np.array(p1)
            p2 = np.array(p2)
            midpoint = (p1 + p2) / 2
            norm = np.linalg.norm(p2 - p1)

            # Update gripper depth based on the grasp pair
            gripper_depth = norm

            # Define gripper base center points symmetrically along y-axis
            gripper_base_center_left = midpoint + np.array([0, gripper_depth / 2, 0])
            gripper_base_center_right = midpoint - np.array([0, gripper_depth / 2, 0])

            # Define gripper tip center points extending along z-axis
            gripper_tip_center_left = gripper_base_center_left + direction * gripper_length / 2
            gripper_tip_center_right = gripper_base_center_right + direction * gripper_length / 2

            # Define base points offset for thickness
            gripper_base_left_centered = gripper_base_center_left - direction * gripper_length / 2
            gripper_base_right_centered = gripper_base_center_right - direction * gripper_length / 2

            for offset in offsets:
                offset_vector = np.array([offset, 0, 0])

                gripper_base_left = gripper_base_left_centered + offset_vector
                gripper_base_right = gripper_base_right_centered + offset_vector
                gripper_tip_left = gripper_tip_center_left + offset_vector
                gripper_tip_right = gripper_tip_center_right + offset_vector

                gripper_points.extend([gripper_base_left, gripper_tip_left, gripper_base_right, gripper_tip_right])

                # Define lines and assign colors
                gripper_lines.append((current_index, current_index + 1))
                gripper_line_colors.append(line_color)
                gripper_lines.append((current_index + 2, current_index + 3))
                gripper_line_colors.append(line_color)
                gripper_lines.append((current_index, current_index + 2))
                gripper_line_colors.append(line_color)
                
                current_index += 4

        return np.array(gripper_points), gripper_lines, gripper_line_colors
