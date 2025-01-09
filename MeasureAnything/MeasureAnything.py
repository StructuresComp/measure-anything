import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage import measure
from skimage import morphology
from skimage import graph
from sklearn.cluster import DBSCAN
from collections import deque
import pyzed.sl as sl
from ultralytics import SAM
from scipy.spatial.distance import cdist


class MeasureAnything:
    def __init__(self, zed, stride, thin_and_long=False, window=30, image_file=None):
        # SAM model
        self.model = SAM("sam2.1_l.pt")

        self.image_file = image_file
        self.window = int(window)  # Steps in Central difference to compute the local slope
        self.stride = int(stride)  # Increment to next line segment

        self.thin_and_long = thin_and_long
        self.initial_binary_mask = None
        self.processed_binary_mask = None
        self.processed_binary_mask_0_255 = None
        self.skeleton = None
        self.skeleton_distance = None
        self.skeleton_coordinates = None
        self.endpoints = None
        self.intersections = None
        self.slope = {}
        self.line_segment_coordinates = None
        self.grasp_coordinates = None
        self.centroid = None

        if zed:
            # ZED camera
            self.zed = zed
            self.depth = sl.Mat()
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            self.fx, self.fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
            self.cx, self.cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy
            # Distortion factor : [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
            self.k1, self.k2, self.p1, self.p2, self.k3 = calibration_params.left_cam.disto[0:5]

    def detect_mask(self, image, positive_prompts, negative_prompts=None):
        """ Inference with SAM2 model, using given positive and negative prompts"""
        # Stack prompts
        prompts = np.vstack([positive_prompts, negative_prompts]) if negative_prompts is not None \
            else positive_prompts
        # Create labels for positive (1) and negative (0) prompts
        labels = np.zeros(prompts.shape[0], dtype=np.int8)
        labels[:positive_prompts.shape[0]] = 1

        # Run SAM2 prediction with prompts and labels
        masks = self.model(image, points=[prompts], labels=[labels])
        final_mask = masks[0].masks.data[0].cpu().numpy()

        self.initial_binary_mask = final_mask

    def process_mask(self):
        """ Process the binary mask output from SAM2, for improved skeletonization"""
        # Remove small objects
        mask_in_process = morphology.remove_small_objects(self.initial_binary_mask, min_size=100)

        # Connected Component Analysis
        label_image = measure.label(mask_in_process, background=0)

        # Find the properties of connected components
        props = measure.regionprops(label_image)

        # Find the largest connected component
        largest_area = 0
        largest_component = None
        for prop in props:
            if prop.area > largest_area:
                largest_area = prop.area
                largest_component = prop

        # Create a mask for the largest connected component
        mask_in_process = np.zeros_like(mask_in_process)
        mask_in_process[label_image == largest_component.label] = 1

        # Convert the binary mask to 0 and 255 format
        mask_in_process_0_255 = (mask_in_process * 255).astype(np.uint8)

        # Apply Morphological Opening to remove small protrusions
        kernel = np.ones((3, 3), np.uint8)  # Define the kernel size for the morphological operations
        mask_in_process_0_255 = cv2.morphologyEx(mask_in_process_0_255, cv2.MORPH_OPEN, kernel)

        # Apply Morphological Closing to fill small holes within the component
        mask_in_process_0_255 = cv2.morphologyEx(mask_in_process_0_255, cv2.MORPH_CLOSE, kernel)

        # Find contours of the mask (boundary of the object)
        contours, _ = cv2.findContours(mask_in_process_0_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate the contour to a polygon using approxPolyDP
        epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon as needed for simplification
        polygon = cv2.approxPolyDP(contours[0], epsilon, True)

        # Create an empty binary mask to draw the polygon
        height, width = mask_in_process.shape  # Size of the original binary mask
        binary_mask_from_polygon = np.zeros((height, width), dtype=np.uint8)  # Empty mask (all zeros)

        # Fill the polygon on the binary mask
        cv2.fillPoly(binary_mask_from_polygon, [polygon], color=255)  # Fill the polygon with white (255)

        # Convert the filled mask back to binary (0s and 1s) for internal use
        self.processed_binary_mask = (binary_mask_from_polygon > 0).astype(np.uint8)
        self.processed_binary_mask_0_255 = (self.processed_binary_mask * 255).astype(np.uint8)


    def skeletonize_and_prune(self):
        """ Skeletonize the processed binary mask and prune short branches. Preserve a single continuous path between two endpoints. """
        # Step 1: Apply Medial Axis Transform
        self.skeleton, self.skeleton_distance = morphology.medial_axis(self.processed_binary_mask_0_255,
                                                                       return_distance=True)

        # Step 2: Identify endpoints and intersections
        self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        # Step 3: Prune branches if more than two endpoints exist and re-identify
        if len(self.endpoints) != 2:
            self.skeleton = self._prune_short_branches(self.skeleton, self.endpoints,
                                                       self.intersections, 2 * np.max(self.skeleton_distance))
            self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        # Step 4: Preserve a single continuous skeleton along path
        # Select two endpoints with the greatest 'y' separation
        start_point = max(self.endpoints, key=lambda x: x[0])  # Endpoint with the highest i value
        end_point = min(self.endpoints, key=lambda x: x[0])  # Endpoint with the lowest i value

        self.skeleton_coordinates, self.skeleton = self._preserve_skeleton_path(self.skeleton, self.endpoints)
        self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        if len(self.endpoints) != 2:
            raise Exception("Number of endpoints of pruned skeleton is not 2")

    def augment_skeleton(self):
        """ Augment the skeleton by extending paths from both endpoints."""
        # Initialize the augmented skeleton
        augmented_skeleton = self.skeleton.copy()

        # Get image dimensions
        height, width = self.processed_binary_mask.shape

        # Sort endpoints to ensure the bottom endpoint comes first
        self.endpoints = sorted(self.endpoints, key=lambda p: p[0], reverse=True)

        def propagate_path_simple(y, x, dy, dx):
            """ Propagates a path from a given start point in a specified direction. Stops if the next pixel creates an intersection."""
            while True:
                # Round to nearest integer for pixel indexing
                y_int, x_int = int(round(y)), int(round(x))

                # Stop if out of bounds or hits the binary mask boundary
                if not (0 <= y_int < height and 0 <= x_int < width):
                    break
                if self.processed_binary_mask[y_int, x_int] == 0:
                    break

                # Check for intersection: count neighbors in the current skeleton
                neighbor_count = 0
                for dy_n, dx_n in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y_int + dy_n, x_int + dx_n
                    if 0 <= ny < height and 0 <= nx < width and augmented_skeleton[ny, nx] == 1:
                        neighbor_count += 1

                # Stop propagation if adding this pixel would create an intersection
                if neighbor_count > 2:
                    y += dy
                    x += dx
                    continue

                # Add pixel to the skeleton
                augmented_skeleton[y_int, x_int] = 1

                # Move to the next position along the slope
                y += dy
                x += dx

        # Extend paths from both endpoints
        for idx, endpoint in enumerate(self.endpoints):
            # Determine the slope using the skeleton points near the endpoint
            if idx == 0:  # Bottom endpoint
                y1, x1 = endpoint
                y2, x2 = self.skeleton_coordinates[min(len(self.skeleton_coordinates) - 1, self.window)]

                outward = -1  # Reverse direction
            elif idx == 1:  # Top endpoint
                y1, x1 = self.skeleton_coordinates[-min(1 + self.window, len(self.skeleton_coordinates))]
                y2, x2 = endpoint
                outward = 1  # Outward direction
            else:
                raise ValueError("Unexpected endpoint ordering.")

            # Compute the local slope
            slope = np.arctan2(y2 - y1, x2 - x1)

            # Calculate direction vector from the slope
            dx = outward * np.cos(slope)
            dy = outward * np.sin(slope)

            # Normalize the direction vector
            magnitude = np.sqrt(dx ** 2 + dy ** 2)
            dx /= magnitude
            dy /= magnitude

            # Propagate the path
            propagate_path_simple(y1, x1, dy, dx)

        # Update the skeleton and key points
        self.skeleton = augmented_skeleton
        self.skeleton_coordinates = np.argwhere(self.skeleton == True)
        self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        # Reorder skeleton coordinates to start from the bottom endpoint
        self.skeleton_coordinates = sorted(self.skeleton_coordinates, key=lambda p: p[0], reverse=True)

    def build_skeleton_from_symmetry_axis(self):
        """ Build the skeleton of a binary mask for general objects by finding symmetry axis """

        # Step 1: Find the closest axis of symmetry
        centerline, centroid = self._find_closest_symmetry_axis(self.processed_binary_mask)
        self.centroid = centroid

        # Step 2: Initialize skeleton points and traverse in both directions
        skeleton_points = []
        height, width = self.processed_binary_mask.shape
        directions = np.array([centerline, -centerline])  # Positive and negative directions

        for direction in directions:
            y, x = centroid[0], centroid[1]
            current_skeleton = []

            while True:
                # Round to nearest integer for pixel indexing
                y_int, x_int = int(round(y)), int(round(x))

                # Stop if out of bounds or hits the binary mask boundary
                if not (0 <= y_int < height and 0 <= x_int < width):
                    break
                if self.processed_binary_mask[y_int, x_int] == 0:
                    break

                # Add the current skeleton point
                current_skeleton.append((y_int, x_int))

                # Move to the next point along the direction vector
                y += direction[0]
                x += direction[1]

            # Append the current skeleton segment
            skeleton_points.append(current_skeleton)

        # Step 3: Reverse one skeleton part and concatenate
        positive_skeleton = skeleton_points[0]
        negative_skeleton = skeleton_points[1][::-1]  # Reverse the negative direction skeleton
        full_skeleton = negative_skeleton + positive_skeleton

        # Step 4: Ensure the bottommost skeleton point is the first index
        if full_skeleton[-1][0] > full_skeleton[0][0]:
            full_skeleton = full_skeleton[::-1]  # Reverse the order if needed

        self.skeleton_coordinates = np.array(full_skeleton)

        # Step 5: Define skeleton map
        self.skeleton = np.zeros_like(self.processed_binary_mask, dtype=np.uint8)
        for coord in self.skeleton_coordinates:
            y, x = coord
            self.skeleton[y, x] = 1

    def calculate_perpendicular_slope(self):
        """ Calculate the slope of the perpendicular line segments at each sampled skeleton. """
        # i = self.window  # The first line segment should be at least i = self.window to avoid indexing errors
        i = 2

        # while i < len(self.skeleton_coordinates) - self.window:
        while i < len(self.skeleton_coordinates) - 2:
            # Check if current skeleton coordinate meets the threshold condition
            # if self.skeleton_coordinates[i][0] <= (1 - self.threshold) * self.skeleton_coordinates[0][0]:
            #     break

            # Get current key point
            key = tuple(self.skeleton_coordinates[i])

            # Calculate the slope using points offset by `self.window` on either side
            y1, x1 = self.skeleton_coordinates[max(0, i - self.window)]
            y2, x2 = self.skeleton_coordinates[min(len(self.skeleton_coordinates) - 1, i + self.window)]
            self.slope[key] = np.arctan2(y2 - y1, x2 - x1)

            # Move to the next point based on stride
            i += self.stride

    def calculate_line_segment_coordinates_and_depth(self, threshold=0.1):
        """ Calculate line segment coordinates and median depths along the line segments. """
        height, width = self.processed_binary_mask_0_255.shape
        line_segment_coordinates = np.zeros((len(self.slope), 4), dtype=int)
        depths = []

        for idx, (key, val) in enumerate(self.slope.items()):
            # Get skeleton point
            y, x = key

            # Calculate direction vector for perpendicular line (normal direction)
            dx = -np.sin(val)
            dy = np.cos(val)

            # Normalize the direction vector
            magnitude = np.sqrt(dx ** 2 + dy ** 2)
            dx /= magnitude
            dy /= magnitude

            # Initialize line segment endpoints
            x1, y1 = x, y
            x2, y2 = x, y

            # Initialize lists to store valid depth values
            left_depths = []
            right_depths = []

            # Get initial depth at the skeleton midpoint
            # status, initial_depth = self.depth.get_value(int(round(x)), int(round(y)))
            if isinstance(self.depth, np.ndarray):
                initial_depth = self.depth[int(round(y)), int(round(x))]
                status = sl.ERROR_CODE.SUCCESS  # Assuming access is always successful
            else:
                status, initial_depth = self.depth.get_value(int(round(x)), int(round(y)))
            if status != sl.ERROR_CODE.SUCCESS or initial_depth <= 0:
                depths.append((np.nan, np.nan))
                continue

            # Propagate in one direction (left)
            while True:
                # Update coordinates
                x1 -= dx
                y1 -= dy

                # Check bounds
                if not (0 <= int(round(y1)) < height and 0 <= int(round(x1)) < width):
                    break
                if not self.processed_binary_mask_0_255[int(round(y1)), int(round(x1))]:
                    break

                # Update depth at (x1, y1) if valid
                # status, new_depth = self.depth.get_value(int(round(x1)), int(round(y1)))
                if isinstance(self.depth, np.ndarray):
                    new_depth = self.depth[int(round(y1)), int(round(x1))]
                    status = sl.ERROR_CODE.SUCCESS  # Assuming access is always successful
                else:
                    status, new_depth = self.depth.get_value(int(round(x1)), int(round(y1)))
                if status == sl.ERROR_CODE.SUCCESS and new_depth > 0:
                    # Append only if within the threshold
                    if abs(new_depth - initial_depth) <= threshold * initial_depth:
                        left_depths.append(new_depth)
            x1 += dx
            y1 += dy

            # Propagate in the opposite direction (right)
            while True:
                # Update coordinates
                x2 += dx
                y2 += dy

                # Check bounds
                if not (0 <= int(round(y2)) < height and 0 <= int(round(x2)) < width):
                    break
                if not self.processed_binary_mask_0_255[int(round(y2)), int(round(x2))]:
                    break

                # Update depth at (x2, y2) if valid
                # status, new_depth = self.depth.get_value(int(round(x2)), int(round(y2)))
                if isinstance(self.depth, np.ndarray):
                    new_depth = self.depth[int(round(y2)), int(round(x2))]
                    status = sl.ERROR_CODE.SUCCESS  # Assuming access is always successful
                else:
                    status, new_depth = self.depth.get_value(int(round(x2)), int(round(y2)))
                if status == sl.ERROR_CODE.SUCCESS and new_depth > 0:
                    # Append only if within the threshold
                    if abs(new_depth - initial_depth) <= threshold * initial_depth:
                        right_depths.append(new_depth)
            x2 -= dx
            y2 -= dy

            # Store integer coordinates of endpoints
            line_segment_coordinates[idx] = [
                int(np.clip(round(y1), 0, height - 1)),
                int(np.clip(round(x1), 0, width - 1)),
                int(np.clip(round(y2), 0, height - 1)),
                int(np.clip(round(x2), 0, width - 1))
            ]

            # Calculate median depths or assign NaN if no valid depth values were found
            median_depth1 = np.median(left_depths) if left_depths else np.nan
            median_depth2 = np.median(right_depths) if right_depths else np.nan
            depths.append((median_depth1, median_depth2))

        return line_segment_coordinates, np.array(depths)

    def calculate_diameter(self, line_segment_coordinates, depth):
        """ Calculate the diameters from each line segment"""
        # Array to store diameters
        diameters = np.zeros(len(line_segment_coordinates))

        for i, (y1, x1, y2, x2) in enumerate(line_segment_coordinates):
            # Undistort the endpoints
            x1_ud, y1_ud = self._undistort_point(x1, y1)
            x2_ud, y2_ud = self._undistort_point(x2, y2)

            # Triangulate the 3D points of (x1, y1) and (x2, y2) using depth
            z1, z2 = depth[i]
            x1_3d = (x1_ud - self.cx) * z1 / self.fx
            y1_3d = (y1_ud - self.cy) * z1 / self.fy
            x2_3d = (x2_ud - self.cx) * z2 / self.fx
            y2_3d = (y2_ud - self.cy) * z2 / self.fy

            # 3D coordinates of endpoints
            point1_3d = np.array([x1_3d, y1_3d, z1])
            point2_3d = np.array([x2_3d, y2_3d, z2])

            # Calculate Euclidean distance between the 3D endpoints
            diameters[i] = np.linalg.norm(point1_3d - point2_3d)

        return diameters

    def calculate_volume_and_length(self, line_segment_coordinates, depth):
        """ Calculate the volume and length of the object. Only returns output if all line segments are valid"""
        if np.any(depth) == np.nan:
            return np.nan

        # Step 1: Calculate diameters along the segments
        diameters = self.calculate_diameter(line_segment_coordinates, depth)

        # Step 2: Initialize total volume, total_length
        total_volume = 0.0
        total_length = 0.0

        # Step 3: Iterate through consecutive line segments to compute truncated cone volumes
        for i in range(len(line_segment_coordinates) - 1):
            # Radii at the current and next segments
            r1 = diameters[i] / 2  # Radius at current segment
            r2 = diameters[i + 1] / 2  # Radius at next segment

            # Get the midpoints of the current and next segments
            x1 = (line_segment_coordinates[i][1] + line_segment_coordinates[i][3]) / 2
            y1 = (line_segment_coordinates[i][0] + line_segment_coordinates[i][2]) / 2
            x2 = (line_segment_coordinates[i + 1][1] + line_segment_coordinates[i + 1][3]) / 2
            y2 = (line_segment_coordinates[i + 1][0] + line_segment_coordinates[i + 1][2]) / 2

            # Undistort the midpoints
            x1_ud, y1_ud = self._undistort_point(x1, y1)
            x2_ud, y2_ud = self._undistort_point(x2, y2)

            # Triangulate the 3D points using depth
            z1 = np.sum(depth[i]) / 2  # Average depth of the current segment
            z2 = np.sum(depth[i + 1]) / 2  # Average depth of the next segment

            x1_3d = (x1_ud - self.cx) * z1 / self.fx
            y1_3d = (y1_ud - self.cy) * z1 / self.fy
            x2_3d = (x2_ud - self.cx) * z2 / self.fx
            y2_3d = (y2_ud - self.cy) * z2 / self.fy

            # 3D coordinates of midpoints
            point1_3d = np.array([x1_3d, y1_3d, z1])
            point2_3d = np.array([x2_3d, y2_3d, z2])

            # Calculate Euclidean distance between the 3D midpoints (height)
            h = np.linalg.norm(point1_3d - point2_3d)

            # Volume of the truncated cone
            volume = (1 / 3) * np.pi * h * (r1 ** 2 + r1 * r2 + r2 ** 2)

            # Add to total volume
            total_volume += volume
            total_length += h

        return total_volume, total_length

    def _identify_key_points(self, skeleton_map):
        """ Identify endpoints and intersections in a skeleton map. """
        padded_img = np.zeros((skeleton_map.shape[0] + 2, skeleton_map.shape[1] + 2), dtype=np.uint8)
        padded_img[1:-1, 1:-1] = skeleton_map
        res = cv2.filter2D(src=padded_img, ddepth=-1,
                           kernel=np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint8))
        raw_endpoints = np.argwhere(res == 11) - 1  # To compensate for padding
        raw_intersections = np.argwhere(res > 12) - 1  # To compensate for padding

        # Consolidate adjacent intersections
        refined_intersections = self._remove_adjacent_intersections(raw_intersections, eps=5, min_samples=1)

        return np.array(raw_endpoints), np.array(refined_intersections)

    @staticmethod
    def _remove_adjacent_intersections(intersections, eps=5, min_samples=1):
        """ Remove adjacent intersections by clustering. """
        if len(intersections) == 0:
            return []

            # Convert intersections to numpy array
        intersections_array = np.array(intersections)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections_array)

        # Extract cluster labels
        labels = clustering.labels_

        # Consolidate intersections by taking the mean of each cluster
        consolidated_intersections = []
        for label in set(labels):
            cluster_points = intersections_array[labels == label]
            consolidated_point = cluster_points[0]
            consolidated_intersections.append(tuple(consolidated_point))

        return consolidated_intersections

    @staticmethod
    def _prune_short_branches(skeleton, endpoints, intersections, threshold=10):
        """ Prune short branches(< threshold) from the skeleton."""
        pruned_skeleton = skeleton.copy()
        endpoints_set = set(map(tuple, endpoints))
        intersections_set = set(map(tuple, intersections))

        def bfs_until_max_distance(start):
            """BFS that stops when path length exceeds threshold or reaches an intersection."""
            queue = deque([(start, 0)])  # (current_point, current_distance)
            visited = {start}
            path = []

            while queue:
                (y, x), dist = queue.popleft()
                path.append((y, x))

                # Stop if distance exceeds threshold (retain branch)
                if dist > threshold:
                    return [], dist

                # Stop if an intersection or another endpoint is reached (short branch identified)
                if (y, x) in intersections_set or ((y, x) in endpoints_set and dist > 0):
                    path.pop()  # Exclude the intersection or endpoint itself from the path
                    return path, dist

                # Explore 8-connected neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y + dy, x + dx
                    if (ny, nx) not in visited and 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if pruned_skeleton[ny, nx]:  # Only continue on skeleton pixels
                            visited.add((ny, nx))
                            queue.append(((ny, nx), dist + 1))

            # Return empty path if no intersection is found within the threshold distance
            return [], float('inf')

        for endpoint in endpoints_set:
            # Run the search to find short branches
            path_instance, path_length = bfs_until_max_distance(endpoint)

            # Prune only if the branch is below the threshold distance
            if path_instance and path_length <= threshold:
                for y, x in path_instance:
                    pruned_skeleton[y, x] = 0  # Remove branch pixels

        return pruned_skeleton

    @staticmethod
    def _preserve_skeleton_path(skeleton, endpoints):
        """ Preserve a single continuous path between two endpoints. """
        # Sort endpoints by the x-coordinate
        endpoints = sorted(endpoints, key=lambda x: x[0])

        max_y_separation = -1
        best_pair = None
        best_path = None

        # Iterate over each pair of endpoints
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                start_point = endpoints[i]
                end_point = endpoints[j]

                # Create a cost array where all skeleton pixels are 1, and non-skeleton pixels are infinity
                cost_array = np.where(skeleton, 1, np.inf)

                try:
                    # Attempt to find a path from start_point to end_point
                    path, _ = graph.route_through_array(cost_array, start_point, end_point, fully_connected=True)

                    # Calculate the vertical separation
                    y_separation = abs(start_point[0] - end_point[0])

                    # Update the best pair if this separation is the largest found
                    if y_separation > max_y_separation:
                        max_y_separation = y_separation
                        best_pair = (start_point, end_point)
                        best_path = path

                except ValueError:
                    # No valid path exists between these points, skip
                    continue

        # Create a new skeleton with only the path retained
        path_skeleton = np.zeros_like(skeleton, dtype=bool)
        if best_path is not None:
            for y, x in best_path:
                path_skeleton[y, x] = True

        return best_path, path_skeleton

    def _find_closest_symmetry_axis(self, binary_object):
        """ Find the closest axis of symmetry for a binary object by evaluating both principal axes. """
        # Step 1: Compute the centroid and principal axes
        coords = np.argwhere(binary_object)
        pca = PCA(n_components=2)
        pca.fit(coords)
        centroid = pca.mean_
        principal_axes = pca.components_  # First and second principal directions

        best_score = float('inf')
        best_axis = None

        # Step 2: Iterate over both principal axes
        for axis_direction in principal_axes:
            # Calculate the symmetry score for the axis
            score = self._split_and_reflect_score(binary_object, centroid, axis_direction)

            # Track the best axis
            if score < best_score:
                best_score = score
                best_axis = axis_direction

        return best_axis, centroid

    def _split_and_reflect_score(self, binary_object, centroid, direction):
        """ Compute the symmetry score """
        coords = np.argwhere(binary_object)
        norm_dir = direction / np.linalg.norm(direction)  # Normalize the direction vector

        # Define the splitting line
        normal_vector = np.array([-norm_dir[1], norm_dir[0]])  # Normal to the axis
        projected_distances = np.dot(coords - centroid, normal_vector)  # Distance from the splitting line

        # Split the object into two parts
        part1_coords = coords[projected_distances > 0]  # One side of the axis
        part2_coords = coords[projected_distances <= 0]  # Other side of the axis

        # Create binary masks for the two parts
        part1_mask = np.zeros_like(binary_object)
        part2_mask = np.zeros_like(binary_object)
        part1_mask[part1_coords[:, 0], part1_coords[:, 1]] = 1
        part2_mask[part2_coords[:, 0], part2_coords[:, 1]] = 1

        # Reflect part 1 and combine it with the original part 1
        reflected_part1 = self._reflect_object(part1_mask, centroid, direction)
        combined_part1 = part1_mask | reflected_part1  # Combine mask 1 with its reflection

        # Reflect part 2 and combine it with the original part 2
        reflected_part2 = self._reflect_object(part2_mask, centroid, direction)
        combined_part2 = part2_mask | reflected_part2  # Combine mask 2 with its reflection

        # Calculate symmetry scores
        score_part1 = np.sum(np.abs(combined_part1 - binary_object))  # Part 1 score
        score_part2 = np.sum(np.abs(combined_part2 - binary_object))  # Part 2 score

        # Combine the scores
        total_score = score_part1 + score_part2

        return total_score

    @staticmethod
    def _reflect_object(binary_object, centroid, direction):
        """ Reflect a binary object across a given axis defined by a centroid and direction.  """
        coords = np.argwhere(binary_object)  # Get foreground coordinates
        norm_dir = direction / np.linalg.norm(direction)  # Normalize the direction vector

        # Project each coordinate onto the reflection axis
        coords_centered = coords - centroid  # Center the coordinates at the centroid
        projection = np.dot(coords_centered, norm_dir)[:, None] * norm_dir  # Projection onto axis
        reflected_coords = coords_centered - 2 * (coords_centered - projection)  # Reflect across the axis
        reflected_coords += centroid  # Translate back to the original position

        # Round and cast to integer for binary image
        reflected_coords = np.round(reflected_coords).astype(int)

        # Create a binary mask for the reflected object
        reflected_object = np.zeros_like(binary_object)
        for coord in reflected_coords:
            if 0 <= coord[0] < binary_object.shape[0] and 0 <= coord[1] < binary_object.shape[1]:
                reflected_object[coord[0], coord[1]] = 1

        return reflected_object

    def _undistort_point(self, x, y):
        """Compensate for lens distortion using the camera's intrinsic parameters."""

        # Normalize coordinates
        x_norm = (x - self.cx) / self.fx
        y_norm = (y - self.cy) / self.fy

        # Radial distortion
        r2 = x_norm ** 2 + y_norm ** 2
        radial_dist = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3

        # Tangential distortion
        x_dist = x_norm * radial_dist + 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm ** 2)
        y_dist = y_norm * radial_dist + self.p1 * (r2 + 2 * y_norm ** 2) + 2 * self.p2 * x_norm * y_norm

        # Return undistorted pixel coordinates
        x_undistorted = x_dist * self.fx + self.cx
        y_undistorted = y_dist * self.fy + self.cy
        return x_undistorted, y_undistorted
    

class StemInstance:
    def __init__(self, keypoints, line_segment_coordinates, diameters, processed_mask, overlay_text):
        self.keypoints = keypoints
        self.line_segment_coordinates = line_segment_coordinates
        self.diameters = diameters
        self.processed_mask = processed_mask
        self.overlay_text = overlay_text
