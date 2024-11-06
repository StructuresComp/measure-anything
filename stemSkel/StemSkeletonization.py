
import numpy as np
from skimage import io
import cv2
from skimage import measure
from skimage import morphology
from skimage import graph
from skimage.filters import threshold_mean
from collections import deque


class StemSkeletonization:
    def __init__(self, threshold, stride, window=30, image_file=None):
        self.image_file = image_file
        self.window = int(window)  # Steps in Central difference to compute the local slope
        self.stride = int(stride)  # Increment to next line segment
        self.threshold = threshold  # Threshold value for range of line segments.
                                    # e.g. 0.5 identifies line segments in bottom half of image

        self.loaded_grayscale_image = None
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

    def load_image(self):
        # Read grayscale image
        self.loaded_grayscale_image = io.imread(self.image_file, as_gray=True)

        # Convert to binary image
        image_ones = np.ones((self.loaded_grayscale_image.shape[0], self.loaded_grayscale_image.shape[1]))
        image_zeros = np.zeros((self.loaded_grayscale_image.shape[0], self.loaded_grayscale_image.shape[1]))
        image_binary = np.where(self.loaded_grayscale_image < threshold_mean(self.loaded_grayscale_image),
                                image_zeros, image_ones)
        self.initial_binary_mask = image_binary.astype(np.uint8)

    def preprocess(self):
        # Pre-processing: Remove small objects
        mask_in_process = morphology.remove_small_objects(self.initial_binary_mask, min_size=100)

        # Connected Component Analysis
        label_image = measure.label(mask_in_process, background=0)

        # Find the properties of connected components
        props = measure.regionprops(label_image)

        # Find the largest connected component (assuming it's the stem)
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

        # Step 2: Apply Morphological Opening to remove small protrusions
        # Define the kernel size for the morphological operations
        kernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size as needed
        mask_in_process_0_255 = cv2.morphologyEx(mask_in_process_0_255, cv2.MORPH_OPEN, kernel)

        # Step 3: Apply Morphological Closing to fill small holes within the component
        mask_in_process_0_255 = cv2.morphologyEx(mask_in_process_0_255, cv2.MORPH_CLOSE, kernel)

        # Step 4: Find contours of the mask (boundary of the object)
        contours, _ = cv2.findContours(mask_in_process_0_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Approximate the contour to a polygon using approxPolyDP
        epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon as needed for simplification
        polygon = cv2.approxPolyDP(contours[0], epsilon, True)

        # Step 6: Create an empty binary mask to draw the polygon
        height, width = mask_in_process.shape  # Size of the original binary mask
        binary_mask_from_polygon = np.zeros((height, width), dtype=np.uint8)  # Empty mask (all zeros)

        # Step 7: Fill the polygon on the binary mask
        cv2.fillPoly(binary_mask_from_polygon, [polygon], color=255)  # Fill the polygon with white (255)

        # Step 8: Convert the filled mask back to binary (0s and 1s) for internal use
        self.processed_binary_mask = (binary_mask_from_polygon > 0).astype(np.uint8)
        self.processed_binary_mask_0_255 = (self.processed_binary_mask * 255).astype(np.uint8)

    @staticmethod
    def visualize_skeleton(mask, skeleton, filename):
        # skeleton = morphology.medial_axis(mask)
        # Convert binary image to an RGB image
        binary_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Define the color for the skeleton overlay (e.g., red)
        skeleton_color = [0, 0, 255]  # RGB for red

        # Overlay the skeleton
        binary_rgb[skeleton == 1] = skeleton_color

        # Display the image
        cv2.imwrite(f"{filename}.png", binary_rgb)

    def skeletonize_and_prune(self):
        # Step 1: Apply Medial Axis Transform
        self.skeleton, self.skeleton_distance = morphology.medial_axis(self.processed_binary_mask_0_255,
                                                                       return_distance=True)
        self.skeleton_coordinates = np.argwhere(self.skeleton == True)

        # Step 2: Identify endpoints and intersections
        self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        # Step 3: Check if there are only two endpoints and return if so
        if len(self.endpoints) == 2:
            if self.skeleton_coordinates[0][0] < self.skeleton_coordinates[-1][0]:
                self.skeleton_coordinates = self.skeleton_coordinates[::-1]
            return

        # Step 4: Prune short branches and check conditions
        pruned_skeleton = self._prune_short_branches(self.skeleton, self.endpoints,
                                                     self.intersections, np.max(self.skeleton_distance))
        self.endpoints, self.intersections = self._identify_key_points(pruned_skeleton)

        if len(self.endpoints) == 2:
            self.skeleton_coordinates = np.argwhere(pruned_skeleton == True)
            if self.skeleton_coordinates[0][0] < self.skeleton_coordinates[-1][0]:
                self.skeleton_coordinates = self.skeleton_coordinates[::-1]
            self.skeleton = pruned_skeleton

            # self.visualize_skeleton(self.processed_binary_mask_0_255, self.skeleton, "Pruned_skeleton")
            return

        # Step 5: Preserve a single continuous skeleton along path
        # Select two endpoints with the greatest 'y' separation
        start_point = max(self.endpoints, key=lambda x: x[0])  # Endpoint with the highest i value
        end_point = min(self.endpoints, key=lambda x: x[0])  # Endpoint with the lowest i value

        self.skeleton = self._preserve_skeleton_path(pruned_skeleton, start_point, end_point)
        self.skeleton_coordinates = np.argwhere(self.skeleton == True)
        self.endpoints, self.intersections = self._identify_key_points(self.skeleton)

        # self.visualize_skeleton(self.processed_binary_mask_0_255, self.skeleton, "Pruned_skeleton")

        if len(self.endpoints) != 2:
            raise Exception("Number of endpoints of pruned skeleton is not 2")

    @staticmethod
    def _prune_short_branches(skeleton, endpoints, intersections, threshold=10):
        """Prune branches with endpoint-to-intersection paths shorter than a threshold."""
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
    def _preserve_skeleton_path(skeleton, start_point, end_point):
        """ Retain only the main path between start_point and end_point in the skeleton. """
        # Create a cost array where all skeleton pixels are 1, and non-skeleton pixels are infinity
        cost_array = np.where(skeleton, 1, np.inf)

        # Find the path from start_point to end_point
        path, _ = graph.route_through_array(cost_array, start_point, end_point, fully_connected=True)

        # Create a new skeleton with only the path retained
        path_skeleton = np.zeros_like(skeleton, dtype=bool)
        for y, x in path:
            path_skeleton[y, x] = True

        return path_skeleton

    def calculate_perpendicular_slope(self):

        i = self.window  # The first line segment should be at least i = self.window to avoid indexing errors

        while i < len(self.skeleton_coordinates) - self.window:
            # Check if current skeleton coordinate meets the threshold condition
            if self.skeleton_coordinates[i][0] <= (1 - self.threshold) * self.skeleton_coordinates[0][0]:
                break

            # Get current key point
            key = tuple(self.skeleton_coordinates[i])

            # Calculate the slope using points offset by `self.window` on either side
            y1, x1 = self.skeleton_coordinates[i - self.window]
            y2, x2 = self.skeleton_coordinates[i + self.window]
            self.slope[key] = np.arctan2(y2 - y1, x2 - x1)

            # Move to the next point based on stride
            i += self.stride

    def calculate_line_segment_coordinates(self):
        # Get the dimensions of the binary mask
        height, width = self.processed_binary_mask_0_255.shape

        # Create an array to store the coordinates of the line segment endpoints
        line_segment_coordinates = np.zeros((len(self.slope), 4), dtype=int)

        idx = 0
        for key, val in self.slope.items():
            # Get skeleton point coordinates
            y, x = key  # Reverse the order for correct indexing
            # Calculate the direction vector for the perpendicular line (normal direction)
            dx = -np.sin(val)
            dy = np.cos(val)

            # Initialize variables to store the endpoints of the line segment
            x1, y1 = x, y
            x2, y2 = x, y

            # Step outward from the skeleton point until we hit the contour or go out of bounds in both directions
            while (0 <= int(round(y1)) < height and 0 <= int(round(x1)) < width and
                   self.processed_binary_mask_0_255[int(round(y1)), int(round(x1))]):  # Move in one direction
                x1 -= dx
                y1 -= dy
            while (0 <= int(round(y2)) < height and 0 <= int(round(x2)) < width and
                   self.processed_binary_mask_0_255[int(round(y2)), int(round(x2))]):  # Move in the opposite direction
                x2 += dx
                y2 += dy

            # Store the integer coordinates of the endpoints
            line_segment_coordinates[idx] = np.array([int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))])
            idx += 1

        return line_segment_coordinates

    @staticmethod
    def _identify_key_points(skeleton_map):
        padded_img = np.zeros((skeleton_map.shape[0] + 2, skeleton_map.shape[1] + 2), dtype=np.uint8)
        padded_img[1:-1, 1:-1] = skeleton_map
        res = cv2.filter2D(src=padded_img, ddepth=-1,
                           kernel=np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint8))
        endpoints = np.argwhere(res == 11) - 1  # To compensate for padding
        intersections = np.argwhere(res > 12) - 1  # To compensate for padding
        return endpoints, intersections
