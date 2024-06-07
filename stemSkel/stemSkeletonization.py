import numpy as np
from numba import jit
import os
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import measure
from skimage import img_as_uint
from skimage import morphology
from skimage.filters import threshold_mean
from pathlib import Path

class stemSkeletonization:
    def __init__(self, threshold, window, stride, image_file=None):
        self.threshold = threshold
        self.image_file = image_file
        self.image = None
        self.mask_binary = None
        self.mask_binary_processed = None
        self.skeleton_distance = None
        self.skeleton_binary_map = None
        self.skeleton_coordinates = None
        self.skeleton_binary_map_pruned = None
        self.skeleton_coordinates_pruned = None
        self.window = int(window)
        self.stride = int(stride)
        self.line_segment_coordinates = None
    def _load_image(self):
        # Read grayscale image
        self.image = io.imread(self.image_file, as_gray=True)

        # Convert to binary image
        image_ones = np.ones((self.image.shape[0], self.image.shape[1]))
        image_zeros = np.zeros((self.image.shape[0], self.image.shape[1]))
        image_binary = np.where(self.image < threshold_mean(self.image), image_zeros, image_ones)
        self.mask_binary = image_binary.astype(np.uint8)

    def _preprocess(self):
        # Pre-processing
        binary_mask = morphology.remove_small_objects(self.mask_binary, min_size=100)
        # Connected Component Analysis
        label_image = measure.label(binary_mask, background=0)
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
        self.mask_binary_processed = np.zeros_like(binary_mask)
        self.mask_binary_processed[label_image == largest_component.label] = 1

    def _save_processed_mask(self):
        # Save final stem mask
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        io.imsave('./temp/final_stem_mask.png', img_as_uint(self.mask_binary_processed))
        np.save(f'./temp_mask/{Path(self.image_file).stem}.npy', self.mask_binary_processed)

    def generate_line_segment_pairs(self, visualize=True, save_images=True):
        # Load image and preprocess
        self._load_image()
        self._preprocess()

        if save_images:
            # Display the result
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(self.mask_binary, cmap='gray')
            axes[0].set_title('Original Binary Mask')
            axes[1].imshow(self.mask_binary_processed, cmap='gray')
            axes[1].set_title('Segmented Stem Mask')
            plt.savefig('./temp/stem_masks_comparison.png')
            plt.show()
            self._save_processed_mask()

        # Run medial axis skeletonization - only one label remains
        self.skeleton_binary_map, self.skeleton_distance = morphology.medial_axis(self.mask_binary_processed, return_distance=True)
        self.skeleton_coordinates = np.argwhere(self.skeleton_binary_map == True)



        # Identify key pixels
        end_pixels, intersections = self._identify_key_pixels(self.skeleton_binary_map)

        # Prune split ends and reorder pruned skeleton pixels
        self.skeleton_coordinates_pruned = self._prune_split_ends(self.skeleton_coordinates, end_pixels, intersections)

        # self.skeleton_coordinates_pruned = self.skeleton_coordinates
        # Reconstruct binary map
        self.skeleton_binary_map_pruned = self._binary_image_from_skeleton()

        # Compute slope and pixel coordinates of perpendicular line segments
        slope = self._calculate_perpendicular_slope(self.skeleton_coordinates_pruned)
        self.line_segment_coordinates = self._calculate_line_segment_coordinates(slope)

        # Visualize perpendicular line segments
        if visualize:
            self._visualize_line_segments()

    def save_coordinates(self, filename):
        # Save the end coordinates of the line segments
        np.save(f"temp/{filename}.npy", self.line_segment_coordinates)

    def save_mask(self, filename):
        # Save the end coordinates of the line segments
        np.save(f"temp/{filename}.npy", self.line_segment_coordinates)

    def _visualize_line_segments(self):
        plt.figure()
        plt.imshow(self.mask_binary_processed, cmap=plt.get_cmap('Spectral'))
        for line in self.line_segment_coordinates:
            plt.plot([line[1], line[3]], [line[0], line[2]], 'r')
        plt.show()

    def _calculate_perpendicular_slope(self, skeleton_pixels):
        # Input:
        # skeleton_coordinates: the pixel coordinates of the skeleton
        # window: the size of the window used to calculate the slope
        # stride: calculate the slope every 'stride' skeleton pixel
        # Output:
        # slope: dictionary containing the slope of the skeleton at each pixel coordinate
        # Calculate the gradient of the skeleton
        slope = {}

        i = self.window + 10
        while skeleton_pixels[i][0] > (1-self.threshold)*skeleton_pixels[0][0]:
            key = tuple(skeleton_pixels[i])
            slope[key] = np.arctan2(skeleton_pixels[i][1] - skeleton_pixels[i - self.window][1],
                                  skeleton_pixels[i][0] - skeleton_pixels[i - self.window][0])
            i += self.stride

        return slope

    def _calculate_line_segment_coordinates(self, slope):
        # slope is a dictionary where the key is the pixel coordinate and the value is the slope

        line_segment_coordinates = np.zeros(shape=(len(slope), 4), dtype=int)
        idx = 0
        for key, val in slope.items():
            line_segment_coordinates[idx] = np.array([ int(round(key[0] - np.sin(val) * self.skeleton_distance[key[0], key[1]])),
                                            int(round(key[1] - np.cos(val) * self.skeleton_distance[key[0], key[1]])),
                                            int(round(key[0] + np.sin(val) * self.skeleton_distance[key[0], key[1]])),
                                            int(round(key[1] + np.cos(val) * self.skeleton_distance[key[0], key[1]]))])
            idx += 1

        return line_segment_coordinates

    def _binary_image_from_skeleton(self):
        binary_image = np.zeros_like(self.skeleton_binary_map, dtype=int)

        for y, x in self.skeleton_coordinates_pruned:
            binary_image[y, x] = 1

        return binary_image

    def _prune_split_ends(self, skeleton, endpoints, intersections):
        """
                Prune the split ends of a skeleton image.

                Args:
                    skeleton (np.ndarray): N x 2 array of skeleton pixel coordinates.
                    intersections (np.ndarray): N x 2 array of intersection pixel coordinates.
                    endpoints (np.ndarray): N x 2 array of end point pixel coordinates.

                Returns:
                    np.ndarray: Pruned N x 2 array of skeleton pixel coordinates.
                """
        # Convert input coordinates to set of tuples for faster lookup
        skeleton_set = set(map(tuple, skeleton))
        intersection_set = set(map(tuple, intersections))
        endpoint_set = set(map(tuple, endpoints))

        # Initialize a set for the pixels to be removed
        remove_set = set()

        # Function to get 8-connected neighbors
        def get_neighbors(point):
            x, y = point
            neighbors = [
                (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                (x, y - 1), (x, y + 1),
                (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
            ]
            return neighbors

        # Iterate over each end point
        for intersection in intersection_set:
            distances = {}
            for end in endpoint_set:
                distance = np.linalg.norm(np.array(end) - np.array(intersection))
                distances[(tuple(end), tuple(intersection))] = distance

            # Find the two closest endpoint-intersection pairs
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            closest_pairs = sorted_distances[:2]

            for end, _ in closest_pairs:
                # Initialize visited set and queue for BFS
                end = end[0]
                visited = set()
                queue = [end]

                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)

                    # Add current pixel to remove set
                    remove_set.add(current)

                    # If the current pixel is an intersection, stop
                    if current in intersection_set:
                        break

                    # Get the 8-connected neighbors
                    neighbors = get_neighbors(current)

                    for neighbor in neighbors:
                        if neighbor in skeleton_set and neighbor not in visited:
                            queue.append(neighbor)

        # Prune the skeleton by removing the marked pixels
        pruned_skeleton_set = skeleton_set - remove_set

        # Find the remaining endpoints in the pruned skeleton
        remaining_endpoints = endpoint_set - remove_set
        remaining_endpoints = remaining_endpoints | intersection_set

        # Ensure there are exactly two remaining endpoints
        assert len(remaining_endpoints) == 2, "There should be exactly two remaining endpoints."

        # Convert the pruned skeleton set to a list of tuples
        pruned_skeleton_list = list(pruned_skeleton_set)

        # Find the bottom and top endpoints
        bottom_endpoint = min(remaining_endpoints, key=lambda x: -x[0])
        top_endpoint = max(remaining_endpoints, key=lambda x: -x[0])

        # Function to order skeleton starting from a given endpoint
        def order_skeleton(start_point, skeleton_set):
            ordered_skeleton = []
            visited = set()
            stack = [start_point]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                ordered_skeleton.append(current)

                # Get the 8-connected neighbors
                neighbors = get_neighbors(current)

                for neighbor in neighbors:
                    if neighbor in skeleton_set and neighbor not in visited:
                        stack.append(neighbor)

            return ordered_skeleton

        # Order the pruned skeleton starting from the bottom endpoint
        ordered_pruned_skeleton = order_skeleton(bottom_endpoint, pruned_skeleton_set)

        # Ensure the last element is the top endpoint
        # assert ordered_pruned_skeleton[-1] == top_endpoint, "The last element should be the top endpoint."

        # Convert the ordered skeleton list back to an N x 2 array
        ordered_pruned_skeleton_array = np.array(ordered_pruned_skeleton)

        return ordered_pruned_skeleton_array

    @jit(debug=True)
    def _obtain_connectivity_matrix(self, label_map):
        connection = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        connectivity_matrix = np.zeros([label_map.shape[0], label_map.shape[1]])
        for i in range(0, label_map.shape[0]):
            for j in range(0, label_map.shape[1]):
                if i == 0 and j == 0:  # top left corner
                    connectivity_matrix[i, j] = np.sum(np.multiply(label_map[i:i + 2, j:j + 2], connection[1:3, 1:3]))
                elif i == label_map.shape[0] - 1 and j == 0:  # bottom left corner
                    connectivity_matrix[i, j] = np.sum(
                        np.multiply(label_map[i - 1:i + 1, j:j + 2], connection[1:3, 0:2]))
                elif i == 0 and j == label_map.shape[1] - 1:  # top right corner
                    connectivity_matrix[i, j] = np.sum(
                        np.multiply(label_map[i:i + 2, j - 1:j + 1], connection[0:2, 1:3]))
                elif i == label_map.shape[0] - 1 and j == label_map.shape[1] - 1:  # bottom right corner
                    connectivity_matrix[i, j] = np.sum(
                        np.multiply(label_map[i - 1:i + 1, j - 1:j + 1], connection[0:2, 0:2]))
                elif i == 0 and 0 < j < label_map.shape[1] - 1:  # top wall
                    connectivity_matrix[i, j] = np.sum(np.multiply(label_map[i:i + 2, j - 1:j + 2], connection[1:3, :]))
                elif label_map.shape[0] - 1 > i > 0 == j:  # left wall
                    connectivity_matrix[i, j] = np.sum(np.multiply(label_map[i - 1:i + 2, j:j + 2], connection[:, 1:3]))
                elif i == label_map.shape[0] - 1 and 0 < j < label_map.shape[1] - 1:  # bottom wall
                    connectivity_matrix[i, j] = np.sum(
                        np.multiply(label_map[i - 1:i + 1, j - 1:j + 2], connection[0:2, :]))
                elif label_map.shape[0] - 1 > i > 0 and j == label_map.shape[1] - 1:  # right wall
                    connectivity_matrix[i, j] = np.sum(
                        np.multiply(label_map[i - 1:i + 2, j - 1:j + 1], connection[:, 0:2]))
                else:
                    connectivity_matrix[i, j] = np.sum(np.multiply(label_map[i - 1:i + 2, j - 1:j + 2], connection))

        return connectivity_matrix

    def _identify_key_pixels(self, label_map):
        connectivity_matrix = self._obtain_connectivity_matrix(label_map)

        end_pixels = np.argwhere(connectivity_matrix == 11)
        intersections = np.argwhere((connectivity_matrix == 13) | (connectivity_matrix == 14))

        return end_pixels, intersections

    def find_closest_slope(current_seg, slope_dict):
        slope_diff = {}
        for slope_index in slope_dict.keys():
            if slope_index != current_seg:
                slope_diff[slope_index] = np.abs(slope_dict[slope_index] - slope_dict[current_seg])

        result_index = min(slope_diff, key=slope_diff.get)
        return result_index

    @jit(debug=True)
    def _find_neighbor_idx(self, pixel_coordinate, skeleton):
        # neighbors = np.where( (pixel_coordinate[0]-1 <= skeleton[:,0]) & (pixel_coordinate[0]+1 >= skeleton[:,0])
        #            & (pixel_coordinate[1]-1 <= skeleton[:,1]) & (pixel_coordinate[1]-1 <= skeleton[:,1]))
        # exclude self index
        neighbor_idx = np.argwhere(np.all((skeleton >= [pixel_coordinate[0] - 1, pixel_coordinate[1] - 1]) &
                                          (skeleton <= [pixel_coordinate[0] + 1, pixel_coordinate[1] + 1]), axis=1))

        return neighbor_idx

    def get_label_pixels(label_map):
        # Initialize an empty dictionary to store the pixel coordinates of each label
        label_pixels = {}

        # Loop over each unique label in the label map
        for label in np.unique(label_map):
            if label == 0:  # Skip the background label
                continue

            # Find the pixel coordinates of the pixels with the current label
            pixels = np.transpose(np.nonzero(label_map == label))

            # Store the pixel coordinates of the current label in the dictionary
            label_pixels[label] = pixels

        return label_pixels
