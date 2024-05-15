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

        # Reorder skeleton coordinates
        if self.skeleton_coordinates[0][0] < self.skeleton_coordinates[-1][0]:
            self.skeleton_coordinates = np.flip(self.skeleton_coordinates, axis=0)

        # Identify key pixels
        end_pixels, intersections = self._identify_key_pixels(self.skeleton_binary_map)

        # Prune split ends
        # self.skeleton_coordinates_pruned = self._prune_split_ends(self.skeleton_coordinates, end_pixels, intersections, self.threshold)
        self.skeleton_coordinates_pruned = self.skeleton_coordinates
        # Reconstruct binary map
        self.skeleton_binary_map_pruned = self._binary_image_from_skeleton()

        # Compute slope and pixel coordinates of perpendicular line segments
        slope = self._calculate_perpendicular_slope()
        self.line_segment_coordinates = self._calculate_line_segment_coordinates(slope)

        # Visualize perpendicular line segments
        if visualize:
            self._visualize_line_segments()

    def _visualize_line_segments(self):
        plt.figure()
        plt.imshow(self.mask_binary_processed, cmap=plt.get_cmap('Spectral'))
        for line in self.line_segment_coordinates:
            plt.plot([line[1], line[3]], [line[0], line[2]], 'r')
        plt.show()
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

    def run(self, save_images=True):

        # Loop through each label
            # Identify key pixels
        for label in range(1, self.num_labels + 1):
            # Get skeleton pixels of label
            skel_pixels = np.argwhere(self.image_labels == label)

            if save_images:
                # Current label
                plt.figure()
                plt.imshow(self.image_labels == label, cmap=plt.get_cmap('Spectral'))
                plt.show()
                plt.savefig('current_label.png')

            # Identify key pixels
            end_pixels, intersections = self._identify_key_pixels(self.image_labels == label)

            # Prune split ends
            skel_pixels_pruned = self._prune_split_ends(skel_pixels, end_pixels, intersections, self.threshold)

            # Reidentify key pixels
            end_pixels_pruned, intersections_ = self._identify_key_pixels(
                self._binary_image_from_skeleton(skel_pixels_pruned, self.image_labels))

            if save_images:
                # pruned
                plt.figure()
                plt.imshow(self._binary_image_from_skeleton(skel_pixels_pruned, self.image_labels),
                                                            cmap=plt.get_cmap('Spectral'))
                plt.show()
                plt.savefig('pruned.png')

            # Remove redundant intersections and identify number of intersections
            skel_pixels_new, intersections_pruned, num_intersections = self._refine_intersections(skel_pixels_pruned, intersections_)

            #
            # NEW_LABELS = self._binary_image_from_skeleton(skel_pixels_new, self.image_labels)
            # Reassign labels
            label_map, num_labels_assigned = self._reassign_labels(self.image_labels == label, intersections_pruned, end_pixels_pruned, skel_pixels_pruned)

            # Save image

    def _refine_intersections(self, skeleton, intersections):
        unique_intersections = []
        intersections_np = np.array(intersections)
        deleted_intersections = np.copy(intersections_np)
        clusterer = DBSCAN(eps=5, min_samples=2)
        clusterer.fit(intersections_np)

        labels = clusterer.labels_
        n_features_in_ = clusterer.n_features_in_

        # Unique indices
        _, indices = np.unique(labels, return_index=True)
        for i in indices:
            unique_intersections.append(intersections[i])
            deleted_intersections = np.delete(deleted_intersections, i, axis=0)

        # Updated skeleton
        for i in deleted_intersections:
            idx = np.where(np.all(skeleton == i, axis=1))[0]
            skeleton = np.delete(skeleton, idx, axis=0)

        # Number of intersections
        num_intersections = len(unique_intersections)

        return skeleton, np.array(unique_intersections), num_intersections



    def _binary_image_from_skeleton(self):
        binary_image = np.zeros_like(self.skeleton_binary_map, dtype=int)

        for y, x in self.skeleton_coordinates_pruned:
            binary_image[y, x] = 1

        return binary_image

    # def _remove_redundant_intersection(self, intersection_pixels):
    #     reduced_intersections = []
    #     index = 1
    #     for i, j in intersection_pixels:
    #         if index == 1:
    #             reduced_intersections.append([i, j])
    #             index += 1
    #             continue
    #         append = True
    #         for m, n in reduced_intersections:
    #             if (i == m and j == n) or (abs(i - m) <= 1 or abs(j - n) <= 1):
    #                 append = False
    #         if append:
    #             reduced_intersections.append([i, j])
    #         index += 1
    #
    #     return reduced_intersections

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

    def _reassign_labels_v2(self, label_map, intersection_pixels_reduced, end_pixels, skeleton_pixels):

        # Create a new label map
        reassigned_label_map = np.zeros_like(label_map, dtype=int, order='C')
        intersection_pixels = [array.tolist() for array in intersection_pixels]

        # Cluster intersection pixels
        clusterer = DBSCAN(eps=self.epsilon, min_samples=2)
        clusterer.fit(intersection_pixels_reduced)
        num_clusters = clusterer.n_features_in_

        # Create segments
        seg_counter = 0
        segments = {}

        for i, j in intersection_pixels_reduced:
            neighbor_idx = np.argwhere(np.all((skeleton_pixels >= [i - 1, j - 1]) &
                                              (skeleton_pixels <= [i + 1, j + 1]), axis=1))

            # Delete self
            index = np.where(np.all(skeleton_pixels == [i, j], axis=1))[0]
            self_index = np.where(neighbor_idx == index)[0]
            neighbor_idx = np.delete(neighbor_idx, self_index, axis=0)
            # np.delete(neighbor_idx, 1)
            
            # Intersection pixel has three neighbors
            for idx in neighbor_idx:
                seg_counter += 1
                segments[seg_counter] = []
                segments[seg_counter].append(skeleton_pixels[idx][0].tolist())

                # if not np.array_equal( skeleton_pixels[idx][0], np.array([i, j]) ):
                #     seg_counter += 1
                #     segments[seg_counter] = []
                #     segments[seg_counter].append(skeleton_pixels[idx][0])

                stop_condition = False
                center_pixel = segments[seg_counter][0]
                while not stop_condition:
                    segment_neighbor_idx = np.argwhere(
                        np.all((skeleton_pixels >= [center_pixel[0] - 1, center_pixel[1] - 1]) &
                               (skeleton_pixels <= [center_pixel[0] + 1, center_pixel[1] + 1]), axis=1))

                    # Delete self
                    index = np.where(np.all(skeleton_pixels == [center_pixel[0], center_pixel[1]], axis=1))[0]
                    self_index = np.where(segment_neighbor_idx == index)[0]
                    segment_neighbor_idx = np.delete(segment_neighbor_idx, self_index, axis=0)

                    # Add only new, nonsignificant pixels that has minimum distance
                    new_neighbor = 0
                    reached_end = False
                    distance = []
                    for count, cur_segment_idx in enumerate(segment_neighbor_idx):
                        distance[count] = np.linalg.norm(skeleton_pixels[cur_segment_idx][0] - center_pixel)
                    to_add_idx = np.argmin(distance)
                    segment_idx = segment_neighbor_idx[to_add_idx]

                    if skeleton_pixels[segment_idx][0].tolist() not in segments[seg_counter]:
                        new_neighbor += 1
                        segments[seg_counter].append(skeleton_pixels[segment_idx][0].tolist())
                        center_pixel = skeleton_pixels[segment_idx][0]
                    if skeleton_pixels[segment_idx][0].tolist() in end_pixels.tolist():
                        reached_end = True
                    if skeleton_pixels[segment_idx][0].tolist() not in intersection_pixels_reduced:
                        reached_intersection = True

                    # if skeleton_pixels[segment_idx][0].tolist() not in intersection_pixels_reduced:
                    #     if skeleton_pixels[segment_idx][0].tolist() not in segments[seg_counter]:
                    #         new_neighbor += 1
                    #         segments[seg_counter].append(skeleton_pixels[segment_idx][0].tolist())
                    #         center_pixel = skeleton_pixels[segment_idx][0]
                    #     if skeleton_pixels[segment_idx][0].tolist() in end_pixels.tolist():
                    #         reached_end = True

                    if new_neighbor == 0 or reached_end == True or reached_intersection == True:
                        stop_condition = True

        # Find redundant elements
        all_indices = []
        redundant_indices = []
        for k in range(1, seg_counter + 1):
            all_indices.append(k)

        # Find all pairs
        pair_indices = [(a, b) for idx, a in enumerate(all_indices) for b in all_indices[idx + 1:]]

        for first, second in pair_indices:
            if sorted(segments[first]) == sorted(segments[second]):
                redundant_indices.append(first)
                redundant_indices.append(second)
                break

        # Delete redundant segment
        del segments[redundant_indices[0]]
        all_indices.remove(redundant_indices[0])

        # Find slope of each segment
        slope = {}
        slope_indices = []
        for seg in all_indices:
            if seg != redundant_indices[1]:
                slope_indices.append(seg)
                pixel1 = segments[seg][0]
                pixel2 = segments[seg][-1]
                if pixel1[0] > pixel2[0]:
                    temp = pixel2
                    pixel2 = pixel1
                    pixel1 = temp
                slope[seg] = (pixel2[1] - pixel1[1]) / (pixel2[0] - pixel1[0])

        # Assign labels
        num_labels = len(slope) / 2
        reassigned_labels = {}
        current_label = 1

        # Assign label to common segment
        reassigned_labels[redundant_indices[1]] = current_label

        # Assign label to intersection
        for int_y, int_x in intersection_pixels:
            reassigned_label_map[int_y][int_x] = current_label

        # Assign labels to other segments
        for seg in slope_indices:
            if seg in reassigned_labels:
                continue
            else:
                reassigned_labels[seg] = current_label
                closest_slope_index = find_closest_slope(seg, slope)
                reassigned_labels[closest_slope_index] = current_label
                current_label += 1
                if current_label > num_labels:
                    break

        # Reassigned label map
        for seg in segments.keys():
            label = reassigned_labels[seg]
            current_seg = np.array(segments[seg])
            for m in range(len(current_seg)):
                reassigned_label_map[current_seg[m, 0]][current_seg[m, 1]] = label

        return reassigned_label_map, 
    
    def _reassign_labels(self, label_map, intersection_pixels_reduced, end_pixels, skeleton_pixels):

        # Cluster intersection pixels
        clusterer = DBSCAN(eps=5, min_samples=2)
        clusterer.fit(intersection_pixels_reduced)
        num_clusters = clusterer.n_features_in_


        # Create a new label map
        reassigned_label_map = np.zeros_like(label_map, dtype=int, order='C')
        intersection_pixels_reduced = [array.tolist() for array in intersection_pixels_reduced]

        # Create three segments        new_neighbor = 0
        seg_counter = 0
        segments = {}

        for i, j in intersection_pixels_reduced:
            neighbor_idx = np.argwhere(np.all((skeleton_pixels >= [i - 1, j - 1]) &
                                              (skeleton_pixels <= [i + 1, j + 1]), axis=1))

            # Delete self
            index = np.where(np.all(skeleton_pixels == [i, j], axis=1))[0]
            self_index = np.where(neighbor_idx == index)[0]
            neighbor_idx = np.delete(neighbor_idx, self_index, axis=0)
            # np.delete(neighbor_idx, 1)

            for idx in neighbor_idx:
                seg_counter += 1
                segments[seg_counter] = []
                segments[seg_counter].append(skeleton_pixels[idx][0].tolist())

                # if not np.array_equal( skeleton_pixels[idx][0], np.array([i, j]) ):
                #     seg_counter += 1
                #     segments[seg_counter] = []
                #     segments[seg_counter].append(skeleton_pixels[idx][0])

                stop_condition = False
                center_pixel = segments[seg_counter][0]
                while not stop_condition:
                    segment_neighbor_idx = np.argwhere(
                        np.all((skeleton_pixels >= [center_pixel[0] - 1, center_pixel[1] - 1]) &
                               (skeleton_pixels <= [center_pixel[0] + 1, center_pixel[1] + 1]), axis=1))

                    # Delete self
                    index = np.where(np.all(skeleton_pixels == [center_pixel[0], center_pixel[1]], axis=1))[0]
                    self_index = np.where(segment_neighbor_idx == index)[0]
                    segment_neighbor_idx = np.delete(segment_neighbor_idx, self_index, axis=0)

                    # gnifpixelsicant
                    # Add only new, nonsignificant pixels that has minimum distance
                    new_neighbor = 0
                    reached_end = False
                    reached_intersection = False
                    distance = np.zeros(len(segment_neighbor_idx))
                    for count, cur_segment_idx in enumerate(segment_neighbor_idx):
                        distance[count] = np.linalg.norm(skeleton_pixels[cur_segment_idx][0] - center_pixel)
                    to_add_idx = np.argmin(distance)
                    segment_idx = segment_neighbor_idx[to_add_idx]

                    if skeleton_pixels[segment_idx][0].tolist() not in segments[seg_counter]:
                        new_neighbor += 1
                        segments[seg_counter].append(skeleton_pixels[segment_idx][0].tolist())
                        center_pixel = skeleton_pixels[segment_idx][0]
                    if skeleton_pixels[segment_idx][0].tolist() in end_pixels.tolist():
                        reached_end = True
                    if skeleton_pixels[segment_idx][0].tolist() in intersection_pixels_reduced:
                        reached_intersection = True

                    if new_neighbor == 0 or reached_end or reached_intersection:
                        stop_condition = True
        # Find redundant elements
        all_indices = []
        redundant_indices = []
        for k in range(1, seg_counter + 1):
            all_indices.append(k)

        # Find all pairs
        pair_indices = [(a, b) for idx, a in enumerate(all_indices) for b in all_indices[idx + 1:]]

        for first, second in pair_indices:
            if sorted(segments[first]) == sorted(segments[second]):
                redundant_indices.append(first)
                redundant_indices.append(second)
                break

        # Delete redundant segment
        del segments[redundant_indices[0]]
        all_indices.remove(redundant_indices[0])

        # Find slope of each segment
        slope = {}
        slope_indices = []
        for seg in all_indices:
            if seg != redundant_indices[1]:
                slope_indices.append(seg)
                pixel1 = segments[seg][0]
                pixel2 = segments[seg][-1]
                if pixel1[0] > pixel2[0]:
                    temp = pixel2
                    pixel2 = pixel1
                    pixel1 = temp
                slope[seg] = (pixel2[1] - pixel1[1]) / (pixel2[0] - pixel1[0])

        # Assign labels
        num_labels = len(slope) / 2
        reassigned_labels = {}
        current_label = 1

        # Assign label to common segment
        reassigned_labels[redundant_indices[1]] = current_label

        # Assign label to intersection
        for int_y, int_x in intersection_pixels:
            reassigned_label_map[int_y][int_x] = current_label

        # Assign labels to other segments
        for seg in slope_indices:
            if seg in reassigned_labels:
                continue
            else:
                reassigned_labels[seg] = current_label
                closest_slope_index = find_closest_slope(seg, slope)
                reassigned_labels[closest_slope_index] = current_label
                current_label += 1
                if current_label > num_labels:
                    break

        # Reassigned label map
        for seg in segments.keys():
            label = reassigned_labels[seg]
            current_seg = np.array(segments[seg])
            for m in range(len(current_seg)):
                reassigned_label_map[current_seg[m, 0]][current_seg[m, 1]] = label

        return reassigned_label_map, num_labels_assigned

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

    @jit(debug=True)
    def _prune_split_ends(self, skeleton, end_pixels, intersections, threshold):
        # Find indices of intersections
        intersection_idx = []
        for j, intersection in enumerate(intersections):
            idx = np.argwhere((skeleton[:, 0] == intersection[0]) & (skeleton[:, 1] == intersection[1]))
            intersection_idx.append(np.int(idx))

        # Define list of indices to prune
        delete_indices = []

        for i, coordinates in enumerate(end_pixels):

            # Find index of current end pixel
            coordinate_idx = np.argwhere((skeleton[:, 0] == coordinates[0]) & (skeleton[:, 1] == coordinates[1]))

            # Add end pixel index to prune skeleton indices
            to_prune_skeleton_idx = [np.int(coordinate_idx)]
            # Initialize first center pixel to be same as endpoint
            center_pixel = coordinates
            while not list(set(to_prune_skeleton_idx).intersection(intersection_idx)):
                # Identify neighbor indices
                neighbors_idx = self._find_neighbor_idx(center_pixel, skeleton)

                # Exclude self
                index = np.where(np.all(skeleton == center_pixel, axis=1))[0]
                self_index = np.where(neighbors_idx == index)[0]
                neighbors_idx = np.delete(neighbors_idx, self_index, axis=0)

                # Append neighbor indices
                new_neighbor = 0
                for j in range(len(neighbors_idx)):
                    if np.int(neighbors_idx[j]) not in to_prune_skeleton_idx:
                        new_neighbor += 1
                        to_prune_skeleton_idx.append(np.int(neighbors_idx[j]))

                if new_neighbor == 1:  # If only 1 neighbor
                    center_pixel = skeleton[to_prune_skeleton_idx[-1]]
                # else: # If 2 neighbors
                #     first_neighbors = find_neighbor_idx(skeleton[to_prune_skeleton_idx[-1]], skeleton)
                #     new_neighbor = 0
                #     for j in range(len(first_neighbors)):
                #         if np.int(first_neighbors[j]) not in to_prune_skeleton_idx:
                #             new_neighbor += 1
                #
                #     if new_neighbor > 0:
                #         center_pixel = skeleton[to_prune_skeleton_idx[-1]]
                #     else:
                #         center_pixel = skeleton[to_prune_skeleton_idx[-1]]

            to_prune_skeleton_idx.pop()  # Get rid of intersection

            if len(to_prune_skeleton_idx) < threshold:
                # skeleton = np.delete(skeleton, to_prune_skeleton_idx, axis=0)
                delete_indices.extend(to_prune_skeleton_idx)

        # Delete indices
        skeleton = np.delete(skeleton, delete_indices, axis=0)

        return skeleton

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

    def _calculate_perpendicular_slope(self):
        # Input:
        # skeleton_coordinates: the pixel coordinates of the skeleton
        # window: the size of the window used to calculate the slope
        # stride: calculate the slope every 'stride' skeleton pixel
        # Output:
        # slope: dictionary containing the slope of the skeleton at each pixel coordinate
        # Calculate the gradient of the skeleton
        slope = {}

        i = self.window
        while self.skeleton_coordinates[i][0] > (1-self.threshold)*self.image.shape[0]:
            key = tuple(self.skeleton_coordinates[i])
            slope[key] = np.arctan2(self.skeleton_coordinates[i][1] - self.skeleton_coordinates[i - self.window][1],
                                  self.skeleton_coordinates[i][0] - self.skeleton_coordinates[i - self.window][0])
            i += self.stride

        return slope