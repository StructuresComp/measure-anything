import argparse
import cv2
import numpy as np
import os
from stemSkel import StemSkeletonization
from stemSegment import StemSegmentation
from skimage import morphology
import logging as log
from demo_utils import get_click_coordinates, display_with_overlay, scale_points
from clubs_dataset_python.clubs_dataset_tools.common import CalibrationParams
# from clubs_dataset_python.clubs_dataset_tools.point_cloud_generation import depth_to_3d
from clubs_dataset_python.clubs_dataset_tools.common import (CalibrationParams,
                                        convert_depth_uint_to_float)
from clubs_dataset_python.clubs_dataset_tools.image_registration import (register_depth_image)

import pudb

def depth_to_3d(depth, point, intrinsics):
    # Extract focal lengths and principal points from the intrinsics matrix
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Calculate the real-world coordinates
    x = (point[0] - cx) * depth / fy
    y = (point[1] - cy) * depth / fx
    z = depth

    # Return 3D point coordinates
    return [x, y, z]

def main():
    log.basicConfig(level=log.DEBUG)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image file (png/jpg)')
    parser.add_argument('--depth_image', type=str, required=True, help='Path to the corresponding depth image (png)')
    parser.add_argument('--calib_file', type=str, required=True, help='Path to the camera calibration YAML file')
    parser.add_argument('--sensor', type=str, required=True, help='Sensor type (e.g., primesense, d415, d435)')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    parser.add_argument('--measurement_threshold', type=float,
                        help='Threshold ratio. eg. 0.1 calculates line segments within bottom 1/10 of the image')
    parser.add_argument('--use_stereo_depth', action='store_true', help='Use stereo depth for depth registration')
    args = parser.parse_args()
    directory_name = os.path.splitext(os.path.basename(args.input_image))[0]

    # Initialize SAM model
    stemSegObj = StemSegmentation()

    # Load input image
    image_ocv = cv2.imread(args.input_image)
    if image_ocv is None:
        print("Error loading image file")
        return
    image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2RGB)

    # Load depth image
    depth_image = cv2.imread(args.depth_image, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print("Error loading depth image file")
        return

    # Load calibration parameters based on sensor type
    calib_params = CalibrationParams()
    if args.sensor == 'd415':
        calib_params.read_from_yaml('clubs_dataset_python/config/realsense_d415_device_depth.yaml')
        depth_x_offset_pixels, depth_y_offset_pixels = 4, 0
    elif args.sensor == 'd435':
        calib_params.read_from_yaml('clubs_dataset_python/config/realsense_d435_device_depth.yaml')
        depth_x_offset_pixels, depth_y_offset_pixels = -3, -1
    else:  # Default to d315 if unspecified
        calib_params.read_from_yaml('clubs_dataset_python/config/primesense.yaml')
        depth_x_offset_pixels, depth_y_offset_pixels = 0, 0

    # Apply depth intrinsics offsets
    calib_params.depth_intrinsics[0, 2] += depth_x_offset_pixels
    calib_params.depth_intrinsics[1, 2] += depth_y_offset_pixels
    calib_params.depth_scale = calib_params.depth_scale * 0.3

    # Resize the image for display
    display_width, display_height = 1600, 900
    resized_image = cv2.resize(image_rgb, (display_width, display_height))

    # Calculate scale factors for x and y
    scale_x = image_rgb.shape[1] / display_width
    scale_y = image_rgb.shape[0] / display_height

    # Set up prompt data
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}

    # Main loop for interactive point selection
    while True:
        # Display frame with basic instructions
        instructions = ["Press 'r' to reset, 'c' to continue, 'q' to quit"]
        display_with_overlay(resized_image, [], [], [], display_dimensions=[display_width, display_height],
                             diameters=None, save=False, save_name="", mask=None, overlay_text=instructions)

        # Wait for key input
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit the loop
            break
        elif key == ord('r'):  # Reset the image
            prompt_data['positive_points'].clear()
            prompt_data['negative_points'].clear()
            prompt_data['clicked'] = False
            continue
        elif key == ord('c'):  # Continue to select points

            # Set mouse callback for collecting points
            cv2.setMouseCallback("Video Feed", get_click_coordinates, param=prompt_data)

            # Wait for user to select points and press 'c' to continue
            while True:
                key = cv2.waitKey(1)

                # Detailed instructions
                detailed_instructions = [
                    "'Left-click' to add positive point",
                    "'Ctrl + Left-click' to add negative point",
                    "Press 'c' to continue"
                ]

                # Display with current positive and negative points
                display_with_overlay(resized_image,
                                     prompt_data['positive_points'],
                                     prompt_data['negative_points'],
                                     [],
                                     display_dimensions=[display_width, display_height],
                                     diameters=None,
                                     save=False,
                                     save_name="",
                                     mask=None,
                                     overlay_text=detailed_instructions)

                if key == ord('c'):  # Continue once points are collected
                    break

            # Remove mouse callback
            cv2.setMouseCallback("Video Feed", lambda *unused: None)

            # Scale up the prompts to the original image dimensions
            positive_prompts = scale_points(prompt_data['positive_points'], scale_x, scale_y)
            negative_prompts = scale_points(prompt_data['negative_points'], scale_x, scale_y) if prompt_data[
                'negative_points'] else None

            # Run SAM segmentation & save initial mask
            mask = stemSegObj.inference(image_rgb, positive_prompts, negative_prompts)

            if not os.path.exists(f"./output/{directory_name}"):
                os.makedirs(f"./output/{directory_name}")

            cv2.imwrite(f"./output/{directory_name}/initial_mask.png",
                        (mask * 255).astype(np.uint8))

            # Skeletonize, identify perpendicular line segments
            stride = args.stride if args.stride else 10
            threshold = args.measurement_threshold if args.measurement_threshold else 1
            current_stem = StemSkeletonization(threshold=threshold, window=25, stride=stride,
                                               image_file=f'./output/{directory_name}/initial_mask.png')

            # Load image, preprocess and save the processed mask
            current_stem.load_image()
            current_stem.preprocess()
            processed_mask = current_stem.processed_binary_mask

            cv2.imwrite(f"./output/{directory_name}/processed_mask.png",
                        current_stem.processed_binary_mask_0_255)

            # Skeletonize mask and prune
            current_stem.skeletonize_and_prune()

            # Compute slope and pixel coordinates of perpendicular line segments
            current_stem.calculate_perpendicular_slope()
            line_segment_coordinates = current_stem.calculate_line_segment_coordinates()

            pixel_diameters_3d = []
            for coord_pair in line_segment_coordinates:
                point_1 = [coord_pair[1], coord_pair[0]]
                point_2 = [coord_pair[3], coord_pair[2]]
                

                # Extract depth values for each point
                depth_1 = depth_image[point_1[1], point_1[0]] / 1000  # Convert depth to meters
                depth_2 = depth_image[point_2[1], point_2[0]] / 1000 

                # Debug log to verify depth values
                log.debug(f"Depth at point 1: {depth_1} meters, Depth at point 2: {depth_2} meters")

                # If depth is not available or is black (0), use the depth at the center of the image
                if depth_1 == 0:
                    depth_1 = depth_image[(point_1[1] + point_2[1]) // 2, (point_1[0] + point_2[0]) // 2] / 1000 
                if depth_2 == 0:
                    depth_2 = depth_image[(point_1[1] + point_2[1]) // 2, (point_1[0] + point_2[0]) // 2] / 1000 

                # If depth is still not available, skip this pair
                if depth_1 == 0 or depth_2 == 0:
                    pixel_diameters_3d.append(np.nan)
                    continue

                # Convert pixel coordinates to 3D coordinates using depth and intrinsics
                point_3d_1 = depth_to_3d(depth_1, point_1, calib_params.depth_intrinsics)
                point_3d_2 = depth_to_3d(depth_2, point_2, calib_params.depth_intrinsics)

                # Calculate the Euclidean distance between the 3D points in centimeters
                diameter_3d = np.linalg.norm(np.array(point_3d_1) - np.array(point_3d_2)) * 100 * calib_params.depth_scale
                pixel_diameters_3d.append(diameter_3d)

            # Save 3D diameters
            np.save(f"./output/{directory_name}/pixel_diameters_3d.npy", pixel_diameters_3d)

            # Display overlay with segmentation and wait for 'c' to continue
            overlay_text = ["Press 'r' to reset, 'c' to continue, 'q' to quit"]
            while True:
                display_with_overlay(image_rgb,
                                     [],
                                     [],
                                     line_segment_coordinates,
                                     diameters=pixel_diameters_3d,
                                     display_dimensions=[display_width, display_height],
                                     save=False,
                                     save_name="",
                                     mask=processed_mask,
                                     overlay_text=overlay_text)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    # Save image when continued
                    display_with_overlay(image_rgb,
                                         [],
                                         [],
                                         line_segment_coordinates,
                                         diameters=pixel_diameters_3d,
                                         save=True,
                                         display_dimensions=[display_width, display_height],
                                         save_name=f"./output/{directory_name}/diameter_result_3d.png",
                                         mask=processed_mask,
                                         overlay_text=overlay_text)
                    break
                elif key == ord('r'):  # Reset the image
                    prompt_data['positive_points'].clear()
                    prompt_data['negative_points'].clear()
                    prompt_data['clicked'] = False
                    break

    # Close windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
