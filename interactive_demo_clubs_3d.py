# Description: This script runs the interactive demo of measure anything

import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
from MeasureAnything import MeasureAnything
from demo_utils import get_click_coordinates, display_with_overlay, scale_points
from clubs_dataset_python.clubs_dataset_tools.common import (CalibrationParams,
                                        convert_depth_uint_to_float)
from clubs_dataset_python.clubs_dataset_tools.image_registration import (register_depth_image)

import pudb


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo for Clubs-3D dataset of Measure Anything using SAM-2 with point prompts")
    # parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image file (png/jpg)')
    parser.add_argument('--depth_image', type=str, required=True, help='Path to the corresponding depth image (png)')
    parser.add_argument('--sensor', type=str, required=True, help='Sensor type (e.g., primesense, d415, d435)')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, default=10, help='Stride used to calculate line segments')
    # parser.add_argument('--measurement_threshold', type=float,
    #                     help='Threshold ratio. eg. 0.1 calculates line segments within bottom 1/10 of the image')
    parser.add_argument('--use_stereo_depth', action='store_true', help='Use stereo depth for depth registration')
    args = parser.parse_args()
    directory_name = os.path.splitext(os.path.basename(args.input_image))[0]

    # Initialize command line inputs
    stride = args.stride
    # threshold = args.measurement_threshold if args.measurement_threshold else 0.95

    # Load input image
    image_ocv = cv2.imread(args.input_image)
    if image_ocv is None:
        print("Error loading image file")
        return
    image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2RGB)
    frame_count = 0
    # Load depth image
    image_depth = cv2.imread(args.depth_image, cv2.IMREAD_UNCHANGED)
    if image_depth is None:
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
    resized_depth = cv2.resize(image_depth, (display_width, display_height))

    # Calculate scale factors for x and y
    scale_x = image_rgb.shape[1] / display_width
    scale_y = image_rgb.shape[0] / display_height

    # Set up prompt data
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}

    # Main loop for interactive point selection
    while True:
        # Display frame with basic instructions
        instructions = ["Press 'r' to reset, 'c' to continue, 'q' to quit"]
        display_with_overlay(resized_image, resized_depth, [], [], [], display_dimensions=[display_width, display_height],
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
                                    resized_depth,
                                    prompt_data['positive_points'],
                                    prompt_data['negative_points'],
                                    [],
                                    display_dimensions=[display_width, display_height],
                                    diameters=None,
                                    volume=None,
                                    length=None,
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

            # Create directory to save results
            if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

            # TODO: remove / for debugging purposes only
            # cv2.imwrite(f"rgb_{frame_count}.png", image_rgb)

            # Initialize MeasureAnything object
            object = MeasureAnything(zed=None, window=25, stride=stride, thin_and_long=args.thin_and_long,
                                        image_file=None)
            
            # For non-zed camera, update calibration parameters
            object.update_calibration_params(image_depth, calib_params.depth_intrinsics)
            
            object.detect_mask(image=image_rgb, positive_prompts=positive_prompts,
                                negative_prompts=negative_prompts)

            # Save initial mask
            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/initial_mask.png",
                        (object.initial_binary_mask * 255).astype(np.uint8))

            # Save depth frame as color
            # Normalize if needed (assuming depth_map is already in 8-bit range)
            depth_map_norm = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
            color_depth_map[image_depth == 0] = [0, 0, 0]
            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/depth_map.png", color_depth_map)

            # Preprocess and save
            object.preprocess()
            processed_mask = object.processed_binary_mask
            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask.png",
                        object.processed_binary_mask_0_255)

            # Skeletonize, obtain skeleton_coordinates
            if args.thin_and_long:
                object.skeletonize_and_prune()
                object.augment_skeleton()

            else:
                object.build_skeleton_from_symmetry_axis()

            # Obtain perpendicular line segment coordinates, respective depth
            object.calculate_perpendicular_slope()
            line_segment_coordinates, depth = object.calculate_line_segment_coordinates_and_depth()

            # TODO: remove / for debugging purposes only
            # object.visualize_line_segment_in_order_cv(line_segment_coordinates,
            #                                           image_size=(object.processed_binary_mask_0_255.shape[0],
            #                                                       object.processed_binary_mask_0_255.shape[1]),
            #                                           pause_time=500)

            # Calculate dimensional measurements
            diameters = object.calculate_diameter(line_segment_coordinates, depth)
            volume, length = object.calculate_volume_and_length(line_segment_coordinates, depth)
            length = object.calculate_length(line_segment_coordinates, depth)

            # Save results
            np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters.npy", diameters)
            np.save(f"./output/{directory_name}/results_frame_{frame_count}/volume.npy", volume)
            np.save(f"./output/{directory_name}/results_frame_{frame_count}/length.npy", length)

            # Display overlay with segmentation and wait for 'c' to continue
            overlay_text = [f"Frame:{0}", "Press 'r' to reset, 'c' to continue, 'q' to quit"]
            while True:
                display_with_overlay(image_rgb,
                                        None,
                                        [],
                                        [],
                                        line_segment_coordinates,
                                        diameters=diameters,
                                        volume=volume,
                                        length=length,
                                        display_dimensions=[display_width, display_height],
                                        save=False,
                                        save_name="",
                                        mask=processed_mask,
                                        overlay_text=overlay_text)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    # Save image when continued
                    display_with_overlay(image_rgb,
                                            None,
                                            [],
                                            [],
                                            line_segment_coordinates,
                                            diameters=diameters,
                                            volume=volume,
                                            length=length,
                                            save=True,
                                            display_dimensions=[display_width, display_height],
                                            save_name=f"./output/{directory_name}/results_frame_{frame_count}/final_result.png",
                                            mask=processed_mask,
                                            overlay_text=overlay_text)
                    break
                elif key == ord('r'):  # Reset the image
                    prompt_data['positive_points'].clear()
                    prompt_data['negative_points'].clear()
                    prompt_data['clicked'] = False
                    break

            # Reset prompt data for the next frame
            prompt_data['positive_points'].clear()
            prompt_data['negative_points'].clear()
            prompt_data['clicked'] = False

    # Close camera and windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
