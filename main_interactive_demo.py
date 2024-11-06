# Description: This script runs the interactive demo of measure anything

import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
from stemSkel import StemSkeletonization
from stemSegment import StemSegmentation
from stemZ import StemZed
from skimage import morphology
from demo_utils import get_click_coordinates, display_with_overlay, scale_points


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    parser.add_argument('--measurement_threshold', type=float,
                        help='Threshold ratio. eg. 0.1 calculates line segments within bottom 1/10 of the image')
    args = parser.parse_args()
    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize SAM model
    stemSegObj = StemSegmentation()

    # Initialize the ZED camera, specify depth mode, minimum distance
    init_params = sl.InitParameters(camera_disable_self_calib=True)
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.2
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    # Enable fill mode
    runtime_parameters = sl.RuntimeParameters(enable_fill_mode=True)

    RGB = sl.Mat()
    frame_count = 0
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}

    # Main loop to extract frames and display video
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve the RGB frame
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            # Resize the image for display
            display_width, display_height = 1600, 900
            resized_image = cv2.resize(image_rgb, (display_width, display_height))

            # Calculate scale factors for x and y
            scale_x = image_rgb.shape[1] / display_width
            scale_y = image_rgb.shape[0] / display_height

            # Display frame with basic instructions
            instructions = ["Press 's' to select frame"]
            display_with_overlay(resized_image, [], [], [], display_dimensions=[display_width, display_height],
                                 diameters=None, save=False, save_name="", mask=None, overlay_text=instructions)

            # Wait for key input
            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit the loop
                break
            elif key == ord('s'):  # Stop on 's' to select points

                # Set mouse callback for collecting points
                cv2.setMouseCallback("Video Feed", get_click_coordinates, param=prompt_data)

                # Wait for user to select points and press 'c' to continue
                while True:
                    key = cv2.waitKey(1)

                    # Detailed instructions
                    detailed_instructions = [
                        f"Frame: {frame_count}",
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

                if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                    os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/initial_mask.png",
                            (mask * 255).astype(np.uint8))

                # Save depth frame
                depth_for_display = sl.Mat()
                zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                depth_map = depth_for_display.get_data()
                depth_map = depth_map[:, :, 0]
                # Normalize if needed (assuming depth_map is already in 8-bit range)
                depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
                color_depth_map[depth_map == 0] = [0, 0, 0]

                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/depth_map.png", color_depth_map)

                # Skeletonize, identify perpendicular line segments
                stride = args.stride if args.stride else 10
                threshold = args.measurement_threshold if args.measurement_threshold else 0.95
                current_stem = StemSkeletonization(threshold=threshold, window=25, stride=stride,
                                                   image_file=f'./output/{directory_name}/results_frame_{frame_count}/initial_mask.png')

                # Load image, preprocess and save the processed mask
                current_stem.load_image()

                skeleton = morphology.medial_axis((mask * 255).astype(np.uint8))
                # current_stem.visualize_skeleton((mask * 255).astype(np.uint8), skeleton, "unprocessed_mask")

                current_stem.preprocess()
                processed_mask = current_stem.processed_binary_mask

                skeleton = morphology.medial_axis(current_stem.processed_binary_mask_0_255)
                # current_stem.visualize_skeleton(current_stem.processed_binary_mask_0_255, skeleton, "processed_mask")
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask.png",
                            current_stem.processed_binary_mask_0_255)

                # Skeletonize mask and prune
                current_stem.skeletonize_and_prune()

                # Compute slope and pixel coordinates of perpendicular line segments
                current_stem.calculate_perpendicular_slope()
                line_segment_coordinates = current_stem.calculate_line_segment_coordinates()

                # Retrieve the depth frame
                stemZedObj = StemZed(zed)
                diameters = stemZedObj.calculate_3d_distance(line_segment_coordinates)

                # Save diameters
                np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters.npy", diameters)

                # Display overlay with segmentation and wait for 'c' to continue
                overlay_text = [f"Frame:{frame_count}", "Press 'c' to continue"]
                while True:
                    display_with_overlay(image_rgb,
                                         [],
                                         [],
                                         line_segment_coordinates,
                                         diameters=diameters,
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
                                             diameters=diameters,
                                             save=True,
                                             display_dimensions=[display_width, display_height],
                                             save_name=f"./output/{directory_name}/results_frame_{frame_count}/diameter_result.png",
                                             mask=processed_mask,
                                             overlay_text=overlay_text)
                        break

                # Reset prompt data for the next frame
                prompt_data['positive_points'].clear()
                prompt_data['negative_points'].clear()
                prompt_data['clicked'] = False

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
