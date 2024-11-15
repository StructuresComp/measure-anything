# Description: Modified script using keypoints from a keypoint detection model instead of user points for segmentation

import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
import time
from stemSkel import StemSkeletonization
from stemSegment import StemSegmentation
from stemZ import StemZed
from skimage import morphology
from demo_utils import display_with_overlay, scale_points
from ultralytics import YOLO
import pudb

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Demo of Measure Anything using keypoints from YOLO model")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the YOLO keypoint detection weights (.pt file)')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    parser.add_argument('--measurement_threshold', type=float, help='Threshold ratio. eg. 0.1 calculates line segments within bottom 1/10 of the image')
    args = parser.parse_args()
    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize SAM model and YOLO keypoint detection model
    stemSegObj = StemSegmentation()
    keypoint_model = YOLO(args.weights)

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
    display_width, display_height = 1600, 900

    # Main loop to extract frames and display video
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve the RGB frame
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            # Resize the image for display
            resized_image = cv2.resize(image_rgb, (display_width, display_height))

            # Display frame with basic instructions
            instructions = ["Press 's' to select frame"]
            display_with_overlay(resized_image, [], [], [], display_dimensions=[display_width, display_height],
                                 diameters=None, save=False, save_name="", mask=None, overlay_text=instructions)

            # Wait for key input
            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit the loop
                break
            elif key == ord('s'):  # Stop on 's' to select points

                # Run YOLO keypoint detection inference on the frame
                results = keypoint_model(image_rgb)

                # Extract keypoints from the inference results
                if results and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy()  # Assuming keypoints are available
                    
                    # Iterate over each detected instance
                    for keypoint_set in keypoints:
                        keypoint_set = keypoint_set[:2]  # Use only the first two keypoints for each instance
                        if len(keypoint_set) >= 2:  # Ensure at least two keypoints are available
                            positive_prompts = np.array([(int(keypoint_set[0][0]), int(keypoint_set[0][1])),
                                                (int(keypoint_set[1][0]), int(keypoint_set[1][1]))])

                            # Display intermediate result with just the points
                            intermediate_instructions = [
                                f"Frame: {frame_count}",
                                "Displaying keypoints. Press 'n' to proceed to segmentation"
                            ]
                            while True:
                                display_with_overlay(image_rgb,
                                                    positive_prompts,
                                                    [],
                                                    [],
                                                    display_dimensions=[display_width, display_height],
                                                    save=False,
                                                    save_name="",
                                                    mask=None,
                                                    overlay_text=intermediate_instructions)
                                key = cv2.waitKey(1)
                                if key == ord('n'):
                                    break
                        if len(keypoint_set) >= 2:  # Ensure at least two keypoints are available
                            positive_prompts = np.array([(int(keypoint_set[0][0]), int(keypoint_set[0][1])),
                                                (int(keypoint_set[1][0]), int(keypoint_set[1][1]))])
                            # positive_prompts = np.array([(int(keypoint_set[0][0]), int(keypoint_set[0][1]))])
                            

                            # Run SAM segmentation & save initial mask
                            mask = stemSegObj.inference(image_rgb, positive_prompts, None)

                            if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                                os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

                            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/initial_mask.png",
                                        (mask * 255).astype(np.uint8))

                            # Save depth frame
                            depth_for_display = sl.Mat()
                            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                            depth_map = depth_for_display.get_data()
                            depth_map = depth_map[:, :, 0]
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
                            current_stem.preprocess()
                            processed_mask = current_stem.processed_binary_mask

                            skeleton = morphology.medial_axis(current_stem.processed_binary_mask_0_255)
                            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask.png",
                                        current_stem.processed_binary_mask_0_255)

                            # Skeletonize mask and prune
                            current_stem.skeletonize_and_prune()

                            # Compute slope and pixel coordinates of perpendicular line segments
                            current_stem.calculate_perpendicular_slope()
                            line_segment_coordinates = current_stem.calculate_line_segment_coordinates()

                            # Retrieve the depth frame and calculate diameters
                            stemZedObj = StemZed(zed)
                            diameters = stemZedObj.calculate_3d_distance(line_segment_coordinates)

                            # Save diameters
                            np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters.npy", diameters)

                            # Display overlay with segmentation and wait for user input to continue
                            detailed_instructions = [
                                f"Frame: {frame_count}",
                                "Press 'q' to stop",
                                "Press 'n' to go to the next keypoint instance",
                                "Press 'c' to continue to the next frame"
                            ]
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
                                                     overlay_text=detailed_instructions)
                                key = cv2.waitKey(1)
                                # time.sleep(3)
                                if key == ord('n'):
                                    break

                if key == ord('c'):  # Continue once the frame is done
                    break

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()