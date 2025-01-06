import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
from ultralytics import YOLO
from MeasureAnything import MeasureAnything, StemInstance
from demo_utils import get_click_coordinates, display_with_overlay, scale_points, display_all_overlay_text


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the YOLO keypoint detection weights (.pt file)')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    args = parser.parse_args()

    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Initialize command line inputs
    stride = args.stride if args.stride else 10

    # Create a ZED camera object
    zed = sl.Camera()

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

    # Initialize YOLO Keypoint Detection Model
    keypoint_model = YOLO(args.weights)

    # Main loop to extract frames and display video
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve the RGB frame
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            # Retrieve the depth frame
            depth_for_display = sl.Mat()
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
            image_depth = depth_for_display.get_data()
            image_depth = image_depth[:, :, 0]

            # Resize the image for display
            display_width, display_height = 1600, 900
            resized_image = cv2.resize(image_rgb, (display_width, display_height))
            resized_depth = cv2.resize(image_depth, (display_width, display_height))

            # Calculate scale factors for x and y
            scale_x = image_rgb.shape[1] / display_width
            scale_y = image_rgb.shape[0] / display_height

            # Display frame with basic instructions
            instructions = ["Press 's' to select frame"]
            display_with_overlay(image_rgb,
                                 resized_depth,
                                 [],
                                 [],
                                 [],
                                 display_dimensions=[display_width, display_height],
                                 diameters=None,
                                 save=False,
                                 save_name="",
                                 mask=None,
                                 overlay_text=instructions)

            # Wait for key input
            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit the loop
                break
            elif key == ord('s'):  # Stop on 's' to select points

                # List to store stem instances for the current frame
                stem_instances = []

                # Run YOLO keypoint detection inference on the frame
                results = keypoint_model(image_rgb)

                # Initialize MeasureAnything object
                object = MeasureAnything(zed=zed, window=25, stride=stride, thin_and_long=args.thin_and_long,
                                        image_file=None)

                if results and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy()  # Assuming keypoints are available
                    
                    # Iterate over each detected instance
                    for keypoint_set in keypoints:
                        keypoint_set = keypoint_set[:2]  # Use only the first two keypoints for each instance
                        if len(keypoint_set) >= 2:  # Ensure at least two keypoints are available
                            positive_prompts = np.array([(int(keypoint_set[0][0]), int(keypoint_set[0][1])),
                                                (int(keypoint_set[1][0]), int(keypoint_set[1][1]))])
                            negative_prompts = np.array([(int(keypoint_set[0][0]) + 25, int(keypoint_set[0][1])),
                                                (int(keypoint_set[1][0]) + 25, int(keypoint_set[1][1])),
                                                (int(keypoint_set[0][0]) - 25, int(keypoint_set[0][1])),
                                                (int(keypoint_set[1][0]) - 25, int(keypoint_set[1][1]))])

                            # Display intermediate result with just the points
                            intermediate_instructions = [
                                f"Frame: {frame_count}",
                                "Displaying keypoints. Press 'n' to proceed to segmentation"
                            ]
                            while True:
                                display_with_overlay(image_rgb,
                                                    None,
                                                    positive_prompts,
                                                    negative_prompts,
                                                    [],
                                                    display_dimensions=[display_width, display_height],
                                                    diameters=None,
                                                    volume=None,
                                                    length=None,
                                                    save=False,
                                                    save_name="",
                                                    mask=None,
                                                    overlay_text=intermediate_instructions)
                                key = cv2.waitKey(1)
                                if key == ord('n'):
                                    break


                            # Create directory to save results
                            if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                                os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

                            object.detect_mask(image=image_rgb, positive_prompts=positive_prompts,
                                            negative_prompts=negative_prompts)

                            # Save initial mask
                            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/initial_mask_{len(stem_instances)}.png",
                                        (object.initial_binary_mask * 255).astype(np.uint8))

                            # Save depth frame as color
                            depth_for_display = sl.Mat()
                            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                            depth_map = depth_for_display.get_data()
                            depth_map = depth_map[:, :, 0]
                            # Normalize if needed (assuming depth_map is already in 8-bit range)
                            depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                            color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
                            color_depth_map[depth_map == 0] = [0, 0, 0]

                            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/depth_map_{len(stem_instances)}.png", color_depth_map)

                            # Preprocess and save
                            object.process_mask()
                            processed_mask = object.processed_binary_mask
                            cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask_{len(stem_instances)}.png",
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


                            # Calculate dimensional measurements
                            diameters = object.calculate_diameter(line_segment_coordinates, depth)
                            volume, length = object.calculate_volume_and_length(line_segment_coordinates, depth)

                            # Save results
                            np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters_{len(stem_instances)}.npy", diameters)
                            np.save(f"./output/{directory_name}/results_frame_{frame_count}/volume_{len(stem_instances)}.npy", volume)
                            np.save(f"./output/{directory_name}/results_frame_{frame_count}/length_{len(stem_instances)}.npy", length)

                            # Display overlay with segmentation and wait for user input to continue
                            detailed_instructions = [
                                f"Frame: {frame_count}",
                                "Press 'q' to stop",
                                "Press 'n' to go to the next keypoint instance",
                                "Press 'c' to continue to the next frame",
                                "Press 'm' to modify keypoints"
                                ]
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
                                                    overlay_text=detailed_instructions)
                                key = cv2.waitKey(1)
                                if key == ord('n'):
                                    # Create a StemInstance object and add it to the list
                                    stem_instance = StemInstance(
                                        keypoints=positive_prompts,
                                        line_segment_coordinates=line_segment_coordinates,
                                        diameters=diameters,
                                        processed_mask=processed_mask,
                                        overlay_text=[f"Stem {len(stem_instances) + 1}", f"Mean Diameter: {np.mean(diameters):.2f} cm"]
                                    )
                                    stem_instances.append(stem_instance)
                                    break
                                elif key == ord('m'):
                                    # Set mouse callback for collecting points
                                    cv2.setMouseCallback("Video Feed", get_click_coordinates, param=prompt_data)

                                    # Wait for user to select points and press 'c' to continue
                                    while True:
                                        key = cv2.waitKey(1)

                                        # Detailed instructions
                                        modify_instructions = [
                                            f"Frame: {frame_count}",
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
                                                            overlay_text=modify_instructions)
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


                                    object.detect_mask(image=image_rgb, positive_prompts=positive_prompts,
                                                    negative_prompts=negative_prompts)

                                    # Save initial mask
                                    cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/initial_mask_{len(stem_instances)}.png",
                                                (object.initial_binary_mask * 255).astype(np.uint8))

                                    # Save depth frame as color
                                    depth_for_display = sl.Mat()
                                    zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                                    depth_map = depth_for_display.get_data()
                                    depth_map = depth_map[:, :, 0]
                                    # Normalize if needed (assuming depth_map is already in 8-bit range)
                                    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                                    color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
                                    color_depth_map[depth_map == 0] = [0, 0, 0]

                                    cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/depth_map_{len(stem_instances)}.png", color_depth_map)

                                    # Preprocess and save
                                    object.process_mask()
                                    processed_mask = object.processed_binary_mask
                                    cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask_{len(stem_instances)}.png",
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

                                    # Calculate dimensional measurements
                                    diameters = object.calculate_diameter(line_segment_coordinates, depth)
                                    volume, length = object.calculate_volume_and_length(line_segment_coordinates, depth)

                                    # Save results
                                    np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters_{len(stem_instances)}.npy", diameters)
                                    np.save(f"./output/{directory_name}/results_frame_{frame_count}/volume_{len(stem_instances)}.npy", volume)
                                    np.save(f"./output/{directory_name}/results_frame_{frame_count}/length_{len(stem_instances)}.npy", length)

                                    # Display overlay with segmentation and wait for 'c' to continue
                                    overlay_text = [f"Frame:{frame_count}", "Press 'c' to continue"]
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
                                                                save_name=f"./output/{directory_name}/results_frame_{frame_count}/final_result_{len(stem_instances)}.png",
                                                                mask=processed_mask,
                                                                overlay_text=overlay_text)
                                            # Reset prompt data for the next instance
                                            prompt_data['positive_points'].clear()
                                            prompt_data['negative_points'].clear()
                                            prompt_data['clicked'] = False
                                            break


                    # After processing all stems, display the combined results
                    current_display_index = 0
                    while True:
                        if current_display_index == 0:
                            # Display the frame with all keypoints
                            display_all_overlay_text(
                                image_rgb,
                                stem_instances,
                                display_dimensions=[display_width, display_height],
                                mode='keypoints'
                            )
                        else:
                            # Display the frame with line segments and overlay texts
                            display_all_overlay_text(
                                image_rgb,
                                stem_instances,
                                display_dimensions=[display_width, display_height],
                                mode='line_segments'
                            )
                        key = cv2.waitKey(0)
                        if key == ord('n'):
                            current_display_index = (current_display_index + 1) % 2
                        else:
                            break    

                if key == ord('q'):  # Continue once the frame is done
                    break
                else:
                    continue

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
