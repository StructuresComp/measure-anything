import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
from ultralytics import YOLO
from MeasureAnything import MeasureAnything, StemInstance
from demo_utils import get_click_coordinates, display_with_overlay, scale_points, display_all_overlay_text

def save_depth_map(zed, directory, instance_index):
    depth_for_display = sl.Mat()
    zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
    depth_map = depth_for_display.get_data()[:, :, 0]
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
    color_depth_map[depth_map == 0] = [0, 0, 0]
    filename = os.path.join(directory, f"depth_map_{instance_index}.png")
    cv2.imwrite(filename, color_depth_map)

def save_mask_and_process(obj, directory, instance_index):
    # Save initial and processed masks
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(
        os.path.join(directory, f"initial_mask_{instance_index}.png"),
        (obj.initial_binary_mask * 255).astype(np.uint8)
    )
    obj.process_mask()
    cv2.imwrite(
        os.path.join(directory, f"processed_mask_{instance_index}.png"),
        obj.processed_binary_mask_0_255
    )

def perform_segmentation_steps(args, obj, base_dir, frame_count, instance_index):
    # Skeletonization or symmetry axis selection
    if args.thin_and_long:
        obj.skeletonize_and_prune()
        obj.augment_skeleton()
    else:
        obj.build_skeleton_from_symmetry_axis()
    obj.calculate_perpendicular_slope()
    line_segment_coordinates, depth = obj.calculate_line_segment_coordinates_and_depth()
    diameters = obj.calculate_diameter(line_segment_coordinates, depth)
    volume, length = obj.calculate_volume_and_length(line_segment_coordinates, depth)
    # Save results
    frame_dir = os.path.join(base_dir, f"results_frame_{frame_count}")
    os.makedirs(frame_dir, exist_ok=True)
    np.save(os.path.join(frame_dir, f"diameters_{instance_index}.npy"), diameters)
    np.save(os.path.join(frame_dir, f"volume_{instance_index}.npy"), volume)
    np.save(os.path.join(frame_dir, f"length_{instance_index}.npy"), length)
    return line_segment_coordinates, diameters, volume, length

def modify_keypoints_workflow(prompt_data, scale_x, scale_y, image_rgb, resized_depth, display_width, display_height, frame_count):
    # Setup for modifying keypoints
    cv2.setMouseCallback("Video Feed", get_click_coordinates, param=prompt_data)
    while True:
        key = cv2.waitKey(1)
        modify_instructions = [
            f"Frame: {frame_count}",
            "'Left-click' to add positive point",
            "'Ctrl + Left-click' to add negative point",
            "Press 'c' to continue"
        ]
        display_with_overlay(
            image_rgb,
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
            overlay_text=modify_instructions
        )
        if key == ord('c'):
            break
    cv2.setMouseCallback("Video Feed", lambda *args: None)
    positive_prompts = scale_points(prompt_data['positive_points'], scale_x, scale_y)
    negative_prompts = None
    if prompt_data['negative_points']:
        negative_prompts = scale_points(prompt_data['negative_points'], scale_x, scale_y)
    # Clear prompts after use
    prompt_data['positive_points'].clear()
    prompt_data['negative_points'].clear()
    prompt_data['clicked'] = False
    return positive_prompts, negative_prompts

def main():
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the YOLO keypoint detection weights (.pt file)')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    args = parser.parse_args()

    directory_name = os.path.split(args.input_svo)[1].split('.')[0]
    stride = args.stride if args.stride else 10

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters(camera_disable_self_calib=True)
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.2
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    runtime_parameters = sl.RuntimeParameters(enable_fill_mode=True)
    RGB = sl.Mat()
    frame_count = 0
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}
    keypoint_model = YOLO(args.weights)
    display_width, display_height = 1600, 900

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            depth_for_display = sl.Mat()
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
            image_depth = depth_for_display.get_data()[:, :, 0]

            resized_image = cv2.resize(image_rgb, (display_width, display_height))
            resized_depth = cv2.resize(image_depth, (display_width, display_height))

            scale_x = image_rgb.shape[1] / display_width
            scale_y = image_rgb.shape[0] / display_height

            instructions = ["Press 's' to select frame"]
            display_with_overlay(
                image_rgb, resized_depth, [], [], [], 
                display_dimensions=[display_width, display_height],
                diameters=None, save=False, save_name="", mask=None,
                overlay_text=instructions
            )

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                stem_instances = []
                results = keypoint_model(image_rgb)
                
                if results and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    for keypoint_set in keypoints:
                        keypoint_set = keypoint_set[:2]
                        if len(keypoint_set) >= 2:
                            positive_prompts = np.array([
                                (int(keypoint_set[0][0]), int(keypoint_set[0][1])),
                                (int(keypoint_set[1][0]), int(keypoint_set[1][1]))
                            ])
                            negative_prompts = np.array([
                                (int(keypoint_set[0][0]) + 25, int(keypoint_set[0][1])),
                                (int(keypoint_set[1][0]) + 25, int(keypoint_set[1][1])),
                                (int(keypoint_set[0][0]) - 25, int(keypoint_set[0][1])),
                                (int(keypoint_set[1][0]) - 25, int(keypoint_set[1][1]))
                            ])

                            intermediate_instructions = [
                                f"Frame: {frame_count}",
                                "Displaying keypoints. Press 'n' to proceed to segmentation"
                            ]
                            while True:
                                display_with_overlay(
                                    image_rgb, None, positive_prompts, negative_prompts, [],
                                    display_dimensions=[display_width, display_height],
                                    diameters=None, save=False, save_name="",
                                    mask=None, overlay_text=intermediate_instructions
                                )
                                if cv2.waitKey(1) == ord('n'):
                                    break

                            base_dir = os.path.join(".", "output", directory_name, f"results_frame_{frame_count}")
                            os.makedirs(base_dir, exist_ok=True)

                            obj = MeasureAnything(zed=zed, window=25, stride=stride, thin_and_long=args.thin_and_long, image_file=None)

                            obj.detect_mask(image=image_rgb, positive_prompts=positive_prompts, negative_prompts=negative_prompts)
                            save_mask_and_process(obj, base_dir, len(stem_instances))
                            save_depth_map(zed, base_dir, len(stem_instances))

                            line_segment_coordinates, diameters, volume, length = perform_segmentation_steps(
                                args, obj, os.path.join(".", "output", directory_name), frame_count, len(stem_instances)
                            )

                            detailed_instructions = [
                                f"Frame: {frame_count}",
                                "Press 'q' to stop",
                                "Press 'n' to go to the next keypoint instance",
                                "Press 'c' to continue to the next frame",
                                "Press 'm' to modify keypoints"
                            ]
                            while True:
                                display_with_overlay(
                                    image_rgb,
                                    None, [], [], 
                                    line_segment_coordinates,
                                    diameters=diameters, 
                                    volume=volume, 
                                    length=length,
                                    display_dimensions=[display_width, display_height],
                                    save=False, save_name="", 
                                    mask=obj.processed_binary_mask,
                                    overlay_text=detailed_instructions
                                )
                                key_inner = cv2.waitKey(1)
                                if key_inner == ord('n'):
                                    stem_instance = StemInstance(
                                        keypoints=positive_prompts,
                                        line_segment_coordinates=line_segment_coordinates,
                                        diameters=diameters,
                                        processed_mask=obj.processed_binary_mask,
                                        overlay_text=[f"Stem {len(stem_instances) + 1}", f"Mean Diameter: {np.mean(diameters):.2f} cm"]
                                    )
                                    stem_instances.append(stem_instance)
                                    break
                                elif key_inner == ord('m'):
                                    pos_prompts, neg_prompts = modify_keypoints_workflow(
                                        prompt_data, scale_x, scale_y, image_rgb, resized_depth,
                                        display_width, display_height, frame_count
                                    )
                                    # Re-run detection with modified prompts
                                    obj.detect_mask(image=image_rgb, positive_prompts=pos_prompts, negative_prompts=neg_prompts)
                                    save_mask_and_process(obj, base_dir, len(stem_instances))
                                    save_depth_map(zed, base_dir, len(stem_instances))
                                    line_segment_coordinates, diameters, volume, length = perform_segmentation_steps(
                                        args, obj, os.path.join(".", "output", directory_name), frame_count, len(stem_instances)
                                    )
                                    overlay_text = [f"Frame:{frame_count}", "Press 'c' to continue"]
                                    while True:
                                        display_with_overlay(
                                            image_rgb, None, [], [], line_segment_coordinates,
                                            diameters=diameters, volume=volume, length=length,
                                            display_dimensions=[display_width, display_height],
                                            save=False, save_name="", mask=obj.processed_binary_mask,
                                            overlay_text=overlay_text
                                        )
                                        if cv2.waitKey(1) == ord('c'):
                                            display_with_overlay(
                                                image_rgb, None, [], [], line_segment_coordinates,
                                                diameters=diameters, volume=volume, length=length,
                                                save=True,
                                                display_dimensions=[display_width, display_height],
                                                save_name=os.path.join(base_dir, f"final_result_{len(stem_instances)}.png"),
                                                mask=obj.processed_binary_mask,
                                                overlay_text=overlay_text
                                            )
                                            break
                                    # End of modification branch
                                elif key_inner == ord('q'):
                                    break

                    # Display combined results after processing all stems
                    current_display_index = 0
                    while True:
                        if current_display_index == 0:
                            display_all_overlay_text(
                                image_rgb,
                                stem_instances,
                                display_dimensions=[display_width, display_height],
                                mode='keypoints'
                            )
                        else:
                            display_all_overlay_text(
                                image_rgb,
                                stem_instances,
                                display_dimensions=[display_width, display_height],
                                mode='line_segments'
                            )
                        if cv2.waitKey(0) == ord('n'):
                            current_display_index = (current_display_index + 1) % 2
                        else:
                            break

            # End of 's' key branch
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
