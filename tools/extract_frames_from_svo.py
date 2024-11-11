import argparse
import os
import cv2
import numpy as np
import pyzed.sl as sl
import pudb

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Extract frames from SVO video")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted frames')
    parser.add_argument('--fps', type=float, default=1, help='Frames per second to extract from the video (can be less than 1 to take one frame every N seconds)')
    args = parser.parse_args()

    # Extract base name of the SVO file
    base_name = os.path.splitext(os.path.basename(args.input_svo))[0]

    # Create output directories for RGB and Depth frames
    rgb_output_dir = os.path.join(args.output_dir, 'RGB_frames')
    depth_output_dir = os.path.join(args.output_dir, 'Depth_frames')
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters for the ZED camera
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.2  # Set minimum depth distance
    # init_params.depth_stabilization = True

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    # Runtime parameters
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    runtime_parameters.texture_confidence_threshold = 50
    runtime_parameters.enable_fill_mode = True

    # Get the SVO FPS to determine frame extraction interval
    camera_fps = zed.get_camera_information().camera_configuration.fps
    frame_interval = max(5, int(camera_fps / args.fps))

    # Mat objects to hold RGB and Depth frames
    rgb_frame = sl.Mat()
    depth_frame = sl.Mat()

    frame_count = 0
    saved_frame_count = 0

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Only save frames at the specified interval
            if frame_count % frame_interval == 0:
                # Retrieve RGB frame
                zed.retrieve_image(rgb_frame, sl.VIEW.LEFT)
                rgb_image = rgb_frame.get_data()
                rgb_image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)

                # Save RGB frame
                rgb_filename = os.path.join(rgb_output_dir, f'{base_name}_rgb_frame_{saved_frame_count:04d}.png')
                cv2.imwrite(rgb_filename, rgb_image_bgr)

                # Retrieve depth frame
                zed.retrieve_image(depth_frame, sl.VIEW.DEPTH)
                depth_image = depth_frame.get_data()[:, :, 0]

                # Normalize depth image for saving
                depth_image_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color_map = cv2.applyColorMap(depth_image_norm, cv2.COLORMAP_JET)

                # Save Depth frame
                depth_filename = os.path.join(depth_output_dir, f'{base_name}_depth_frame_{saved_frame_count:04d}.png')
                cv2.imwrite(depth_filename, depth_color_map)

                saved_frame_count += 1

            frame_count += 1

        else:
            break

    # Close the camera
    zed.close()
    print(f"Extracted {saved_frame_count} frames successfully.")

if __name__ == '__main__':
    main()
