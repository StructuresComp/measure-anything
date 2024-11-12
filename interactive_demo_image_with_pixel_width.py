import argparse
import cv2
import numpy as np
import os
from stemSkel import StemSkeletonization
from stemSegment import StemSegmentation
from skimage import morphology
from demo_utils import get_click_coordinates, display_with_overlay, scale_points
import pudb

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image file (png/jpg)')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    parser.add_argument('--measurement_threshold', type=float,
                        help='Threshold ratio. eg. 0.1 calculates line segments within bottom 1/10 of the image')
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

            skeleton = morphology.medial_axis((mask * 255).astype(np.uint8))

            current_stem.preprocess()
            processed_mask = current_stem.processed_binary_mask

            skeleton = morphology.medial_axis(current_stem.processed_binary_mask_0_255)
            cv2.imwrite(f"./output/{directory_name}/processed_mask.png",
                        current_stem.processed_binary_mask_0_255)
            current_stem.visualize_skeleton(current_stem.processed_binary_mask_0_255, skeleton, "processed_mask")
            # pu.db
            # Skeletonize mask and prune
            current_stem.skeletonize_and_prune()

            # Compute slope and pixel coordinates of perpendicular line segments
            current_stem.calculate_perpendicular_slope()
            
            line_segment_coordinates = current_stem.calculate_line_segment_coordinates()

            # Calculate pixel-based diameters
            pixel_diameters = [np.linalg.norm(np.array(coord[0]) - np.array(coord[1]))/300 for coord in line_segment_coordinates]
            


            # Save pixel diameters
            np.save(f"./output/{directory_name}/pixel_diameters.npy", pixel_diameters)

            # Display overlay with segmentation and wait for 'c' to continue
            overlay_text = ["Press 'r' to reset, 'c' to continue, 'q' to quit"]
            while True:
                display_with_overlay(image_rgb,
                                     [],
                                     [],
                                     line_segment_coordinates,
                                     diameters=pixel_diameters,
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
                                         diameters=pixel_diameters,
                                         save=True,
                                         display_dimensions=[display_width, display_height],
                                         save_name=f"./output/{directory_name}/diameter_result.png",
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
