# Description: This script contains utilities functions related to the demo (main_demo.py)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class StemInstance:
    def __init__(self, keypoints, line_segment_coordinates, diameters, processed_mask, overlay_text):
        self.keypoints = keypoints
        self.line_segment_coordinates = line_segment_coordinates
        self.diameters = diameters
        self.processed_mask = processed_mask
        self.overlay_text = overlay_text

def get_click_coordinates(event, x, y, flags, param):
    """Handles mouse click events for positive and negative point prompts."""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if Ctrl is held down for negative points
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            print(f"Negative point added at: ({x}, {y})")
            param['negative_points'].append((x, y))
        else:
            print(f"Positive point added at: ({x}, {y})")
            param['positive_points'].append((x, y))
    param['clicked'] = True

def get_overlay_box_size(instructions, font_path="./misc/arial.ttf", font_size=16, padding=10, line_spacing=5):
    # Load custom font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font '{font_path}': {e}. Using default font.")
        font = ImageFont.load_default()

    text_width = 0
    text_height = 0
    for line in instructions:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        text_width = max(text_width, line_width)
        text_height += line_height + line_spacing

    text_height -= line_spacing  # Remove extra spacing after the last line

    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding

    return (box_width, box_height)

def adjust_position_to_avoid_overlap(position, box_size, occupied_regions, image_shape):
    max_attempts = 10
    shift_amount = 20  # Pixels to shift in each attempt
    attempt = 0
    x, y = position

    while attempt < max_attempts:
        box_left = x
        box_top = y
        box_right = x + box_size[0]
        box_bottom = y + box_size[1]

        # Ensure the box is within image boundaries
        if box_right > image_shape[1]:
            x = image_shape[1] - box_size[0]
        if box_left < 0:
            x = 0
        if box_bottom > image_shape[0]:
            y = image_shape[0] - box_size[1]
        if box_top < 0:
            y = 0

        box = (x, y, x + box_size[0], y + box_size[1])

        overlap = False
        for occupied in occupied_regions:
            if boxes_overlap(box, occupied):
                overlap = True
                break

        if not overlap:
            return (x, y)

        # Adjust position to try to avoid overlap
        y += shift_amount  # Move down
        attempt += 1

    # If no non-overlapping position found, return the original position
    return (x, y)

def boxes_overlap(box1, box2):
    # Each box is a tuple (left, top, right, bottom)
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # Check if boxes overlap
    return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)


def draw_instructions(image, instructions, position=(10, 10), box_size=(440, 120), font_path="./misc/arial.ttf",
                      font_size=20):
    """Draws a semi-transparent box with instructions at the specified position on the image using a custom font."""

    # Convert OpenCV image to PIL Image to write text using loaded font
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Load custom font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font '{font_path}': {e}. Using default font.")
        font = ImageFont.load_default()

    # Calculate the size of the text box
    text_width = 0
    text_height = 0
    line_spacing = 5  # Adjust line spacing as needed
    for line in instructions:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        text_width = max(text_width, line_width)
        text_height += line_height + line_spacing

    text_height -= line_spacing  # Remove extra spacing after the last line

    # Adjust the box size based on text size and padding
    padding = 10
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding

    # Ensure the box doesn't go beyond the image boundaries
    image_width, image_height = pil_image.size
    box_left = position[0]
    box_top = position[1]
    box_right = min(box_left + box_width, image_width)
    box_bottom = min(box_top + box_height, image_height)

    # Draw the semi-transparent rectangle
    overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [box_left, box_top, box_right, box_bottom],
        fill=(255, 255, 255, 180)  # White with transparency
    )

    # Composite the overlay with the image
    pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay)

    # Draw each line of instruction text
    draw = ImageDraw.Draw(pil_image)
    y = box_top + padding
    for line in instructions:
        draw.text((box_left + padding, y), line, font=font, fill=(0, 0, 0, 255))
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y += line_height + line_spacing  # Line height + spacing

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def display_with_overlay(image,
                         positive_points,
                         negative_points,
                         line_segments_coordinates,
                         display_dimensions,
                         diameters=None,
                         save=False, save_name="",
                         mask=None, overlay_text=None):
    """Displays the image with overlay instructions, point prompts, line segments and diameters if available"""

    display_image = image.copy()

    # Draw mask overlay in green if provided
    if mask is not None:
        overlay = display_image.copy()
        overlay[mask == 1] = (0, 255, 0)  # Mask in green
        display_image = cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0)

    # Draw points
    for point in positive_points:
        cv2.circle(display_image, point, 5, (0, 0, 255), -1)  # Positive points in red
    for point in negative_points:
        cv2.circle(display_image, point, 5, (255, 0, 0), -1)  # Negative points in blue

    # Visualize line segments with color based on diameter validity
    if diameters is not None:
        for idx, segment in enumerate(line_segments_coordinates):
            color = (0, 0, 255) if not np.isnan(diameters[idx]) else (255, 0, 0)  # Red for valid, blue for NaN
            cv2.line(display_image, (segment[1], segment[0]), (segment[3], segment[2]), color, 2)

        # Compute statistics for valid diameters
        valid_diameters = [d for d in diameters if not np.isnan(d)]
        if valid_diameters:
            mean_diameter = np.mean(valid_diameters)
            median_diameter = np.median(valid_diameters)
            min_diameter = np.min(valid_diameters)
            max_diameter = np.max(valid_diameters)

            # Format statistics for overlay
            stats_text = [
                f"Mean: {mean_diameter:.2f} cm",
                f"Median: {median_diameter:.2f} cm",
                f"Min: {min_diameter:.2f} cm",
                f"Max: {max_diameter:.2f} cm"
            ]

            # Draw statistics overlay by the upper-right corner
            display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
            display_image = draw_instructions(display_image, stats_text, position=(display_image.shape[1] - 220, 10),
                                              box_size=(200, 120), font_path="misc/ARIAL.TTF", font_size=25)

    # Add instructions box if overlay_text is provided
    if overlay_text:
        display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
        display_image = draw_instructions(display_image, overlay_text, position=(10, 10), font_path="misc/ARIAL.TTF",
                                          font_size=25)

    # Save the image if required
    if save:
        cv2.imwrite(save_name, display_image)

    # Show the image with overlays
    cv2.imshow("Video Feed", display_image)

def display_all_overlay_text(image, stem_instances, display_dimensions, mode='keypoints'):
    """Displays the image with all keypoints or all line segments and overlay texts for each stem instance."""
    display_image = image.copy()

    # Resize the image for display
    display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
    scaling_factor_x = display_dimensions[0] / image.shape[1]
    scaling_factor_y = display_dimensions[1] / image.shape[0]

    # List to keep track of occupied regions (bounding boxes of overlay texts)
    occupied_regions = []

    for stem_instance in stem_instances:
        # Scale keypoints and line segments to the display size
        scaled_keypoints = [(int(pt[0] * scaling_factor_x), int(pt[1] * scaling_factor_y)) for pt in stem_instance.keypoints]
        scaled_line_segments = []
        for segment in stem_instance.line_segment_coordinates:
            scaled_segment = [
                int(segment[1] * scaling_factor_x),
                int(segment[0] * scaling_factor_y),
                int(segment[3] * scaling_factor_x),
                int(segment[2] * scaling_factor_y)
            ]
            scaled_line_segments.append(scaled_segment)

        # Scale the processed mask
        if stem_instance.processed_mask is not None:
            scaled_mask = cv2.resize(stem_instance.processed_mask, (display_dimensions[0], display_dimensions[1]))
        else:
            scaled_mask = None

        if mode == 'keypoints':
            # Draw keypoints
            for point in scaled_keypoints:
                cv2.circle(display_image, point, 5, (0, 0, 255), -1)  # Positive points in red
        elif mode == 'line_segments':
            # Overlay the processed mask
            if scaled_mask is not None:
                overlay = display_image.copy()
                overlay[scaled_mask == 1] = (0, 255, 0)  # Mask in green
                display_image = cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0)

            # Draw line segments and diameters
            for idx, segment in enumerate(scaled_line_segments):
                color = (0, 0, 255) if not np.isnan(stem_instance.diameters[idx]) else (255, 0, 0)
                cv2.line(display_image, (segment[0], segment[1]), (segment[2], segment[3]), color, 2)

            # Determine position for overlay text
            text_position = (scaled_keypoints[0][0], scaled_keypoints[0][1] - 50)  # Initial position

            # Adjust position to avoid overlaps
            overlay_box_size = get_overlay_box_size(stem_instance.overlay_text, font_path="./misc/arial.ttf", font_size=16)
            text_position = adjust_position_to_avoid_overlap(text_position, overlay_box_size, occupied_regions, display_image.shape)

            # Update occupied regions
            box_left = text_position[0]
            box_top = text_position[1]
            box_right = box_left + overlay_box_size[0]
            box_bottom = box_top + overlay_box_size[1]
            occupied_regions.append((box_left, box_top, box_right, box_bottom))

            # Draw overlay text with adjusted position
            display_image = draw_instructions(
                display_image,
                stem_instance.overlay_text,
                position=(box_left, box_top),
                font_size=16,  # Smaller font size
                font_path="./misc/arial.ttf"
            )

    # Show the image with overlays
    cv2.imshow("Video Feed", display_image)


def scale_points(points, scale_x, scale_y):
    # Scales point coordinates to match the original image dimensions. Input points are in window dimensions.

    return np.array([(int(x * scale_x), int(y * scale_y)) for x, y in points])
