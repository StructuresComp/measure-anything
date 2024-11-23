# Description: This script contains utilities functions related to the demo (main_demo.py)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


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
    shift_amount = 20  # Pixels to shift in each attempt
    max_attempts = 50
    attempt = 0
    x_original, y_original = position

    # Generate a list of shifts in a spiral pattern
    shifts = []
    for shift in range(0, max_attempts * shift_amount, shift_amount):
        shifts.extend([
            (0, shift),            # Down
            (0, -shift),           # Up
            (shift, 0),            # Right
            (-shift, 0),           # Left
            (shift, shift),        # Down-Right
            (-shift, shift),       # Down-Left
            (shift, -shift),       # Up-Right
            (-shift, -shift)       # Up-Left
        ])

    for dx, dy in shifts:
        x = x_original + dx
        y = y_original + dy

        # Ensure the box is within image boundaries
        x = max(0, min(x, image_shape[1] - box_size[0]))
        y = max(0, min(y, image_shape[0] - box_size[1]))

        box_left = x
        box_top = y
        box_right = x + box_size[0]
        box_bottom = y + box_size[1]

        box = (box_left, box_top, box_right, box_bottom)

        # Check for overlaps
        overlap = False
        for occupied in occupied_regions:
            if boxes_overlap(box, occupied):
                overlap = True
                break

        if not overlap:
            return (x, y)

    # If no non-overlapping position found, return the position adjusted to be within image boundaries
    x = max(0, min(x_original, image_shape[1] - box_size[0]))
    y = max(0, min(y_original, image_shape[0] - box_size[1]))
    return (x, y)

def boxes_overlap(box1, box2):
    # Each box is a tuple (left, top, right, bottom)
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # Check if boxes overlap
    return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)


def draw_instructions(image, instructions, position=(10, 10), box_size=(440, 120), font_path="misc/ARIAL.TTF",
                      font_size=20):
    """Draws a semi-transparent box with instructions at the specified position on the image using a custom font."""

    # Convert OpenCV image to PIL Image in order to write text using loaded font
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw the semi-transparent rectangle
    overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [position, (position[0] + box_size[0], position[1] + box_size[1])],
        fill=(255, 255, 255, 180)  # White with transparency
    )

    # Load custom font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found: {font_path}. Using default font.")
        font = ImageFont.load_default()

    # Draw each line of instruction text
    y = position[1] + 5
    for line in instructions:
        overlay_draw.text((position[0] + 10, y), line, font=font, fill=(0, 0, 0, 255))
        y += font_size + 5  # Line spacing

    # Convert back to OpenCV format
    pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# def display_with_overlay(image,
#                          depth,
#                          positive_points,
#                          negative_points,
#                          line_segments_coordinates,
#                          display_dimensions,
#                          diameters=None,
#                          volume=None,
#                          height=None,
#                          save=False, save_name="",
#                          mask=None, overlay_text=None):
#     """Displays the image with overlay instructions, point prompts, line segments and diameters if available"""
#
#     display_image = image.copy()
#     display_depth = depth.copy()
#
#     # Draw mask overlay in green if provided
#     if mask is not None:
#         overlay = display_image.copy()
#         overlay[mask == 1] = (0, 255, 0)  # Mask in green
#         display_image = cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0)
#
#     # Draw points
#     for point in positive_points:
#         cv2.circle(display_image, point, 5, (0, 0, 255), -1)  # Positive points in red
#     for point in negative_points:
#         cv2.circle(display_image, point, 5, (255, 0, 0), -1)  # Negative points in blue
#
#     # Visualize line segments with color based on diameter validity
#     if diameters is not None:
#         for idx, segment in enumerate(line_segments_coordinates):
#             color = (0, 0, 255) if not np.isnan(diameters[idx]) else (255, 0, 0)  # Red for valid, blue for NaN
#             cv2.line(display_image, (segment[1], segment[0]), (segment[3], segment[2]), color, 2)
#
#         # Compute statistics for valid diameters
#         valid_diameters = [d for d in diameters if not np.isnan(d)]
#         if valid_diameters:
#             mean_diameter = np.mean(valid_diameters)
#             median_diameter = np.median(valid_diameters)
#             min_diameter = np.min(valid_diameters)
#             max_diameter = np.max(valid_diameters)
#
#             # Format statistics for overlay
#             stats_text = [
#                 f"Mean: {mean_diameter:.2f} cm",
#                 f"Median: {median_diameter:.2f} cm",
#                 f"Min: {min_diameter:.2f} cm",
#                 f"Max: {max_diameter:.2f} cm",
#                 f"Volume: {volume:.2f} ml",
#                 f"Height: {height:.2f} cm"
#             ]
#
#             # Draw statistics overlay by the upper-right corner
#             display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
#             display_image = draw_instructions(display_image, stats_text, position=(display_image.shape[1] - 250, 10),
#                                               box_size=(230, 200), font_path="./misc/ARIAL.TTF", font_size=25)
#
#     # Add instructions box if overlay_text is provided
#     if overlay_text:
#         display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
#         display_image = draw_instructions(display_image, overlay_text, position=(10, 10), font_path="./misc/ARIAL.TTF",
#                                           font_size=25)
#
#     # Save the image if required
#     if save:
#         cv2.imwrite(save_name, display_image)
#
#     # Show the image with overlays
#     cv2.imshow("Video Feed", display_image)

def display_with_overlay(image,
                         depth,
                         positive_points,
                         negative_points,
                         line_segments_coordinates,
                         display_dimensions,
                         diameters=None,
                         volume=None,
                         length=None,
                         save=False, save_name="",
                         mask=None, overlay_text=None):
    """Displays the image with overlay instructions, point prompts, line segments, and diameters if available.
    Also displays the depth image in the bottom-right corner of the main image."""

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
                f"Max: {max_diameter:.2f} cm",
                f"Volume: {volume:.2f} ml",
                f"Length: {length:.2f} cm"
            ]

            # Draw statistics overlay by the upper-right corner
            display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
            # display_image = draw_instructions(display_image, stats_text, position=(display_image.shape[1] - 250, 10),
            #                                   box_size=(230, 200), font_path="./misc/ARIAL.TTF", font_size=25)
            display_image = draw_instructions(display_image, stats_text, position=(display_image.shape[1] - 250, display_image.shape[0] - 190),
                                              box_size=(230, 200), font_path="./misc/ARIAL.TTF", font_size=25)


    # Add instructions box if overlay_text is provided
    if overlay_text:
        display_image = cv2.resize(display_image, (display_dimensions[0], display_dimensions[1]))
        display_image = draw_instructions(display_image, overlay_text, position=(10, 10), font_path="./misc/ARIAL.TTF",
                                          font_size=25)

    if depth is not None:
        display_depth = depth.copy()
        # Resize the depth image for the bottom-right corner
        depth_height, depth_width = display_image.shape[0] // 4, display_image.shape[1] // 4  # Resize to 1/4 size
        resized_depth = cv2.resize(display_depth, (depth_width, depth_height))
        resized_depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(resized_depth, alpha=255.0 / np.max(resized_depth)), cv2.COLORMAP_JET)

        # Place the depth image in the bottom-right corner
        start_y = display_image.shape[0] - depth_height
        start_x = display_image.shape[1] - depth_width
        display_image[start_y:, start_x:] = resized_depth_colored

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

def get_heatmap_colors(num_grasps, cmap_name='viridis'):
    """
    Generates a list of colors based on the rank of grasp pairs using a colormap.
 
    Parameters:
        num_grasps (int): Total number of grasp pairs.
        cmap_name (str): Name of the matplotlib colormap to use.
 
    Returns:
        List[Tuple[int, int, int]]: List of RGB color tuples.
    """
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=0, vmax=num_grasps - 1)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    heatmap_colors = []
    for i in range(num_grasps):
        rgba = scalar_map.to_rgba(i)
        # Convert RGBA to RGB tuple with values in 0-255
        rgb = tuple(int(255 * c) for c in rgba[:3])
        heatmap_colors.append(rgb)
    return heatmap_colors

# def write_ply_with_lines(filename, points, colors, lines, line_color=(255, 0, 0)):
#     """
#     Writes a PLY file containing points and lines.

#     Parameters:
#         filename (str): Path to the output PLY file.
#         points (np.ndarray): Array of shape (N, 3) containing point coordinates.
#         colors (np.ndarray): Array of shape (N, 3) containing RGB colors for each point.
#         lines (list of tuples): List of tuples where each tuple contains two indices defining a line.
#         line_color (tuple): RGB color for the lines (default is red).
#     """
#     # Prepare vertex data
#     vertex = np.array(
#         [tuple(point) + tuple(color) for point, color in zip(points, colors)],
#         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
#     )
#     vertex_element = PlyElement.describe(vertex, 'vertex')

#     # Prepare edge data
#     edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
#     edges = np.array(
#         [(*line, *line_color) for line in lines],
#         dtype=edge_dtype
#     )
#     edge_element = PlyElement.describe(edges, 'edge')

#     # Write to PLY
#     PlyData([vertex_element, edge_element], text=True).write(filename)

def write_ply_with_lines(filename, points, colors, lines, lines_colors):
    """
    Writes a PLY file containing points and lines with per-line colors.

    Parameters:
        filename (str): Path to the output PLY file.
        points (np.ndarray): Array of shape (N, 3) containing point coordinates.
        colors (np.ndarray): Array of shape (N, 3) containing RGB colors for each point.
        lines (List[Tuple[int, int]]): List of tuples where each tuple contains two indices defining a line.
        lines_colors (List[Tuple[int, int, int]]): List of RGB color tuples for each line.
    """
    # Prepare vertex data
    vertex = np.array(
        [tuple(point) + tuple(color) for point, color in zip(points, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    vertex_element = PlyElement.describe(vertex, 'vertex')

    # Prepare edge data
    edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    edge_data = np.array(
        [(*line, *line_color) for line, line_color in zip(lines, lines_colors)],
        dtype=edge_dtype
    )
    edge_element = PlyElement.describe(edge_data, 'edge')

    # Write to PLY
    PlyData([vertex_element, edge_element], text=True).write(filename)