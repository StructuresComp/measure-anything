# Description: This script contains utilities functions related to the demo (main_demo.py)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement


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
            display_image = draw_instructions(display_image, stats_text, position=(display_image.shape[1] - 250, 10),
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


def scale_points(points, scale_x, scale_y):
    # Scales point coordinates to match the original image dimensions. Input points are in window dimensions.

    return np.array([(int(x * scale_x), int(y * scale_y)) for x, y in points])


def write_ply_with_lines(filename, points, colors, lines, line_color=(255, 0, 0)):
    """
    Writes a PLY file containing points and lines.

    Parameters:
        filename (str): Path to the output PLY file.
        points (np.ndarray): Array of shape (N, 3) containing point coordinates.
        colors (np.ndarray): Array of shape (N, 3) containing RGB colors for each point.
        lines (list of tuples): List of tuples where each tuple contains two indices defining a line.
        line_color (tuple): RGB color for the lines (default is red).
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
    edges = np.array(
        [(*line, *line_color) for line in lines],
        dtype=edge_dtype
    )
    edge_element = PlyElement.describe(edges, 'edge')

    # Write to PLY
    PlyData([vertex_element, edge_element], text=True).write(filename)