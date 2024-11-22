"""Contains tools for creating label images."""

import os
import cv2
import random
import numpy as np
import logging as log
from copy import deepcopy


def random_color():
    """Generate a random RGB color.

    Returns:
        color (list): Random RGB color.

    """
    color = (random.randint(0, 255), random.randint(0, 255),
             random.randint(0, 255))
    return color


def poly_to_image(img, poly, value):
    """Paint a polygon in an image.

    Note:
        Input image will be modified.

    Args:
        img (np.array): Input image that will be modifed.
        poly (list(float)): List of polygon points.
        value (list(int) or int): Value of the points inside the polygon. Can
            be 3-channel (RGB) or 1-channel (mono) depending on the input
            image.
 
    """
    mask = np.array([poly], dtype='int32')
    cv2.fillPoly(img, mask, value)


def bbox_to_image(img, bbox, value, line_thickness=2):
    """Paint a bounding box in an image.

    Note:
        Input image will be modified.

    Args:
        img (np.array): Input image that will be modifed.
        bbox (list(float)): Bounding box (x, y, widht, height).
        value (list(int) or int): Value of the points on the bounding box. Can
            be 3-channel (RGB) or 1-channel (mono) depending on the input
            image.
        line_thickness (int, optional): Thickness of the bounding box in
            pixels. Defaults to 2.

    """
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), value,
                  line_thickness)


def get_source_image_from_label_path(label_full_path):
    """Get the source image and relevant information from the label file path.

    Args:
        label_full_path (str): Full path to the label file.

    Returns:
        label_file_name (str): Name of the label file.
        sensor (str): Name of sensor that was was used for capturing the source
            image.
        source_image (str): Source image.

    """
    label_file = label_full_path.split('_images/')[-1]
    label_file_path = label_full_path.split('/' + label_file)[0]
    label_file_name, ext = os.path.splitext(label_file)
    sensor = (label_file_path.split('/labels')[0]).split('/')[-1]
    source_image_path = label_full_path.split('/labels')
    source_image_path = source_image_path[0] + source_image_path[1]
    source_image_path = source_image_path.replace('_label_' + sensor + '.json',
                                                  '.png')

    source_image = cv2.imread(source_image_path)

    return label_file_name, sensor, source_image


def create_label_image_from_json_data(json_data,
                                      original_img,
                                      draw_bbox=True,
                                      image_save_file=None):
    """Create a label image based on the data from a json label file.

    Args:
        json_data (dict): Dictonary containing the data from a single json
            label file. It should contain 'poly', 'bbox' and 'labels' keys.
        original_img (np.array): Original image for which the labeling is done.
        draw_bbox (bool): If set to false, bounding boxes will not be in the
            output image.
        image_save_file (str, optional): Save location and file name for the
            created label image. Defaults to None.

    Returns:
        output (np.array): Image with annotations and bounding boxes.

    """
    log.debug("Creating a label image from a json file.")

    color_label_img = deepcopy(original_img)
    bbox_img = deepcopy(original_img)
    output = deepcopy(original_img)

    colors = {}
    for i in range(len(json_data['labels'])):
        colors[json_data['labels'][i]] = random_color()

    for i in range(len(json_data['labels'])):
        poly = json_data['poly'][i]
        bbox = json_data['bbox'][i]

        color = colors[json_data['labels'][i]]
        poly_to_image(color_label_img, poly, color)
        if (draw_bbox):
            bbox_to_image(bbox_img, bbox, color)
        alpha = 0.6
        cv2.addWeighted(color_label_img, alpha, bbox_img, 1 - alpha, 0, output)

    if image_save_file is not None:
        cv2.imwrite(image_save_file, output)

    return output
