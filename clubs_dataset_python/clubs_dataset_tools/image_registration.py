"""Contains tools for image projection.


The function register_depth_image registeres a depth image to an RGB image, and
project_points_to_camera projects 3D points to a specific camera.
"""

import cv2
import numpy as np
import logging as log

from clubs_dataset_python.clubs_dataset_tools.common import (convert_depth_float_to_uint)

def depth_to_3d(depth_image, intrinsics):
    # Create a meshgrid of image coordinates
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # Extract focal lengths and principal points from the intrinsics matrix
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Calculate the real-world coordinates
    x = (j - cx) * depth_image / fx
    y = (i - cy) * depth_image / fy
    z = depth_image
    
    # Stack into a 3D point cloud
    points_3d = np.dstack((x, y, z))
    return points_3d


def register_depth_image(float_depth_image,
                         rgb_intrinsics,
                         depth_intrinsics,
                         extrinsics,
                         rgb_shape,
                         depth_scale=1.0):
    """Register a depth image to an RGB image.

    Registered depth image has the same size as the original RGB image.
    RGB intrinsics are used to convert the registered depth image to 3D points.

    Args:
        float_depth_image (np.array): Float depth image.
        rgb_intrinsics (np.array): Intrinsic parameters of the RGB camera.
        depth_intrinsics (np.array): Intrinsic parameters of the depth camera.
        extrinsics (np.array): Extrinsic parameters between the RGB and the
            depth camera.
        rgb_shape (tuple(int)): Image size of the RGB image (rows, columns).
        depth_scale (float, optional): Conversion factor for the depth (e.g.
            1 means that value of 1000 in uint16 depth image corresponds to 1.0
            in float depth image and to 1m in real world). Defaults to 1.0.

    Returns:
        float_depth_registered (np.array): Depth image registered to rgb, float
            type.
        uint_depth_registered (np.array): Depth image registered to rgb, uint16
            type.

    """
    log.debug("Computing the registered depth image.")

    # depth_points_3d = cv2.rgbd.depthTo3d(float_depth_image, depth_intrinsics)
    depth_points_3d = depth_to_3d(float_depth_image, depth_intrinsics)
    depth_points_in_rgb_frame = cv2.perspectiveTransform(
        depth_points_3d, extrinsics)

    fx = rgb_intrinsics[0, 0]
    fy = rgb_intrinsics[1, 1]
    cx = rgb_intrinsics[0, 2]
    cy = rgb_intrinsics[1, 2]

    float_depth_registered = np.zeros(rgb_shape, dtype='float')

    for points in depth_points_in_rgb_frame:
        for point in points:
            u = int(fx * point[0] / point[2] + cx)
            v = int(fy * point[1] / point[2] + cy)

            height = rgb_shape[0]
            width = rgb_shape[1]
            if (u >= 0 and u < width and v >= 0 and v < height):
                float_depth_registered[v, u] = point[2]

    uint_depth_registered = convert_depth_float_to_uint(float_depth_registered,
                                                        depth_scale)
    kernel = np.ones((3, 3), np.uint16)
    float_depth_registered = cv2.morphologyEx(float_depth_registered,
                                              cv2.MORPH_CLOSE, kernel)
    uint_depth_registered = cv2.morphologyEx(uint_depth_registered,
                                             cv2.MORPH_CLOSE, kernel)

    return float_depth_registered, uint_depth_registered


def project_points_to_camera(points_3d, extrinsics, intrinsics, distortion,
                             image_size):
    """Project points to the specific camera.

    Camera is defined with the extrinsic and intrinsic parameteres, distortion
    coefficients, and image size. Additionally, this function provides a
    bounding box for the projected points.

    Args:
        points_3d (np.array): Points in 3D
        extrinsics (np.array): Intrinsic parameters of the camera
        intrinsics (np.array): Extrinsic parameters of the camera
        distortion (np.array): Distortion coefficients of the camera
        image_size (tuple(int)): Image size of the camera (rows, columns)

    Returns:
        projected_points (np.array): Points in the new camera image, capped at
        the image_size
        bounding_box (np.array): Bounding box of the points in the new camera
        view

    """
    log.debug("Projecting 3D points to the specified camera frame.")

    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]

    projected_points_raw = cv2.projectPoints(points_3d, rotation, translation,
                                             intrinsics, distortion)

    projected_points = np.array(projected_points_raw[0]).reshape(-1, 2)

    projected_points[projected_points[:, 0] < 0, 0] = 0.0
    projected_points[projected_points[:, 0] > image_size[1], 0] = image_size[1]
    projected_points[projected_points[:, 1] < 0, 1] = 0.0
    projected_points[projected_points[:, 1] > image_size[0], 1] = image_size[0]

    bounding_box = np.array([
        np.min(projected_points[:, 0]),
        np.min(projected_points[:, 1]),
        np.max(projected_points[:, 0]) - np.min(projected_points[:, 0]),
        np.max(projected_points[:, 1]) - np.min(projected_points[:, 1])
    ])

    return projected_points, bounding_box
