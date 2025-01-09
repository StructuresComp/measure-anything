"""Contains tools for point cloud generation.

Function save_colored_point_cloud_to_ply takes in an RGB and depth image,
creates a point cloud and saves it as a PLY file.
"""

import cv2
import numpy as np
import logging as log
import pudb

ply_header = '''ply
format ascii 1.0
element vertex %(n_points)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element camera 1
property int viewportx
property int viewporty
end_header
'''

point_ply = '%(x)f %(y)f %(z)f %(r)d %(g)d %(b)d\n'

end_ply = '%(width)d %(height)d\n'

DISTANCE_LOWER_LIMIT = 0.0
DISTANCE_UPPER_LIMIT = 5.0

def depth_to_3d(depth_image, intrinsics):
    # Create a meshgrid of image coordinates
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # Extract focal lengths and principal points from the intrinsics matrix
    # fx, fy = intrinsics[1, 1], intrinsics[0, 0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Calculate the real-world coordinates
    y = (j - cy) * depth_image / fx
    x = (i - cx) * depth_image / fy
    z = depth_image
    
    # Stack into a 3D point cloud
    points_3d = np.dstack((x, y, z))
    return points_3d


def save_colored_point_cloud_to_ply(rgb_image,
                                    depth_image,
                                    rgb_intrinsics,
                                    rgb_distortion,
                                    depth_intrinsics,
                                    extrinsics,
                                    cloud_path,
                                    use_registered_depth=False):
    """Create a point cloud, from RGB-D frame, and save it as a PLY file.

    Note:
        Input depth image should be in meters.

    Args:
        rgb_image (np.array): Input RGB image.
        depth_image (np.array): Input float depth image in meters.
        rgb_intrinsics (np.array): Intrinsic parameters of the RGB camera.
        rgb_distortion (np.array): Distortion parameters of the RGB camera.
        depth_intrinsics (np.array): Intrinsic parameters of the depth camera.
        extrinsics (np.array): Transformation between RGB and depth camera.
        rgb_shape (tuple(int)): Image size of the RGB image (rows, columns).
        cloud_path (str): Path where to store the point cloud, including
            file name and extension.
        use_registered_depth (bool, optional): If True, registered depth images
            will be used and therefore the resulting point cloud will be
            organized in the order of the RGB image. Defaults to False.

    """
    log.debug("Generating a colored point cloud from an RGB-D frame.")

    if use_registered_depth:
        log.debug(("Using depth_registered image, therefore the resulting "
                   "point cloud is organized in the order of the rgb image."
                   "NOTE: Make sure that the input depth_image is "
                   "registered!"))

        # depth_points_3d = cv2.rgbd.depthTo3d(depth_image, rgb_intrinsics)
        depth_points_3d = depth_to_3d(depth_image, rgb_intrinsics)

        n_rows, n_cols, n_coord = np.shape(depth_points_3d)

        with open(cloud_path, 'wb') as ply_file:
            ply_file.write(
                (ply_header % dict(n_points=n_rows * n_cols)).encode('utf-8'))

            for i in range(n_rows):
                for j in range(n_cols):
                    point_x = depth_points_3d[i, j, 0]
                    point_y = depth_points_3d[i, j, 1]
                    point_z = depth_points_3d[i, j, 2]

                    point_b = rgb_image[i, j, 0]
                    point_g = rgb_image[i, j, 1]
                    point_r = rgb_image[i, j, 2]

                    if (point_z > DISTANCE_LOWER_LIMIT and
                            point_z < DISTANCE_UPPER_LIMIT):
                        ply_file.write((point_ply % dict(
                            x=point_x,
                            y=point_y,
                            z=point_z,
                            r=point_r,
                            g=point_g,
                            b=point_b)).encode('utf-8'))
                    else:
                        ply_file.write((point_ply % dict(
                            x=0.0, y=0.0, z=0.0, r=0, g=0,
                            b=0)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

    else:
        log.debug(("Using unregistered depth image, therefore the resulting "
                   "point cloud is organized in the order of the "
                   "depth image."))

        # depth_points_3d = cv2.rgbd.depthTo3d(depth_image, depth_intrinsics)
        depth_points_3d = depth_to_3d(depth_image, depth_intrinsics)

        depth_points_in_rgb_frame = cv2.perspectiveTransform(
            depth_points_3d, extrinsics)

        n_rows, n_cols, n_coord = np.shape(depth_points_in_rgb_frame)

        fx = rgb_intrinsics[0, 0]
        fy = rgb_intrinsics[1, 1]
        cx = rgb_intrinsics[0, 2]
        cy = rgb_intrinsics[1, 2]

        with open(cloud_path, 'wb') as ply_file:
            ply_file.write(
                (ply_header % dict(n_points=n_rows * n_cols)).encode('utf-8'))

            for i in range(n_rows):
                for j in range(n_cols):
                    point_x = depth_points_in_rgb_frame[i, j, 0]
                    point_y = depth_points_in_rgb_frame[i, j, 1]
                    point_z = depth_points_in_rgb_frame[i, j, 2]

                    height, width, channels = rgb_image.shape
                    if (point_z > DISTANCE_LOWER_LIMIT and
                            point_z < DISTANCE_UPPER_LIMIT):
                        u = int(fx * point_x / point_z + cx)
                        v = int(fy * point_y / point_z + cy)

                        if (u >= 0 and u < width and v >= 0 and v < height):
                            point_b = rgb_image[v, u, 0]
                            point_g = rgb_image[v, u, 1]
                            point_r = rgb_image[v, u, 2]

                            ply_file.write((point_ply % dict(
                                x=point_x,
                                y=point_y,
                                z=point_z,
                                r=point_r,
                                g=point_g,
                                b=point_b)).encode('utf-8'))
                        else:
                            ply_file.write((point_ply % dict(
                                x=0.0, y=0.0, z=0.0, r=0, g=0,
                                b=0)).encode('utf-8'))
                    else:
                        ply_file.write((point_ply % dict(
                            x=0.0, y=0.0, z=0.0, r=0, g=0,
                            b=0)).encode('utf-8'))

            ply_file.write(
                (end_ply % dict(width=n_cols, height=n_rows)).encode('utf-8'))

    log.debug('Finished writing the file: ' + cloud_path)
