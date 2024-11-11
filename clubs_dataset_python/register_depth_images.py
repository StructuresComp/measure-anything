#!/usr/bin/env python
"""Executable for registering depth to corresponding RGB image."""

import argparse
import cv2
import logging as log
from tqdm import tqdm

from clubs_dataset_tools.filesystem_tools import (
    read_images, find_files_with_extension_in_folder, find_all_folders,
    find_rgb_d_image_folders, compare_image_names,
    create_depth_registered_folder, create_stereo_depth_registered_folder)
from clubs_dataset_tools.common import (CalibrationParams,
                                        convert_depth_uint_to_float)
from clubs_dataset_tools.image_registration import (register_depth_image)


def generate_registered_depth(scene_folder,
                              sensor_folder,
                              calib_params,
                              use_stereo_depth=False):
    """Generate a registered depth image.

    Args:
        scene_folder (str): Path to the scene folder.
        sensor_folder (list(str)): List containing folder names for RGB and
            depth image, as well as the sensor root folder.
        calib_params (CalibrationParams): Calibration parameters from the
            camera.
        use_stereo_depth (bool, optional): If set to True, stereo depth will be
            used and therefore generated stereo depth intrinsics instead of
            device depth intrinsics. Defaults to False.

    """
    images_rgb = find_files_with_extension_in_folder(scene_folder +
                                                     sensor_folder[0])
    images_depth = find_files_with_extension_in_folder(scene_folder +
                                                       sensor_folder[1])

    timestamps = compare_image_names(images_rgb, images_depth)

    if len(timestamps) != 0:
        image_paths_rgb = [
            scene_folder + sensor_folder[0] + '/' + image_rgb
            for image_rgb in images_rgb
        ]
        image_paths_depth = [
            scene_folder + sensor_folder[1] + '/' + image_depth
            for image_depth in images_depth
        ]

        rgb_images = read_images(
            image_paths_rgb, image_type=cv2.IMREAD_ANYCOLOR)
        depth_images = read_images(image_paths_depth, image_type=cv2.CV_16UC1)

        if use_stereo_depth and sensor_folder[2] != '/primesense':
            depth_registered_folder = create_stereo_depth_registered_folder(
                scene_folder + sensor_folder[2])
        else:
            depth_registered_folder = create_depth_registered_folder(
                scene_folder + sensor_folder[2])

        stereo_bar = tqdm(
            total=len(rgb_images), desc="PointCloud/DepthRegistered progress")
        for i in range(len(rgb_images)):
            float_depth_image = convert_depth_uint_to_float(
                depth_images[i], calib_params.z_scaling,
                calib_params.depth_scale)

            depth_registerd_path = (depth_registered_folder + '/' +
                                    timestamps[i] + '_registered_depth.png')

            float_depth_reg, uint_depth_reg = register_depth_image(
                float_depth_image, calib_params.rgb_intrinsics,
                calib_params.depth_intrinsics, calib_params.depth_extrinsics,
                (calib_params.rgb_height, calib_params.rgb_width),
                calib_params.depth_scale)

            cv2.imwrite(depth_registerd_path, uint_depth_reg)

            stereo_bar.update()
        stereo_bar.close()
    else:
        log.error("\nImage names are not consistent for rgb and depth image.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Generate point clouds from RGB and depth images. There are two "
            "different ways this function can be called. First one is by "
            "passing in the dataset root folder (flag --dataset_folder) which "
            "will create a new folder for each object/scene and each sensor "
            "(ps, d415 and d435), containing the generated point clouds. "
            "Second way is to pass object/box scene root folder "
            "(flag --scene_folder) which will do the same for that specific "
            "scene."))
    parser.add_argument(
        '--dataset_folder', type=str, help="Path to the dataset root folder.")
    parser.add_argument(
        '--scene_folder', type=str, help="Path to the scene root folder.")
    parser.add_argument(
        '--ps_calib_file',
        type=str,
        default='config/primesense.yaml',
        help=("Path to Primesense calibration yaml file. Defaults to "
              "config/primesense.yaml"))
    parser.add_argument(
        '--d415_calib_file',
        type=str,
        default='config/realsense_d415_device_depth.yaml',
        help=("Path to RealSense D415 calibration yaml file. Defaults to "
              "config/realsense_d415_device_depth.yaml"))
    parser.add_argument(
        '--d435_calib_file',
        type=str,
        default='config/realsense_d435_device_depth.yaml',
        help=("Path to RealSense D435 calibration yaml file. Defaults to "
              "config/realsense_d435_device_depth.yaml"))
    parser.add_argument(
        '--use_only_boxes',
        action='store_true',
        help=("If this flag is set, depth from stereo will only be computed "
              "for the box scenes."))
    parser.add_argument(
        '--use_stereo_depth',
        action='store_true',
        help=("If this flag is set, depth from stereo will be used "
              "for cloud generation. Make sure to pass in the correct "
              "calibration file and that stereo depth images exist."))
    parser.add_argument(
        '--log',
        type=str,
        default='CRITICAL',
        help=("Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
              "Defaults to CRITICAL."))
    args = parser.parse_args()

    numeric_level = getattr(log, args.log.upper(), None)
    log.basicConfig(level=numeric_level)
    log.debug("Setting log verbosity to " + args.log)

    used_scenes = []

    calib_params = CalibrationParams()

    ps_ir_x_offset_pixels = -3
    ps_ir_y_offset_pixels = -3
    if args.use_stereo_depth:
        d415_depth_x_offset_pixels = 0
        d415_depth_y_offset_pixels = 0
        d435_depth_x_offset_pixels = 0
        d435_depth_y_offset_pixels = 0
    else:
        d415_depth_x_offset_pixels = 4
        d415_depth_y_offset_pixels = 0
        d435_depth_x_offset_pixels = -3
        d435_depth_y_offset_pixels = -1

    if args.dataset_folder is not None:
        log.debug("Received dataset_folder.")
        object_scenes, box_scenes = find_all_folders(args.dataset_folder)

        if args.use_only_boxes is True:
            log.debug("Processing only box scenes.")
            used_scenes = box_scenes
        else:
            log.debug("Processing both box and object scenes.")
            used_scenes = object_scenes + box_scenes

        progress_bar = tqdm(total=len(used_scenes) * 3, desc="Overall Progress")
        for i in range(len(used_scenes)):
            scene = used_scenes[i]
            log.debug("Processing " + str(scene))

            ps_folder, d415_folder, d435_folder = find_rgb_d_image_folders(
                scene, args.use_stereo_depth)

            calib_params.read_from_yaml(args.ps_calib_file)
            calib_params.depth_intrinsics[0, 2] += ps_ir_x_offset_pixels
            calib_params.depth_intrinsics[1, 2] += ps_ir_y_offset_pixels
            if ps_folder != []:
                generate_registered_depth(scene, ps_folder, calib_params)
            progress_bar.update()

            calib_params.read_from_yaml(args.d415_calib_file)
            calib_params.depth_intrinsics[0, 2] += d415_depth_x_offset_pixels
            calib_params.depth_intrinsics[1, 2] += d415_depth_y_offset_pixels
            if d415_folder != []:
                generate_registered_depth(scene, d415_folder, calib_params,
                                          args.use_stereo_depth)
            progress_bar.update()

            calib_params.read_from_yaml(args.d435_calib_file)
            calib_params.depth_intrinsics[0, 2] += d435_depth_x_offset_pixels
            calib_params.depth_intrinsics[1, 2] += d435_depth_y_offset_pixels
            if d435_folder != []:
                generate_registered_depth(scene, d435_folder, calib_params,
                                          args.use_stereo_depth)
            progress_bar.update()
        progress_bar.close()
    elif args.scene_folder is not None:
        scene = args.scene_folder
        log.debug("Processing single scene " + str(scene))

        ps_folder, d415_folder, d435_folder = find_rgb_d_image_folders(
            scene, args.use_stereo_depth)

        calib_params.read_from_yaml(args.ps_calib_file)
        calib_params.depth_intrinsics[0, 2] += ps_ir_x_offset_pixels
        calib_params.depth_intrinsics[1, 2] += ps_ir_y_offset_pixels
        if ps_folder != []:
            generate_registered_depth(scene, ps_folder, calib_params)

        calib_params.read_from_yaml(args.d415_calib_file)
        calib_params.depth_intrinsics[0, 2] += d415_depth_x_offset_pixels
        calib_params.depth_intrinsics[1, 2] += d415_depth_y_offset_pixels
        if d415_folder != []:
            generate_registered_depth(scene, d415_folder, calib_params,
                                      args.use_stereo_depth)

        calib_params.read_from_yaml(args.d435_calib_file)
        calib_params.depth_intrinsics[0, 2] += d435_depth_x_offset_pixels
        calib_params.depth_intrinsics[1, 2] += d435_depth_y_offset_pixels
        if d435_folder != []:
            generate_registered_depth(scene, d435_folder, calib_params,
                                      args.use_stereo_depth)
    else:
        parser.print_help()
