"""Contains tools for stereo rectification and stereo matching."""

import yaml
import cv2
import numpy as np
import logging as log


class StereoMatchingParams(object):
    """Stereo matching algorithm parameters.

    Note:
        For more details on the attributes please consult OpenCV documentation.

    Attributes:
        d415_min_disparity (int): Minimum possible disparity value for d415
            camera.
        d415_num_disparities (int): Maximum disparity minus minimum disparity
            for d415 camera.
        d415_block_size (int): Matched block size for d415 camera.
        d415_p1 (int): The first parameter controlling the disparity
            smoothness for d415 camera.
        d415_p2 (int): The first parameter controlling the disparity
            smoothness for d415 camera.
        d435_min_disparity (int): Minimum possible disparity value for d435
            camera.
        d435_num_disparities (int): Maximum disparity minus minimum disparity
            for d435 camera.
        d435_block_size (int): Matched block size for d435 camera.
        d435_p1 (int): The first parameter controlling the disparity
            smoothness for d435 camera.
        d435_p2 (int): The first parameter controlling the disparity
            smoothness for d435 camera.
        disp_12_max_diff (int): Maximum allowed difference (in integer pixel
            units) in the left-right disparity check.
        pre_filter_cap (int): Truncation value for the prefiltered image
            pixels.
        uniqueness_ratio (int): Margin in percentage by which the best
            (minimum) computed cost function value should "win" the second best
            value to consider the found match correct.
        speckle_window_size (int): Maximum size of smooth disparity regions to
            consider their noise speckles and invalidate.
        speckle_range (int): Maximum disparity variation within each connected
            component.
        mode (int): Stereo algorithm to use. Check OpenCV documentation for
            available options.
        apply_bilateral_filter (bool): If set to True, bilateral filter will
            be applied to the generated depth image.
        bilateral_filter_size (int): Diameter of each pixel neighborhood that
            is used during filtering.
        bilateral_filter_sigma (float): Filter sigma in the color and
            coordinate space.
        apply_wls_filter (bool): If set to True, wls filter will be applied to
        the generated depth image.
        wls_filter_sigma_color (float): Parameter defining how sensitive the
            filtering process is to source image edges.
        wls_filter_lambda (float): Parameter defining the amount of
            regularization during filtering.
        wls_filter_radius (int): Defines the size of low-confidence regions
            around depth discontinuities.
        wls_filter_sigma_lrc_thresh (int): Threshold of disparity difference
            used in left-right-consistency check during confidence map
            computation.
        use_median_filter (bool): If set to True, median filter will be applied
            to the generated depth image.
        median_filter_size (int):

    """

    def __init__(self):
        """Construct the StereoMatchingParams class.

        Constructor initializes all the attributes with the default values.
        """
        log.debug("Initialized StereoMatchingParams with default values.")

        self.d415_min_disparity = 15
        self.d415_num_disparities = 272
        self.d415_block_size = 25
        self.d415_p1 = 150
        self.d415_p2 = 520
        self.d435_min_disparity = 15
        self.d435_num_disparities = 272
        self.d435_block_size = 25
        self.d435_p1 = 150
        self.d435_p2 = 520
        self.disp_12_max_diff = -1
        self.pre_filter_cap = 10
        self.uniqueness_ratio = 30
        self.speckle_window_size = 2048
        self.speckle_range = 1
        self.mode = cv2.StereoSGBM_MODE_HH4
        self.apply_bilateral_filter = False
        self.bilateral_filter_size = 21
        self.bilateral_filter_sigma = 30
        self.apply_wls_filter = False
        self.wls_filter_sigma_color = 0.8
        self.wls_filter_lambda = 100
        self.wls_filter_radius = 9
        self.wls_filter_sigma_lrc_thresh = 24
        self.use_median_filter = False
        self.median_filter_size = 11

    def read_from_yaml(self, yaml_file):
        """Read the stereo parameters from a yaml file.

        Args:
            yaml_file (str): Path to the yaml file containing the stereo
                parameters.
        """
        log.debug("Initialized StereoMatchingParams from a yaml file: " +
                  yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            stereo_params = yaml.load(file_pointer)

        self.d415_min_disparity = stereo_params['d415_min_disparity']
        self.d415_num_disparities = stereo_params['d415_num_disparities']
        self.d415_block_size = stereo_params['d415_block_size']
        self.d415_p1 = stereo_params['d415_p1']
        self.d415_p2 = stereo_params['d415_p2']
        self.d435_min_disparity = stereo_params['d435_min_disparity']
        self.d435_num_disparities = stereo_params['d435_num_disparities']
        self.d435_block_size = stereo_params['d435_block_size']
        self.d435_p1 = stereo_params['d435_p1']
        self.d435_p2 = stereo_params['d435_p2']
        self.disp_12_max_diff = stereo_params['disp_12_max_diff']
        self.pre_filter_cap = stereo_params['pre_filter_cap']
        self.uniqueness_ratio = stereo_params['uniqueness_ratio']
        self.speckle_window_size = stereo_params['speckle_window_size']
        self.speckle_range = stereo_params['speckle_range']
        self.mode = stereo_params['mode']
        self.apply_bilateral_filter = stereo_params['apply_bilateral_filter']
        self.bilateral_filter_size = stereo_params['bilateral_filter_size']
        self.bilateral_filter_sigma = stereo_params['bilateral_filter_sigma']
        self.apply_wls_filter = stereo_params['apply_wls_filter']
        self.wls_filter_sigma_color = stereo_params['wls_filter_sigma_color']
        self.wls_filter_lambda = stereo_params['wls_filter_lambda']
        self.wls_filter_radius = stereo_params['wls_filter_radius']
        self.wls_filter_sigma_lrc_thresh = stereo_params[
            'wls_filter_sigma_lrc_thresh']
        self.use_median_filter = stereo_params['use_median_filter']
        self.median_filter_size = stereo_params['median_filter_size']


def rectify_images(image_l, camera_matrix_l, dist_coeffs_l, image_r,
                   camera_matrix_r, dist_coeffs_r, extrinsics_r, extrinsics_t):
    """Rectifie and undisotort left and right stereo images.

    Function takes in left and right image, together with their intrinsic and
    extrinsic parameters and returns rectified and undistorted images and Q
    matrix that maps disparity image to 3D.

    Args:
        image_l (np.array): Left image.
        camera_matrix_l (np.array): 3x3 calibration matrix of the left image.
        dist_coeffs_l (np.array): Distortion coefficients of the left image.
        image_r (np.array): Right image.
        camera_matrix_r (np.array): 3x3 calibration matrix of the right image.
        dist_coeffs_r (np.array): Distortion coefficients of the right image.
        extrinsics_r (np.array): Extrinsics 3x3 rotation matrix.
        extrinsics_t (np.array): Extrinsics translation vector.

    Returns:
        rectified_l (np.array): Rectified and undistorted left image.
        rectified_r (np.array): Rectified and undistorted right image.
        disparity_to_depth_map (np.array): 4x4 perspective transformation
            matrix: [X Y Z W]^t = Q * [u v disp(u,v) 1]^t.
        rotation_matrix_left (np.array): 3x3 rotation matrix of the rectified
            left image.
        new_calibration_left (np.array): 3x4 new calibration matrix for left
            image.

    """
    log.debug("Rectifying stereo pair images.")

    (rotation_matrix_left, rotation_matrix_right, new_calibration_left,
     new_calibration_right, disparity_to_depth_map, valid_ROI_left,
     valid_ROI_right) = cv2.stereoRectify(
         camera_matrix_l,
         dist_coeffs_l,
         camera_matrix_r,
         dist_coeffs_r,
         image_l.shape[::-1],
         extrinsics_r,
         extrinsics_t,
         flags=cv2.CALIB_ZERO_DISPARITY,
         alpha=-1)

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        camera_matrix_l, dist_coeffs_l, rotation_matrix_left,
        new_calibration_left, image_l.shape[::-1], cv2.CV_16SC2)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        camera_matrix_r, dist_coeffs_r, rotation_matrix_right,
        new_calibration_right, image_r.shape[::-1], cv2.CV_16SC2)

    rectified_l = cv2.remap(image_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(image_r, map_r1, map_r2, cv2.INTER_LINEAR)

    log.debug("Rectification results: ")
    log.debug("rotation_matrix_left:\n" + str(rotation_matrix_left))
    log.debug("rotation_matrix_right:\n" + str(rotation_matrix_right))
    log.debug("new_calibration_left:\n" + str(new_calibration_left))
    log.debug("new_calibration_right:\n" + str(new_calibration_right))
    log.debug("disparity_to_depth_map:\n" + str(disparity_to_depth_map))
    log.debug("valid_ROI_left:\n" + str(valid_ROI_left))
    log.debug("valid_ROI_right:\n" + str(valid_ROI_right))

    return (rectified_l, rectified_r, disparity_to_depth_map,
            rotation_matrix_left, new_calibration_left)


def stereo_match(undistorted_rectified_l,
                 undistorted_rectified_r,
                 baseline,
                 focal_length,
                 stereo_params,
                 sensor_name,
                 scale=1000.0):
    """Perform stereo matching using semi global block matcher (SGBM).

    Args:
        undistorted_rectified_l (np.array): Undistorted and rectified left
            image.
        undistorted_rectified_r (np.array): Undistorted and rectified right
            image.
        baseline (float): Baseline between the left and right image in meters.
        focal_length (float): Focal length of the left and right camera
            (cameras with different x and y focal length are not supported).
        stereo_params (StereoMatchingParams): StereoMatchingParams class
            containing parameters for the SGBM algorithm.
        sensor_name (str): Name of the sensor.
        scale (float, optional): Scaling used to convert to uint depth image
            (1000 for converting m to mm). Defaults to 1000.0.

    Returns:
        depth_uint (np.array): Depth image represented as a 16-bit uint image.
        depth_float (np.array): Depth image represented as a 32-bit float
            image.
        disparity_float (np.array): Disparity image represented as a 32-bit
            float image.

    """
    log.debug("Performing stereo matching on undistorted, rectified images.")

    if sensor_name == "realsense_d415":
        stereo_matcher = cv2.StereoSGBM_create(
            stereo_params.d415_min_disparity,
            stereo_params.d415_num_disparities, stereo_params.d415_block_size,
            stereo_params.d415_p1, stereo_params.d415_p2,
            stereo_params.disp_12_max_diff, stereo_params.pre_filter_cap,
            stereo_params.uniqueness_ratio, stereo_params.speckle_window_size,
            stereo_params.speckle_range, stereo_params.mode)
    elif sensor_name == "realsense_d435":
        stereo_matcher = cv2.StereoSGBM_create(
            stereo_params.d435_min_disparity,
            stereo_params.d435_num_disparities, stereo_params.d435_block_size,
            stereo_params.d435_p1, stereo_params.d435_p2,
            stereo_params.disp_12_max_diff, stereo_params.pre_filter_cap,
            stereo_params.uniqueness_ratio, stereo_params.speckle_window_size,
            stereo_params.speckle_range, stereo_params.mode)
    else:
        log.error("\nUnknown sensor!")
        return

    if undistorted_rectified_l.dtype == 'uint8':
        uint8_undistorted_rectified_l = undistorted_rectified_l
    elif undistorted_rectified_l.dtype == 'uint16':
        uint8_undistorted_rectified_l = (
            undistorted_rectified_l / 255.0).astype('uint8')
    else:
        log.error("\nUnknown image type!")
        return

    if undistorted_rectified_r.dtype == 'uint8':
        uint8_undistorted_rectified_r = undistorted_rectified_r
    elif undistorted_rectified_r.dtype == 'uint16':
        uint8_undistorted_rectified_r = (
            undistorted_rectified_r / 255.0).astype('uint8')
    else:
        log.error("\nUnknown image type!")
        return

    log.debug("Performing stereo matching.")
    disparity = stereo_matcher.compute(uint8_undistorted_rectified_l,
                                       uint8_undistorted_rectified_r)

    if stereo_params.apply_wls_filter:
        if sensor_name == "realsense_d415":
            right_macher = cv2.StereoSGBM_create(
                -(stereo_params.d415_min_disparity +
                  stereo_params.d415_num_disparities) + 1,
                stereo_params.d415_num_disparities,
                stereo_params.d415_block_size, stereo_params.d415_p1,
                stereo_params.d415_p2, stereo_params.disp_12_max_diff,
                stereo_params.pre_filter_cap, stereo_params.uniqueness_ratio,
                stereo_params.speckle_window_size, stereo_params.speckle_range,
                stereo_params.mode)
        elif sensor_name == "realsense_d435":
            right_macher = cv2.StereoSGBM_create(
                -(stereo_params.d435_min_disparity +
                  stereo_params.d435_num_disparities) + 1,
                stereo_params.d435_num_disparities,
                stereo_params.d435_block_size, stereo_params.d435_p1,
                stereo_params.d435_p2, stereo_params.disp_12_max_diff,
                stereo_params.pre_filter_cap, stereo_params.uniqueness_ratio,
                stereo_params.speckle_window_size, stereo_params.speckle_range,
                stereo_params.mode)
        else:
            log.error("\nUnknown sensor!")
            return

        disparity_right = right_macher.compute(uint8_undistorted_rectified_r,
                                               uint8_undistorted_rectified_l)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_matcher)
        wls_filter.setLambda(stereo_params.wls_filter_lambda)
        wls_filter.setSigmaColor(stereo_params.wls_filter_sigma_color)
        wls_filter.setDepthDiscontinuityRadius(stereo_params.wls_filter_radius)
        wls_filter.setLRCthresh(stereo_params.wls_filter_sigma_lrc_thresh)
        wls_disparity = wls_filter.filter(
            disparity, uint8_undistorted_rectified_l, None, disparity_right,
            None, uint8_undistorted_rectified_r)

        confidence = wls_filter.getConfidenceMap()
        wls_disparity_float = -wls_disparity.astype(np.float32) / 16.0
        wls_depth = baseline * focal_length / wls_disparity_float
        wls_depth[confidence < 200] = float('nan')
        depth_float = wls_depth
    else:
        disparity_float = -disparity.astype(np.float32) / 16.0
        depth_float = baseline * focal_length / disparity_float
        depth_float[depth_float <= 0] = float('nan')
        depth_float[depth_float == np.amax(depth_float)] = float('nan')

    if stereo_params.apply_bilateral_filter:
        depth_float = cv2.bilateralFilter(depth_float,
                                          stereo_params.bilateral_filter_size,
                                          stereo_params.bilateral_filter_sigma,
                                          stereo_params.bilateral_filter_sigma)

    depth_uint = depth_float * scale
    depth_uint = depth_uint.astype(np.uint16)

    return depth_uint, depth_float, disparity_float
