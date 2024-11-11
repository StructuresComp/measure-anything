"""Contains common classes and functions for camera and image handling.

The CalibrationParams class handles camera paramters and allows loading params
from a yaml file. convert_depth_uint_to_float and convert_depth_float_to_uint
are two convinience functions for converting depth images to and from float and
uint16 type. SensorTransformations class contains transformations from a
specific camera to all the sensors in the scanning setup.
"""

import yaml
import numpy as np
import logging as log


class CalibrationParams(object):
    """Camera intrinsic and extrinsic parameters.

    Atributes:
        rgb_intrinsics (np.array: 3x3 RGB camera intrinsic parameters matrix.
        rgb_distortion_coeffs (np.array()): 1x5 RGB camera distortion
            coefficients (r1, r2, t1, t2, r3).
        hand_eye_transform (np.array): 4x4 homogeneous transformation between
            RGB camera and robot's end-effector.
        depth_intrinsics (np.array): 3x3 Depth camera intrinsic parameters
            matrix.
        depth_distortion_coeffs (np.array): 1x5 Depth camera distortion
            coefficients (r1, r2, t1, t2, r3).
        depth_extrinsics (np.array): 4x4 homogeneous transformation between
            RGB camera and Depth(IR1) camera.
        ir1_intrinsics (np.array): 3x3 IR1 camera intrinsic parameters matrix.
        ir1_distortion_coeffs (np.array): 1x5 IR1 camera distortion
            coefficients (r1, r2, t1, t2, r3).
        ir2_intrinsics (np.array): 3x3 IR2 camera intrinsic parameters matrix.
        ir2_distortion_coeffs (np.array): 1x5 IR2 camera distortion
            coefficients (r1, r2, t1, t2, r3).
        ir_extrinsics (np.array): 4x4 homogeneous transformation between
            IR1 camera and IR2 camera.
        extrinsics_r (np.array): 3x3 rotation matrix representing the
            rotation part of inverse ir_extrinsics.
        extrinsics_t (np.array): 3x1 translation vector representing the
            translation part of inverse ir_extrinsics.
        rgb_width (double): Width of RGB camera image.
        rgb_height (double): Height of RGB camera image.
        depth_width (double): Width of Depth camera image.
        depth_height (double): Height of Depth camera image.
        z_scaling (double): Scaling factor for the Depth image.
        depth_scale_mm (double): Units of the Depth image.

    """

    def __init__(self):
        """Construct the CalibrationParams class.

        The constructor for the CalibrationParams class initializes all the
        atributes of the class to either empty numpy array or zero values.
        """
        log.debug("Initialized an empty CalibrationParams class.")

        self.rgb_intrinsics = np.array([])
        self.rgb_distortion_coeffs = np.array([])
        self.hand_eye_transform = np.array([])
        self.depth_intrinsics = np.array([])
        self.depth_distortion_coeffs = np.array([])
        self.depth_extrinsics = np.array([])
        self.ir1_intrinsics = np.array([])
        self.ir1_distortion_coeffs = np.array([])
        self.ir2_intrinsics = np.array([])
        self.ir2_distortion_coeffs = np.array([])
        self.ir_extrinsics = np.array([])
        self.extrinsics_r = np.array([])
        self.extrinsics_t = np.array([])
        self.rgb_width = 0.0
        self.rgb_height = 0.0
        self.depth_width = 0.0
        self.depth_height = 0.0
        self.z_scaling = 0.0
        self.depth_scale = 0.0

    def read_from_yaml(self, yaml_file):
        """Read the calibration parameters from a yaml file.

        Args:
            yaml_file (str): Path to the yaml file containing the calibration
                parameters.

        """
        log.debug("Initialized CalibrationParams from a yaml file: " +
                  yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            calibration_params = yaml.load(file_pointer, Loader=yaml.SafeLoader)

        raw_rgb_intrinsics = calibration_params['rgb_intrinsics']
        self.rgb_intrinsics = np.array(
            np.reshape(
                raw_rgb_intrinsics['data'],
                (raw_rgb_intrinsics['rows'], raw_rgb_intrinsics['cols'])))
        raw_rgb_distortion_coeffs = calibration_params['rgb_distortion_coeffs']
        self.rgb_distortion_coeffs = np.array(raw_rgb_distortion_coeffs['data'])
        raw_hand_eye_transform = calibration_params['hand_eye_transform']
        self.hand_eye_transform = np.array(
            np.reshape(raw_hand_eye_transform['data'],
                       (raw_hand_eye_transform['rows'],
                        raw_hand_eye_transform['cols'])))
        raw_depth_intrinsics = calibration_params['depth_intrinsics']
        self.depth_intrinsics = np.array(
            np.reshape(
                raw_depth_intrinsics['data'],
                (raw_depth_intrinsics['rows'], raw_depth_intrinsics['cols'])))
        raw_depth_distortion_coeffs = calibration_params[
            'depth_distortion_coeffs']
        self.depth_distortion_coeffs = np.array(
            raw_depth_distortion_coeffs['data'])
        raw_depth_extrinsics = calibration_params['depth_extrinsics']
        self.depth_extrinsics = np.array(
            np.reshape(
                raw_depth_extrinsics['data'],
                (raw_depth_extrinsics['rows'], raw_depth_extrinsics['cols'])))
        raw_ir1_intrinsics = calibration_params['ir1_intrinsics']
        self.ir1_intrinsics = np.array(
            np.reshape(
                raw_ir1_intrinsics['data'],
                (raw_ir1_intrinsics['rows'], raw_ir1_intrinsics['cols'])))
        raw_ir1_distortion_coeffs = calibration_params['ir1_distortion_coeffs']
        self.ir1_distortion_coeffs = np.array(raw_ir1_distortion_coeffs['data'])
        raw_ir2_intrinsics = calibration_params['ir2_intrinsics']
        self.ir2_intrinsics = np.array(
            np.reshape(
                raw_ir2_intrinsics['data'],
                (raw_ir2_intrinsics['rows'], raw_ir2_intrinsics['cols'])))
        raw_ir2_distortion_coeffs = calibration_params['ir2_distortion_coeffs']
        self.ir2_distortion_coeffs = np.array(raw_ir2_distortion_coeffs['data'])
        raw_ir_extrinsics = calibration_params['ir_extrinsics']
        self.ir_extrinsics = np.array(
            np.reshape(raw_ir_extrinsics['data'],
                       (raw_ir_extrinsics['rows'], raw_ir_extrinsics['cols'])))

        inv_extrinsics = np.linalg.inv(self.ir_extrinsics)

        self.extrinsics_r = inv_extrinsics[:3, :3]
        self.extrinsics_t = inv_extrinsics[:3, 3]

        self.rgb_width = calibration_params['rgb_width']
        self.rgb_height = calibration_params['rgb_height']
        self.depth_width = calibration_params['depth_width']
        self.depth_height = calibration_params['depth_height']
        self.z_scaling = calibration_params['z_scaling']
        self.depth_scale = calibration_params['depth_scale']


class SensorTransformations(object):
    """Transformations to all the sensors.

    Attributes:
        d415_rgb (np.array): 4x4 homogeneous transformation to RealSense D415
            RGB camera.
        d415_depth (np.array): 4x4 homogeneous transformation to RealSense D415
            depth camera.
        d435_rgb (np.array): 4x4 homogeneous transformation to RealSense D435
            RGB camera.
        d435_depth (np.array): 4x4 homogeneous transformation to RealSense D415
            depth camera.
        ps_rgb (np.array): 4x4 homogeneous transformation to PrimeSense RGB
            camera.
        ps_depth (np.array): 4x4 homogeneous transformation to PrimeSense depth
            camera.
        cham_rgb (np.array): 4x4 homogeneous transformation to Chameleon3 RGB
            camera.

    """

    def __init__(self):
        """Construct the SensorTransformations class.

        The constructor for the SensorTransformations class initializes all the
        atributes of the class to an empty numpy array.
        """
        log.debug("Initialized an empty SensorTransformations class.")

        self.d415_rgb = np.array([])
        self.d415_depth = np.array([])
        self.d435_rgb = np.array([])
        self.d435_depth = np.array([])
        self.ps_rgb = np.array([])
        self.ps_depth = np.array([])
        self.cham_rgb = np.array([])

    def read_from_yaml(self, yaml_file):
        """Read the sensor transformations from a yaml file.

        Args:
            yaml_file (str): Path to the yaml file containing the sensor
                transformations.

        """
        log.debug("Initialized SensorTransformations from a yaml file: " +
                  yaml_file)

        with open(yaml_file, 'r') as file_pointer:
            transformations = yaml.load(file_pointer)

        raw_d415_rgb = transformations['d415_rgb']
        self.d415_rgb = np.array(
            np.reshape(raw_d415_rgb['data'],
                       (raw_d415_rgb['rows'], raw_d415_rgb['cols'])))
        raw_d415_depth = transformations['d415_depth']
        self.d415_depth = np.array(
            np.reshape(raw_d415_depth['data'],
                       (raw_d415_depth['rows'], raw_d415_depth['cols'])))
        raw_d435_rgb = transformations['d435_rgb']
        self.d435_rgb = np.array(
            np.reshape(raw_d435_rgb['data'],
                       (raw_d435_rgb['rows'], raw_d435_rgb['cols'])))
        raw_d435_depth = transformations['d435_depth']
        self.d435_depth = np.array(
            np.reshape(raw_d435_depth['data'],
                       (raw_d435_depth['rows'], raw_d435_depth['cols'])))
        raw_ps_rgb = transformations['ps_rgb']
        self.ps_rgb = np.array(
            np.reshape(raw_ps_rgb['data'],
                       (raw_ps_rgb['rows'], raw_ps_rgb['cols'])))
        raw_ps_depth = transformations['ps_depth']
        self.ps_depth = np.array(
            np.reshape(raw_ps_depth['data'],
                       (raw_ps_depth['rows'], raw_ps_depth['cols'])))
        raw_cham_rgb = transformations['cham_rgb']
        self.cham_rgb = np.array(
            np.reshape(raw_cham_rgb['data'],
                       (raw_cham_rgb['rows'], raw_cham_rgb['cols'])))


def convert_depth_uint_to_float(uint_depth_image,
                                z_scaling=1.0,
                                depth_scale=1.0):
    """Convert uint16 depth image to float.

    During conversion, the depth scale and z_scaling are taken into
    consideration.

    Args:
        uint_depth_image (np.array): Depth image of type uint16.
        z_scaling (float, optional): Correction for z values to correspond to
            true metric values. Defaults to 1.0.
        depth_scale (float, optional): Conversion factor for depth (e.g. 1
            means that value of 1000 in uint16 depth image corresponds to 1.0
            in float depth image and to 1m in real world). Defaults to 1.0.

    Returns:
        float_depth_image (np.array): Depth image of type float.

    """
    log.debug(("Converting uint depth to float and applying z_scaling and "
               "depth_scaling"))

    return (uint_depth_image / 1000.0 * depth_scale * z_scaling).astype('float')


def convert_depth_float_to_uint(float_depth_image, depth_scale=1.0):
    """Convert float depth image to uint16, also considering the depth scale.

    Args:
        float_depth_image (np.array): Depth image of type float.
        depth_scale (float, optional): Conversion factor for depth (e.g. 1
            means that value of 1000 in uint16 depth image corresponds to 1.0
            in float depth image and to 1m in real world). Defualts to 1.0.

    Returns:
        uint_depth_image (np.array) - Depth image of type uint16.

    """
    log.debug("Converting float depth to uint16 and applying depth_scaling")

    return (float_depth_image * 1000.0 / depth_scale).astype('uint16')
