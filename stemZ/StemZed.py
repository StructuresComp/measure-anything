import numpy as np
import pyzed.sl as sl


class StemZed:
    def __init__(self, zed):
        self.zed = zed
        self.depth = sl.Mat()
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

        # Get camera intrinsics
        calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        self.fx, self.fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
        self.cx, self.cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy

        # # Distortion parameters
        # Distortion factor : [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
        self.k1, self.k2, self.p1, self.p2, self.k3 = calibration_params.left_cam.disto[0:5]

    def _undistort_point(self, x, y):
        """Compensate for lens distortion using the camera's intrinsic parameters."""

        # Normalize coordinates
        x_norm = (x - self.cx) / self.fx
        y_norm = (y - self.cy) / self.fy

        # Radial distortion
        r2 = x_norm ** 2 + y_norm ** 2
        radial_dist = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3

        # Tangential distortion
        x_dist = x_norm * radial_dist + 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm ** 2)
        y_dist = y_norm * radial_dist + self.p1 * (r2 + 2 * y_norm ** 2) + 2 * self.p2 * x_norm * y_norm

        # Return undistorted pixel coordinates
        x_undistorted = x_dist * self.fx + self.cx
        y_undistorted = y_dist * self.fy + self.cy
        return x_undistorted, y_undistorted

    def calculate_3d_distance(self, line_segment_coordinates):
        # Array to store distances
        diameters = np.zeros(len(line_segment_coordinates))

        for i, (y1, x1, y2, x2) in enumerate(line_segment_coordinates):
            # Calculate midpoint coordinates
            x_m = int((x1 + x2) / 2)
            y_m = int((y1 + y2) / 2)

            # Get depth at midpoint
            _, midpoint_depth = self.depth.get_value(x_m, y_m)

            # Skip if depth is invalid (e.g., out of range)
            if midpoint_depth <= 0:
                diameters[i] = np.nan  # or set to 0 or another value indicating invalid depth
                continue

            # Undistort the endpoints
            x1_ud, y1_ud = self._undistort_point(x1, y1)
            x2_ud, y2_ud = self._undistort_point(x2, y2)

            # Triangulate the 3D points of (x1, y1) and (x2, y2) using midpoint depth
            z = midpoint_depth
            x1_3d = (x1_ud - self.cx) * z / self.fx
            y1_3d = (y1_ud - self.cy) * z / self.fy
            x2_3d = (x2_ud - self.cx) * z / self.fx
            y2_3d = (y2_ud - self.cy) * z / self.fy

            # 3D coordinates of endpoints
            point1_3d = np.array([x1_3d, y1_3d, z])
            point2_3d = np.array([x2_3d, y2_3d, z])

            # Calculate Euclidean distance between the 3D endpoints
            diameters[i] = np.linalg.norm(point1_3d - point2_3d)

        return diameters
