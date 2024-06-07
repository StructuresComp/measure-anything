import numpy as np
import pyzed.sl as sl
import cv2

class stemZed:
    def __init__(self, svo_path, pixel_pairs, frame, binary_mask, filename):
        self.svo_path = svo_path
        self.pixel_pairs = pixel_pairs
        self.frame = frame
        self.binary_mask = binary_mask
        self.filename = filename
        self.initial_diameter_readings = len(self.pixel_pairs)
        self.valid_diameter_readings = 0
        self.median = None


    def calculate_diameters(self, visualize=False, save_images=False):
        # Create a ZED camera object
        zed = sl.Camera()

        # Initialize the ZED camera
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.svo_path)
        init_params.coordinate_units = sl.UNIT.CENTIMETER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Error opening ZED camera")
            return

        runtime_parameters = sl.RuntimeParameters()
        RGB = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        frame_count = 0
        while True:
            # Grab a new frame
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1

                # Process only the specified frames
                if frame_count == self.frame:
                    zed.retrieve_image(RGB, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                    # Convert image to OpenCV format
                    image_ocv = RGB.get_data()
                    depth_ocv = depth.get_data()

                    # cv2.imshow("RGB", image_ocv)
                    # cv2.imshow("Depth", depth_ocv)
                    # cv2.waitKey(0)


                    image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
                    white_mask = np.zeros_like(image_rgb)
                    white_mask[self.binary_mask == 1] = [255, 255, 255]
                    overlay = cv2.addWeighted(image_rgb, 0.7, white_mask, 0.3, 0)

                    # Loop through each pair of pixel coordinates
                    valid_diameters = []

                    for pair in self.pixel_pairs:
                        err_code, point1_3d, point2_3d = self._find_valid_coordinates(point_cloud, pair[:2], pair[2:])

                        if err_code == 1:
                            # Compute the Euclidean distance between the two points
                            distance = np.linalg.norm(np.array([point1_3d[0] - point2_3d[0], point1_3d[1] - point2_3d[1], point1_3d[2] - point2_3d[2]]))
                            # print(f"Distance between points {pair[0]} and {pair[1]}: {distance} centimeters")
                            self.valid_diameter_readings += 1
                            valid_diameters.append(distance)

                            overlay = cv2.line(overlay, (int(pair[1]), int(pair[0])), (int(pair[3]), int(pair[2])), (0, 0, 255), 2)
                                # Visualize the perpendicular line segments

                    print(f'Filename: {self.filename}')
                    print(f'Number of valid diameter readings: {self.valid_diameter_readings}/{self.initial_diameter_readings}')
                    self.median = np.median(np.array(valid_diameters))
                    print(f'Median diameter: {self.median} cm')
                    if visualize:
                        cv2.imshow("Overlayed", overlay)
                        cv2.waitKey(0)

                    if save_images:
                        # Save the images
                        # cv2.imwrite(f"./output_main_zed/{self.filename}_RGB.png", image_rgb)
                        # cv2.imwrite(f"./output_main_zed/{self.filename}_depth.png", depth_ocv)
                        cv2.imwrite(f"./output_main_zed/{self.filename}_overlay.png", overlay)

                    zed.close()
                    break
        return

    def _find_valid_coordinates(self, point_cloud, point1, point2, distance_threshold=0.1):
        def is_valid_point(x, y):
            err, point = point_cloud.get_value(x, y)
            if not np.isnan(point[0]):
                return point
            return None

        # Get the initial points
        point1_3d = is_valid_point(int(point1[1]), int(point1[0]))
        point2_3d = is_valid_point(int(point2[1]), int(point2[0]))

        # If both points are valid, return them with err_code 1
        if point1_3d is not None and point2_3d is not None:
            return 1, point1_3d, point2_3d

        # Calculate the line length
        line_length = np.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)
        max_distance = distance_threshold * line_length  # max_distance percentage of the line length

        # Search along the line for valid points
        num_steps = int(line_length)
        for step in range(1, num_steps):
            # Calculate the interpolation factor
            t = step / num_steps

            # Interpolate the coordinates for point1
            if point1 is None:
                x_new1 = int(point1[1] + t * (point2[1] - point1[1]))
                y_new1 = int(point1[0] + t * (point2[0] - point1[0]))
                point1_3d = is_valid_point(x_new1, y_new1)
                if point1_3d is not None:
                    continue

            # Interpolate the coordinates for point2
            if point2_3d is None:
                x_new2 = int(point2[1] - t * (point2[1] - point1[1]))
                y_new2 = int(point2[0] - t * (point2[0] - point1[0]))
                point2_3d = is_valid_point(x_new2, y_new2)
                if point2_3d is not None:
                    continue

            # Calculate the distance traveled
            distance_traveled = t * line_length
            if distance_traveled > max_distance:
                break

        err_code = 1 if point1_3d is not None and point2_3d is not None else 0
        return err_code, point1_3d, point2_3d

