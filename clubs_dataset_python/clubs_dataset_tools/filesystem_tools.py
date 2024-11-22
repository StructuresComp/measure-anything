"""Contains tools for proper data handling.

These include functions for reading image and CSV files, finding proper folders
and creating missing ones.
"""

import os
import cv2
# import libtiff
from PIL import Image
import glob
import json
import csv
import logging as log


def read_label_files(label_files, extension='.json'):
    """Read all the annotations from a list of file paths.

    Args:
        label_files (list(str)): List containing full label file paths.
        extension (str, optional): Extension of the label files.
            Defaults to '.json'.

    Returns:
        labels (list(dict)): List of dicts containing the annotations.

    """
    labels = []

    for file in label_files:
        if file.endswith(extension):
            log.debug("Loading " + file)
            with open(file) as f:
                json_data = json.load(f)
            labels.append(json_data)
        else:
            log.error("\nUnknown extension of the image file!")

    if len(labels) is 0:
        log.error("\nNo label files could be retrieved!\n" +
                  "List containing label file full paths:\n" + str(label_files))

    return labels


def read_images(image_files, image_extension='.png', image_type=cv2.CV_16UC1):
    """Read all the images from a list of file paths.

    Args:
        image_files (list(str)): List containing full image paths.
        image_extension (str, optional): Extension of the images to be used.
            Defaults to '.png'.
        image_type (int, optional): OpenCV image type, specifies the number of
            channels and type of data image contains. Defaults to cv2.CV_16UC1

    Returns:
        images (list(np.array)): List of numpy arrays containing the images.

    """
    images = []

    for file in image_files:
        if file.endswith(image_extension):
            log.debug("Loading " + file)
            if image_extension == '.png' or image_extension == '.jpg':
                image = cv2.imread(file, image_type)
                if image is not None:
                    images.append(image)
            elif image_extension == '.tiff':
                # tiff = libtiff.TIFF.open(file, mode='r')
                # image = tiff.read_image()
                # tiff.close()
                # images.append(image)
                with Image.open(file) as img:
                    image = img.copy()
                    images.append(image)
            else:
                log.error("\nUnknown extension of the image file!")

    if len(images) is 0:
        log.error("\nNo images could be retrieved!\n" +
                  "List containing image full paths:\n" + str(image_files))

    return images


def find_files_with_extension_in_folder(folder, extension='.png'):
    """Return filenames that are present in the folder that end with extension.

    Args:
        folder (str): Path to the desired folder.
        extension (str, optional): Extension to search for. Defaults to '.png'.

    Returns:
        images (list(str)): List containing image filenames.

    """
    images = []

    files = os.listdir(folder)

    for file in files:
        log.debug("Found " + file)
        if file.endswith(extension):
            log.debug("It has the right extension, adding it to the image list")
            images.append(file)

    return sorted(images)


def find_ir_image_folders(input_folder):
    """Return left and right IR image folders paths for Realsense sensors.

    Note:
        Path is relative to the input folder path.

    Args:
        input_folder (str): Path to specific object/box folder.

    Returns:
        d415_image_folders (list(str)): Realsense d415 root, IR left and
            right image folder path relative to the input_folder and sensor
            root folder.
        d435_image_folders (list(str)): Realsense d435 root, IR left and
            right image folder path relative to the input_folder and sensor
            root folder.

    """
    expected_number_of_folders = 3

    d415_image_folders = []
    d415_ir_l = '/realsense_d415/ir1_images'
    if os.path.isdir(input_folder + d415_ir_l):
        d415_image_folders.append(d415_ir_l)
    d415_ir_r = '/realsense_d415/ir2_images'
    if os.path.isdir(input_folder + d415_ir_r):
        d415_image_folders.append(d415_ir_r)
    d415_image_folders.append('/realsense_d415')
    if len(d415_image_folders) is not expected_number_of_folders:
        log.error("\nD415 ir folders could not be found!\n" + "Looking for:\n" +
                  str(input_folder + d415_ir_l) + "\n" +
                  str(input_folder + d415_ir_r))
        d415_image_folders = []
    log.debug("Found d415 folders: \n" + str(d415_image_folders))

    d435_image_folders = []
    d435_ir_l = '/realsense_d435/ir1_images'
    if os.path.isdir(input_folder + d435_ir_l):
        d435_image_folders.append(d435_ir_l)
    d435_ir_r = '/realsense_d435/ir2_images'
    if os.path.isdir(input_folder + d435_ir_r):
        d435_image_folders.append(d435_ir_r)
    d435_image_folders.append('/realsense_d435')
    if len(d435_image_folders) is not expected_number_of_folders:
        log.error("\nD435 ir folders could not be found!\n" + "Looking for:\n" +
                  str(input_folder + d435_ir_l) + "\n" +
                  str(input_folder + d435_ir_r))
        d435_image_folders = []
    log.debug("Found d435 folders: \n" + str(d435_image_folders))

    return d415_image_folders, d435_image_folders


def find_rgb_d_image_folders(input_folder,
                             use_stereo_depth=False,
                             use_registered_depth=False):
    """Return RGB and depth image folders paths for all the sensors.

    Note:
        Path is relative to the input folder path.

    Args:
        input_folder (str): Path to specific object/box folder.
        use_stereo_depth (bool, optional): If True, depth from stereo will be
            used. Defaults to False.
        use_registered_depth (bool, optional): If True, registered depth will
            be used. Defaults to False.

    Returns:
        ps_image_folders (list(str)): Primesense root, RGB and depth image
            folder path relative to the input_folder.
        d415_image_folders (list(str)): Realsense d415 root, RGB and depth
            image folder path relative to the input_folder.
        d435_image_folders (list(str)): Realsense d435 root, RGB and depth
            image folder path relative to the input_folder.

    """
    expected_number_of_folders = 3

    ps_image_folders = []
    d415_image_folders = []
    d435_image_folders = []
    ps_rgb = '/primesense/rgb_images'
    d415_rgb = '/realsense_d415/rgb_images'
    d435_rgb = '/realsense_d435/rgb_images'

    if use_registered_depth:
        ps_depth = '/primesense/registered_depth_images'
        if use_stereo_depth:
            d415_depth = '/realsense_d415/registered_stereo_depth_images'
            d435_depth = '/realsense_d435/registered_stereo_depth_images'
        else:
            d415_depth = '/realsense_d415/registered_depth_images'
            d435_depth = '/realsense_d435/registered_depth_images'
    else:
        ps_depth = '/primesense/depth_images'
        if use_stereo_depth:
            d415_depth = '/realsense_d415/stereo_depth_images'
            d435_depth = '/realsense_d435/stereo_depth_images'
        else:
            d415_depth = '/realsense_d415/depth_images'
            d435_depth = '/realsense_d435/depth_images'

    if os.path.isdir(input_folder + ps_rgb):
        ps_image_folders.append(ps_rgb)
    if os.path.isdir(input_folder + d415_rgb):
        d415_image_folders.append(d415_rgb)
    if os.path.isdir(input_folder + d435_rgb):
        d435_image_folders.append(d435_rgb)

    if os.path.isdir(input_folder + ps_depth):
        ps_image_folders.append(ps_depth)
    if os.path.isdir(input_folder + d415_depth):
        d415_image_folders.append(d415_depth)
    if os.path.isdir(input_folder + d435_depth):
        d435_image_folders.append(d435_depth)

    ps_image_folders.append('/primesense')
    d415_image_folders.append('/realsense_d415')
    d435_image_folders.append('/realsense_d435')

    if len(ps_image_folders) is not expected_number_of_folders:
        log.error("\nPS rgb and depth folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + ps_rgb) + "\n" +
                  str(input_folder + ps_depth))
        ps_image_folders = []
    log.debug("Found ps folders: \n" + str(ps_image_folders))

    if len(d415_image_folders) is not expected_number_of_folders:
        log.error("\nD415 rgb and depth folders could not be found!\n" +
                  "Looking for:\n" + str(input_folder + d415_rgb) + "\n" +
                  str(input_folder + d415_depth))
        d415_image_folders = []
    log.debug("Found d415 folders: \n" + str(d415_image_folders))

    if len(d435_image_folders) is not expected_number_of_folders:
        log.error("\nD435 rgb and depth folders could not be found!" +
                  "Looking for:\n" + str(input_folder + d435_rgb) + "\n" +
                  str(input_folder + d435_depth))
        d435_image_folders = []
    log.debug("Found d435 folders: \n" + str(d435_image_folders))

    return ps_image_folders, d415_image_folders, d435_image_folders


def find_all_folders(dataset_folder):
    """Return folder names for object and box scenes.

    Args:
        dataset_folder (str): Path to the dataset folder.

    Returns:
        folders_objects (list(str)): Paths of folders containing object
            scenes.
        folders_boxes (list(str)): Paths of folders containing box scenes.

    """
    folders_objects = glob.glob(dataset_folder + '/object_scenes/' +
                                '[0-9]' * 3 + '_' + '[0-9]' * 1 + '_' + '*')
    log.debug("Found following object scene folders: \n" + str(folders_objects))

    folders_boxes = glob.glob(dataset_folder + '/box_scenes/box' + '_' +
                              '[0-9]' * 3 + '_' + '[0-9]' * 3)
    log.debug("Found following box scene folders: \n" + str(folders_boxes))

    return folders_objects, folders_boxes


def compare_image_names(image_list1, image_list2):
    """Compare if the two image lists have the same names (timestamps).

    Args:
        image_list1 (list(str)): List of names from the first image set.
        image_list2 (list(str)): List of names from the second image set.

    Returns:
        timestamps (list(int)): Timestamps found in both image lists.

    """
    timestamps_list1 = [image[:10] for image in image_list1]
    timestamps_list2 = [image[:10] for image in image_list2]

    if len(set(timestamps_list1) & set(timestamps_list2)) == len(
            timestamps_list1):
        log.debug("Timestamp lists are equal.")
        return timestamps_list1

    log.error("\nTwo timestamp lists are not equal: " +
              str(len(timestamps_list1)) + " vs " + str(len(timestamps_list2)))

    return []


def create_stereo_depth_folder(sensor_folder):
    """Create a folder for stereo depth images, if it does not exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        stereo_depth_folder (str): Path to the created stereo depth folder.

    """
    stereo_depth_folder = sensor_folder + '/stereo_depth_images'

    if not os.path.exists(stereo_depth_folder):
        os.makedirs(stereo_depth_folder)

    log.debug("Created a new stereo depth folder: \n" + stereo_depth_folder)

    return stereo_depth_folder


def create_point_cloud_folder(sensor_folder):
    """Create a folder for point clouds, if it does not exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        point_cloud_folder (str): Path to the created point cloud folder.

    """
    point_cloud_folder = sensor_folder + '/point_clouds'

    if not os.path.exists(point_cloud_folder):
        os.makedirs(point_cloud_folder)

    log.debug("Created a new point_clouds folder: \n" + point_cloud_folder)

    return point_cloud_folder


def create_stereo_point_cloud_folder(sensor_folder):
    """Create a folder for point clouds genereated using the stereo depth
    images, if it does not exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        stereo_point_cloud_folder (str): Path to the created point cloud
            folder.

    """
    stereo_point_cloud_folder = sensor_folder + '/stereo_point_clouds'

    if not os.path.exists(stereo_point_cloud_folder):
        os.makedirs(stereo_point_cloud_folder)

    log.debug("Created a new stereo_point_clouds folder: \n" +
              stereo_point_cloud_folder)

    return stereo_point_cloud_folder


def create_depth_registered_folder(sensor_folder):
    """Create a folder for registered depth images, if it does not exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        depth_registered_folder (str): Path to the created depth registered
            folder.

    """
    depth_registered_folder = sensor_folder + '/registered_depth_images'

    if not os.path.exists(depth_registered_folder):
        os.makedirs(depth_registered_folder)

    log.debug("Created a new registered_depth_images folder: \n" +
              depth_registered_folder)

    return depth_registered_folder


def create_stereo_depth_registered_folder(sensor_folder):
    """Create a folder for registered stereo depth images, if it does not
    exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        stereo_depth_registered_folder (str): Path to the created stereo
            depth registered folder.

    """
    stereo_depth_registered_folder = (
        sensor_folder + '/registered_stereo_depth_images')

    if not os.path.exists(stereo_depth_registered_folder):
        os.makedirs(stereo_depth_registered_folder)

    log.debug("Created a new registered_stereo_depth_images folder: \n" +
              stereo_depth_registered_folder)

    return stereo_depth_registered_folder


def create_rectified_images_folder(sensor_folder):
    """Create a folder for rectified stereo images, if it does not exist.

    Args:
        sensor_folder (str): Path to the sensor folder.

    Returns:
        rectified_images_folder (str): Path to the created rectified images
            folder.

    """
    rectified_images_folder = sensor_folder + '/rectified_images'

    if not os.path.exists(rectified_images_folder):
        os.makedirs(rectified_images_folder)

    log.debug("Created a new rectified_images folder: \n" +
              rectified_images_folder)

    return rectified_images_folder


def read_from_csv_file(file_path):
    """Read a CSV file and returns a list of tuples where each tuple in the
    list contains one row of the file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        output (list(tuple(float))): List of tuples which contain rows of the
            CSV file.

    """
    output = []

    with open(file_path, 'rb') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in file_reader:
            output.append(map(float, row))

    return output


def save_to_csv_file(file_path, input):
    """Write a CSV file from a list of tuples, where each tuple is stored in
    one row.

    Args:
        file_path (str): Path to the CSV file.
        input (list(tuple(float))): List of numpy arrays which should be saved
            as rows in the CSV file.

    """
    with open(file_path, 'wb') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        file_writer.writerows(input)


def create_label_folders(input_folder):
    """Create a folder for RGB and depth image labels, if it does not exist.

    Args:
        input_folder (str): Path to specific object/box folder.

    Returns:
        realsense_d415_rgb_folder (str): Path to the created d415 RGB label
            folder.
        realsense_d415_depth_folder (str): Path to the created d415 depth
            label folder.
        realsense_d435_rgb_folder (str): Path to the created d435 RGB label
            folder.
        realsense_d435_depth_folder (str): Path to the created d435 depth
            label folder.
        primesense_rgb_folder (str): Path to the created ps RGB label
            folder.
        primesense_depth_folder (str): Path to the created ps depth
            label folder.
        chameleon_rgb_folder (str): Path to the created cham3 RGB label
            folder.

    """
    chameleon_labels_folder = (input_folder + 'chameleon3/labels')
    primesense_labels_folder = (input_folder + 'primesense/labels')
    realsense_d415_labels_folder = (input_folder + 'realsense_d415/labels')
    realsense_d435_labels_folder = (input_folder + 'realsense_d435/labels')

    if not os.path.exists(chameleon_labels_folder):
        os.makedirs(chameleon_labels_folder)
    if not os.path.exists(primesense_labels_folder):
        os.makedirs(primesense_labels_folder)
    if not os.path.exists(realsense_d415_labels_folder):
        os.makedirs(realsense_d415_labels_folder)
    if not os.path.exists(realsense_d435_labels_folder):
        os.makedirs(realsense_d435_labels_folder)

    chameleon_rgb_folder = (input_folder + 'chameleon3/labels/rgb_images')
    primesense_rgb_folder = (input_folder + 'primesense/labels/rgb_images')
    primesense_depth_folder = (input_folder + 'primesense/labels/depth_images')
    realsense_d415_rgb_folder = (
        input_folder + 'realsense_d415/labels/rgb_images')
    realsense_d415_depth_folder = (
        input_folder + 'realsense_d415/labels/depth_images')
    realsense_d435_rgb_folder = (
        input_folder + 'realsense_d435/labels/rgb_images')
    realsense_d435_depth_folder = (
        input_folder + 'realsense_d435/labels/depth_images')

    if not os.path.exists(chameleon_rgb_folder):
        os.makedirs(chameleon_rgb_folder)
    if not os.path.exists(primesense_rgb_folder):
        os.makedirs(primesense_rgb_folder)
    if not os.path.exists(primesense_depth_folder):
        os.makedirs(primesense_depth_folder)
    if not os.path.exists(realsense_d415_rgb_folder):
        os.makedirs(realsense_d415_rgb_folder)
    if not os.path.exists(realsense_d415_depth_folder):
        os.makedirs(realsense_d415_depth_folder)
    if not os.path.exists(realsense_d435_rgb_folder):
        os.makedirs(realsense_d435_rgb_folder)
    if not os.path.exists(realsense_d435_depth_folder):
        os.makedirs(realsense_d435_depth_folder)

    log.debug("Created a new chameleon3 rgb label folder: \n" +
              chameleon_rgb_folder)
    log.debug("Created a new primesense rgb label folder: \n" +
              primesense_rgb_folder)
    log.debug("Created a new primesense depth label folder: \n" +
              primesense_depth_folder)
    log.debug("Created a new realsense d415 rgb label folder: \n" +
              realsense_d415_rgb_folder)
    log.debug("Created a new realsense d415 depth label folder: \n" +
              realsense_d415_depth_folder)
    log.debug("Created a new realsense d435 rgb label folder: \n" +
              realsense_d435_rgb_folder)
    log.debug("Created a new realsense d435 depth label folder: \n" +
              realsense_d435_depth_folder)

    return (realsense_d415_rgb_folder, realsense_d415_depth_folder,
            realsense_d435_rgb_folder, realsense_d435_depth_folder,
            primesense_rgb_folder, primesense_depth_folder,
            chameleon_rgb_folder)
