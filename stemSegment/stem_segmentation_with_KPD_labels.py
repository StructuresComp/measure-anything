import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import argparse
import os
from tqdm import tqdm
import pudb

# set up CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', dest='image_directory', required=True, help='Specify input image directory')
parser.add_argument('--labels_directory', dest='labels_directory', required=True, help='Specify labels directory')
parser.add_argument('--outdir', dest='outdir', required=True, help='Specify location of output directory')
# get arguments
args = parser.parse_args()
image_directory = args.image_directory
labels_directory = args.labels_directory
outdir = args.outdir if args.outdir[-1] != '/' else args.outdir[:-1]

# create output directory if it doesn't exist
def create_outdir():
    try:
        os.mkdir(outdir)
    except FileExistsError:
        print("Output directory already exists.")
    except Exception as e:
        raise e

# returns image from input file
def get_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Specified path isn't a valid image file: {image_path}")
        exit()

    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# returns keypoints from label file
def get_keypoints(label_path, image_width, image_height):
    keypoint_lists = []
    
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Assuming the format is <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> <px3> <py3>
            keypoints = np.array(parts[5:], dtype=float).reshape(-1, 2)
            
            # Scale keypoints from [0,1] to actual pixel dimensions
            keypoints[:, 0] *= image_width
            keypoints[:, 1] *= image_height
            
            keypoint_lists.append(keypoints)
    
    return keypoint_lists

# create segmentation mask for each set of keypoints
def create_mask(keypoint_lists, image, base_filename):
    # edge case, no keypoints detected for image
    if len(keypoint_lists) == 0:
        return
    
    # load SAM predictor
    sam_checkpoint = "SAM/sam_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # set image for SAM predictor
    predictor.set_image(image)

    # predict mask for each set of keypoints
    input_label = np.array([1] * len(keypoint_lists[0]))
    for index, keypoint_list in enumerate(keypoint_lists):
        mask, _, _ = predictor.predict(
            point_coords = keypoint_list,
            point_labels=input_label,
            mask_input=None,
            multimask_output=False,
        )

        # mask out pixels not in stem
        mask = mask.astype('uint8')
        mask = np.transpose(mask, axes=[1,2,0])

        # Remove small areas: Sort the areas and take only top 3
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]  # Skip background label 0
        # areas.sort(key=lambda x: x[1], reverse=True)  # Sort by area size in descending order

        # top_3_labels = [area[0] for area in areas[:3]]  # Get the labels of the top 3 largest areas

        # # Initialize a new mask with zeros
        # top_3_mask = np.zeros_like(mask)

        # # Combined Step 2 and Step 3 - Retain areas that meet criteria and remove others
        # area_threshold = 2000  # Set your desired area threshold here
        # for i in top_3_labels:
        #     if stats[i, cv2.CC_STAT_AREA] < area_threshold:
        #         continue
            
        #     # Check if any keypoint is inside this area
        #     area_has_keypoint = False
        #     for keypoint in keypoint_list:
        #         x, y = int(keypoint[0]), int(keypoint[1])
        #         if labels[y, x] == i:  # If the keypoint is within the area
        #             area_has_keypoint = True
        #             break

        #     if area_has_keypoint:
        #         top_3_mask[labels == i] = 1  # Retain this area in the new mask

        top_3_mask = mask
        
        # Create overlay mask
        out = cv2.bitwise_and(image, image, mask=top_3_mask)

        # Draw keypoints on the overlay mask
        for keypoint in keypoint_list:
            x, y = keypoint
            cv2.circle(out, (int(x), int(y)), 25, (0, 0, 255), -1)  # Draw a red circle with larger radius

        # Save the overlay mask
        overlay_mask_path = f'{outdir}/{base_filename}_overlay_mask{index:02}.jpg'
        cv2.imwrite(overlay_mask_path, out)

        # Save the binary mask
        binary_mask_path = f'{outdir}/{base_filename}_binary_mask{index:02}.jpg'
        cv2.imwrite(binary_mask_path, top_3_mask*255)
        
def main():
    # create output directory
    create_outdir()

    # get list of images in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.JPG')]

    # iterate through each image and its respective label with progress tracking
    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        base_filename = os.path.splitext(filename)[0]
        image_path = os.path.join(image_directory, filename)
        label_path = os.path.join(labels_directory, base_filename + '.txt')
        
        # check if label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found for image: {filename}")
            continue

        # get image
        image = get_image(image_path)
        image_height, image_width, _ = image.shape

        # get keypoints from label file and scale them
        keypoint_lists = get_keypoints(label_path, image_width, image_height)

        # extract and save segment mask for each set of keypoints
        create_mask(keypoint_lists, image, base_filename)

if __name__ == '__main__':
    main()