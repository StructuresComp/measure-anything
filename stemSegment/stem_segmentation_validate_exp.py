import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import argparse
import os
import pudb

# set up CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', dest='input_file', required=True, help='Specify input SVO file')
parser.add_argument('--outdir', dest='outdir', required=True, help='Specify location of output directory')
parser.add_argument('--annotated_image', dest='annotated_image', required=True, help='Specify annotated image file')  # Added argument for annotated image
# get arguments
args = parser.parse_args()
input_file = args.input_file
outdir = args.outdir if args.outdir[-1] != '/' else args.outdir[:-1]
annotated_image_path = args.annotated_image

# returns image from input file
def get_image():
    if not os.path.isfile(input_file):
        print("Specified path isn't a file.")
        exit()
    image = cv2.imread(input_file)
    if image is None:
        print("Specified path isn't a valid image file.")
        exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# returns keypoints from image
def get_keypoints(image):
    model = YOLO('YOLO/best.pt')
    results = model(image)
    for result in results:
        keypoint_obj = result.keypoints
        keypoint_lists = np.array(keypoint_obj.xy.cpu())
    return keypoint_lists

# try to create output directory
def create_outdir():
    try:
        os.mkdir(outdir)
    except FileExistsError:
        print("output directory already exists.")
    except Exception:
        raise

# create segmentation mask for each set of keypoints
def create_mask(keypoint_lists, image, area_threshold=10000):
    if keypoint_lists.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)  # Return an empty mask if no keypoints are detected
    
    sam_checkpoint = "SAM/sam_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    union_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Initialize an empty mask for union of all masks

    input_label = np.array([1,1,1])
    for index, keypoint_list in enumerate(keypoint_lists):
            if keypoint_list.shape[0] > 0:
                first_keypoint = keypoint_list[0:]  # Select only the first keypoint
                mask, _, _ = predictor.predict(
                    point_coords=first_keypoint,
                    point_labels=input_label,  # Use the corresponding label for the first keypoint
                    mask_input=None,
                    multimask_output=False,
                )
                mask = mask.astype('uint8')
                mask = np.transpose(mask, axes=[1, 2, 0])
                mask = np.squeeze(mask, axis=2)  # Squeeze the mask to match dimensions
                union_mask = cv2.bitwise_or(union_mask, mask)  # Union of masks
    
    # Remove small areas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
    
    for i in range(1, num_labels):  # Skip the background label (0)
        if stats[i, cv2.CC_STAT_AREA] < area_threshold:
            union_mask[labels == i] = 0

    return union_mask

def process_annotated_image():
    if not os.path.isfile(annotated_image_path):
        print("Specified annotated image path isn't a file.")
        exit()
    annotated_image = cv2.imread(annotated_image_path)
    if annotated_image is None:
        print("Specified path isn't a valid image file.")
        exit()
    # The stem annotation is at rgb = (7, 7, 7)
    annotated_mask = cv2.inRange(annotated_image, (7, 7, 7), (7, 7, 7))
    return annotated_mask

def measure_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def overlay_masks(mask1, mask2, output_path):
    # Create an RGB image to overlay the masks
    overlay_image = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    
    intersection = np.logical_and(mask1, mask2)
    mask1_only = np.logical_and(mask1, np.logical_not(mask2))
    mask2_only = np.logical_and(mask2, np.logical_not(mask1))

    # Color the intersection area in white
    overlay_image[intersection] = [255, 255, 255]
    # Color the mask1-only area in red
    overlay_image[mask1_only] = [255, 0, 0]
    # Color the mask2-only area in blue
    overlay_image[mask2_only] = [0, 0, 255]

    # Save the overlay image
    cv2.imwrite(output_path, overlay_image)

def main():
    image = get_image()
    create_outdir()
    keypoint_lists = get_keypoints(image)
    union_mask = create_mask(keypoint_lists, image)
    
    annotated_mask = process_annotated_image()
    iou_score = measure_iou(union_mask, annotated_mask)
    
    print(f"IoU score: {iou_score}")

    # Overlay the masks and save the output
    overlay_output_path = os.path.join(outdir, 'overlay.png')
    overlay_masks(union_mask, annotated_mask, overlay_output_path)

if __name__ == '__main__':
    main()
