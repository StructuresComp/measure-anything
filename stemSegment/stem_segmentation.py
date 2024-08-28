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
# get arguments
args = parser.parse_args()
input_file = args.input_file
outdir = args.outdir if args.outdir[-1] != '/' else args.outdir[:-1]

# returns image from input file
def get_image():
    # if input file not valid, immediately exit
    if not os.path.isfile(input_file):
        print("Specified path isn't a file.")
        exit()

    # extract image from input file
    image = cv2.imread(input_file)
    if image is None:
        print("Specified path isn't a valid image file.")
        exit()

    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# returns keypoints from image
def get_keypoints(image):
    model = YOLO('YOLO/best.pt')
    results = model(image)
    # grab keypoints from list of result objects
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
def create_mask(keypoint_lists, image):
    # edge case, no keypoints detected for image
    if keypoint_lists.size == 0:
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
    input_label = np.array([1,1,1])
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
        out = cv2.bitwise_and(image, image, mask=mask)

        # Draw keypoints on the output image
        for keypoint in keypoint_list:
            x, y = keypoint
            cv2.circle(out, (int(x), int(y)), 25, (0, 0, 255), -1)  # Draw a red circle with larger radius

        # write final image
        final_image_path = outdir + f'/mask{index}.jpg'
        cv2.imwrite(final_image_path, out)
        
def main():
    # get image from input file
    image = get_image()

    # create output directory
    create_outdir()

    # get keypoints from image
    keypoint_lists = get_keypoints(image)
    
    # extract and save segment mask for each set of keypoints
    create_mask(keypoint_lists, image)

if __name__ == '__main__':
    main()