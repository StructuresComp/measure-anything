# Description: Main script to run the stemZed class for measuring the stem diameter of a plant in 3D space

# Input:
# 1) .svo file
# 2) Directory of the line segment coordinates (.npy) files
# 3) Directory of the binary mask (.npy) files

# Output:
# 1) Median stem measurement for each mask


import argparse
import glob
import os
import numpy as np
from stemZ import stemZed
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo_path', type=str, required=True, help='Specify path of the .svo file')
    parser.add_argument('--line_dir', type=str, required=True, help='Specify input directory of the segment coordinates (.npy) files')
    parser.add_argument('--mask_dir', type=str, required=True, help='Specify input directory of the binary mask (.npy) files')
    parser.add_argument('--visualize', type=bool, default=False, help='If True, visualize the perpendicular line segments')
    parser.add_argument('--save_visualization', type=bool, default=False, help='If True, save the visualization')

    args = parser.parse_args()

    for file in glob.glob(os.path.join(args.line_dir, '*.npy')):
        # Load the pixel pairs and binary mask: They should have the same filenames
        pixel_pairs = np.load(file)
        binary_mask = np.load(os.path.join(args.mask_dir, Path(file).stem + '.npy'))
        # Frame number is hardcoded for demonstration purposes
        stemZed_obj = stemZed(svo_path=args.svo_path, pixel_pairs=pixel_pairs, frame=258, binary_mask=binary_mask, filename=Path(file).stem)
        stemZed_obj.calculate_diameters(visualize=args.visualize, save_images=args.save_visualization)

if __name__ == '__main__':
    main()