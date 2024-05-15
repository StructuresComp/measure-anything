# Run this script to obtain the pixel coordinates of the perpendicular line segments
# The processes are as follows:
# 1. Read binary image from input folder.
# 2. Apply preproccessing and preserve the binary component with the largest area.
# 3. Apply medial axis skeletonization to the binary image.
# 4. Visualize the perpendicular line segments.
# 5. Save end coordinates of the line segments.

import argparse
import glob
import os
from stemSkel import stemSkeletonization

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scaling_factor', type=int, default=100, help='Define scaling factor in percentage to scale image')
    parser.add_argument('--threshold', type=float, default=0.3, help='Specify threshold. Line segments with the midpoint lower than the threshold will be identified. Ex) 0.5 corresponds to half of the image')
    parser.add_argument('--input_folder', type=str, required=True, help='Specify input folder of the binary image')
    parser.add_argument('--visualize', type=bool, default=False, help='If True, visualize the perpendicular line segments')
    args = parser.parse_args()

    for file in glob.glob(os.path.join(args.input_folder, '*.png')):
        # Read binary image as grayscale image

        current_stem = stemSkeletonization(threshold=args.threshold, window=20, stride=10, image_file=file)
        current_stem.generate_line_segment_pairs(visualize=True, save_images=True)

if __name__ == '__main__':
    main()