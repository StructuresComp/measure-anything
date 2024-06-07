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
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.3, help='Specify threshold. Line segments with the midpoint lower than the threshold will be identified. Ex) 0.5 corresponds to half of the image')
    parser.add_argument('--input_folder', type=str, required=True, help='Specify input folder of the binary image')
    parser.add_argument('--visualize', type=bool, default=False, help='If True, visualize the perpendicular line segments')
    args = parser.parse_args()

    for file in glob.glob(os.path.join(args.input_folder, '*.png')):
        # Define the stemSkeletonization object
        current_stem = stemSkeletonization(threshold=args.threshold, window=70, stride=15, image_file=file)
        current_stem.generate_line_segment_pairs(visualize=True, save_images=True)

        # Save the end coordinates of the line segments
        current_stem.save_coordinates(filename=Path(file).stem)

if __name__ == '__main__':
    main()