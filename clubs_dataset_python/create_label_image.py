#!/usr/bin/env python
"""Executable for generating a label image from a json file."""

import argparse
import json
import cv2
import logging as log

from clubs_dataset_tools.label_tools import create_label_image_from_json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Generate a label image from a json file."))
    parser.add_argument('--json_file', type=str, help="Path to the json file.")
    parser.add_argument(
        '--source_image_path', type=str, help="Path to the source image file.")
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help="Save path for the generated label image. Defaults to None.")
    parser.add_argument(
        '--log',
        type=str,
        default='CRITICAL',
        help=("Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
              "Defaults to CRITICAL."))
    args = parser.parse_args()

    numeric_level = getattr(log, args.log.upper(), None)
    log.basicConfig(level=numeric_level)
    log.debug("Setting log verbosity to " + args.log)

    if args.json_file is not None and args.source_image_path is not None:
        full_path = args.json_file
        source_image = cv2.imread(args.source_image_path)
        file_name = full_path.split('/')[-1]

        with open(full_path) as f:
            json_data = json.load(f)

        if args.save_path is not None:
            image_save_path = (args.save_path + '/' + file_name[:-5] + '.png')
        else:
            image_save_path = None

        create_label_image_from_json_data(json_data, source_image, True,
                                          image_save_path)
    else:
        parser.print_help()
