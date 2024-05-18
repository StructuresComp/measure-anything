# Stem-diameter-estimation
Complete repository for stem diameter estimation through keypoint detection, segmentation, skeletonization and deprojection

## Conda Environment ##
### Create conda environment for managing dependencies ###
```bash
$ conda create --name <environment> python=3.12.3
$ conda activate <environment>
```
### Install PyTorch ###
- Use https://pytorch.org/get-started/locally/ guide to install torch, torchvision, torchaudio
### Install Ultralytics for YOLO ###
```bash
$ pip install ultralytics
```
### Install segment-anything (SAM) ###
```bash
$ git clone https://github.com/facebookresearch/segment-anything.git
$ cd segment-anything
$ pip install -e .
$ pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Running Stem Segmentation Script ##
- In stemSegment folder, create SAM and YOLO directories
    - install SAM model into SAM directory
    - install YOLO model into YOLO directory
```bash
python3 stem_segmentation.py --input_file <input file> --outdir <output directory>
```