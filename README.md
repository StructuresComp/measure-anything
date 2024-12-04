# Measure Anything: Real-time, Multi-stage Vision-based Dimensional Measurement using Segment Anything

---
**Measure Anything** is an **interactive** / **automated** dimensional measurement tool that leverages the 
[Segment Anything Model (SAM) 2](https://github.com/facebookresearch/sam2) to segment objects of interest and 
provide real-time, **diameter**, **length** and **volume** measurements. Our streamlined pipeline 
comprises five stages: 1) segmentation, 2) binary mask processing, 3) skeleton construction, 4) line segment and 
depth identification and 5) 2D-3D transform and measurement. We envision that this pipeline can be easily
adapted to other fully automated or minimally human-assisted, vision-based measurement tasks.




[[`Paper`](https://google.com/)] [[`Project`](https://measure-anything.github.io/)]

<p align="center">
  <img src="figures/1.gif" alt="GIF 1" width="49%">
  <img src="figures/6.gif" alt="GIF 2" width="49%">
</p>
<p align="center">
  <img src="figures/7.gif" alt="GIF 3" width="49%">
  <img src="figures/3.gif" alt="GIF 4" width="49%">
</p>

[//]: # (<p align="center"><em>Interactive Demo Examples</em></p>)

---
# Installation #
### 1. Create conda environment for managing dependencies ###
```bash
$ conda create --name <environment> python=3.12
$ conda activate <environment>
```
### 2. Setting up the ZED SDK and Python API ###
This repository was tested with the ZED SDK 4.2.1 and CUDA 12 on Ubuntu 22.04. To set up the ZED SDK, follow these steps:
- Install dependencies
```bash
$ pip install cython numpy==1.26.4 opencv-python==4.9.0.80 pyopengl
```
- Download the ZED SDK from the [official website](https://www.stereolabs.com/developers/release#82af3640d775)
- Run the ZED SDK Installer
```bash
$ cd path/to/download/folder
$ sudo apt install zstd
$ chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.2.1.zstd.run
$ ./ZED_SDK_Ubuntu22_cuda12.1_v4.2.1.zstd.run
```
- To install the Python API, press Y to the following when running the installer:
``` bash
Do you want to install the Python API (recommended) [Y/n] ?
``` 
  or alternatively, you can install the Python API separately running the following script:
```bash
$ cd "/usr/local/zed/"
$ python3 get_python_api.py
````
### 3. Install Pytorch ###
- Follow instructions on the [official website](https://pytorch.org/get-started/locally/)
- For example, to install Pytorch with CUDA 12.1:
```bash
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### 4. Install Ultralytics ###
We use the SAM 2 API provided by Ultralytics. The Segment Anything Model checkpoint will be downloaded automatically when running the the demo script for the first time.
```bash
$ pip install ultralytics
```
### 5. Install remaining dependencies ###
```bash
$ pip install scikit-image scikit-learn pillow plyfile
```
---
# Demo #
## Interactive Demo ##
The interactive demo requires `.svo` files collected using a ZED stereo camera. Example `.svo` files can be found [here](https://drive.google.com/drive/folders/1Q6). Run the demo and follow onscreen instructions.
```bash
python interactive_demo.py --input_svo path/to/svo/file.svo --stride 10 --thin_and_long
```
- `--thin_and_long` is a flag variable that decides the skeleton construction method. Toggling this flag will construct the skeleton based on skeletonization (recommended for rod-like geometries).
- `--stride (int)` is an optional parameter that determines the distance between consecutive measurements. The default value is 10.
- Red line indicate valid measurements.
- Blue line segments indicate invalid measurements, due to unavailable depth data.
- The calculated stem diameters are available as a numpy file in `./output/{svo_file_name}/{frame}/diameters.npy` ordered from the bottommost to the topmost line measurements.

<p align="center">
<img src="figures/canola.gif" alt="GIF 1" width="98%">
</p>

## Automated Demo using Keypoint Detection ##

The keypoint detection weights can be specified using the `--weights` parameter. This enables automated point prompting for segmentation. If the automated segmentations are inaccurate, users can manually intervene to refine the results. The keypoints can be configured as positive prompts or a combination of positive and negative prompts, depending on the specific requirements of the video.

<p align="center">
  <img src="figures/kpd_1_1.gif" alt="KPD GIF 1" width="49%">
  <img src="figures/kpd_1_2.gif" alt="KPD GIF 2" width="49%">
</p>
<p align="center">
  <img src="figures/kpd_2_1.gif" alt="KPD GIF 3" width="49%">
  <img src="figures/kpd_2_2.gif" alt="KPD GIF 4" width="49%">
</p>
<p align="center"><em>Interactive Automated Demo Examples</em></p>

## Demo on Robotic Grasping

Measure Anything can be used to provide geometric priors for obtaining optimized grasping points according to a model. In our experiments we are using a simple stability model based on form-closure and perpendicular distance from CoM. However this stability model can be switched with a SOTA deep learning model.

Check `interactive_demo_clubs_3d.py` to run an interactive demonstration on [Clubs-3D](https://clubs.github.io/#:~:text=CLUBS%20is%20an%20RGB%2DD,objects%20packed%20in%20different%20configurations.) dataset. 

<p align="center">
  <img src="figures/clubs_3d_00.png" alt="CLUBS 1" width="49%">
</p>
<p align="center">
  <img src="figures/clubs_3d_10_heatmap.png" alt="CLUBS 2" width="49%">
  <img src="figures/clubs_3d_203.png" alt="CLUBS 3" width="49%">
</p>
<p align="center"><em>Robotic Grasping Demo</em></p>