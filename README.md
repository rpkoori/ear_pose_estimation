# README

## Description
This is the project, Ear Acupoint Detection, for HKUST MSBD5001.

## Task
We have two models: RTMPose and YOLOv8

All codes related to RTMPose are in [RTMPose](RTMPose)


## Image Augmentation
Our code for image augmentation is [Image_Augmentation.ipynb](Image_Augmentation.ipynb)

## RTMPose
### Dataset
- Our dataset is saved in [RTMPose/ear_pose/Ear210_Keypoint_Dataset_coco](RTMPose/ear_pose/Ear210_Keypoint_Dataset_coco)

### Codes
- Our Model definitions are in [RTMPose/ear_pose/configs](RTMPose/ear_pose/configs)
- Use [RTMPose/MMPose.ipynb](RTMPose/MMPose.ipynb) to train and test
- To successfully execute it, you should create a directory 5001Group and then execute commands in [RTMPose/MMPose.ipynb](RTMPose/MMPose.ipynb) to install mmpose. Then you should move [ear_pose](RTMPose/ear_pose) into mmpose for continue commands.

### Log And Checkpoints
- Our Log and checkpoints are stored in [RTMPose/work_dirs_default](RTMPose/work_dirs_default), and [RTMPose/work_dirs_x](RTMPose/work_dirs_x)