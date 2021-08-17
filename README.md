# 2D-Keypoint-Localization-on-Depth-Video
2D Hand Keypoint Localization  on Depth Video through Video to Video Translation

This repository contains the code for method introduced in:
Markerless 2D Fingertip Localization on Depth Videos Using PairedVideo-To-Video Translation

# Requirements
To run the code you can, e.g., install the following requirements:

* python 3
* PyTorch 0.4
* FlowNet2 
* NVIDIA GPU + CUDA cuDNN

# Testing with pretrained model
* Please first download the example of test dataset for [source](https://drive.google.com/drive/folders/1UyQr-1a0COy-V0JHyzzfb_jfqJLBl1mD?usp=sharing) and [targert](https://drive.google.com/drive/folders/1qPIIgzLvuEqND9VDEecEk6L3bRVA5u-x?usp=sharing) domain and put it under "datasets/handpose_5_new/".
* Also, download pre-trained model from [here](https://drive.google.com/drive/folders/1GI6llwGMj9vQ0DFOkNaC3ty4VVO8zrsi?usp=sharing) and put it under "checkpoints/handpose_5_new/"
* Next, compile a snapshot of FlowNet2 by running python util/download_flownet2.py
* To get the translated sequenced from the first stage, run: 

 
   python stage1_test.py --name handpose_5_new  --dataroot  datasets/handpose_5_new/  --label_nc  0  --loadSize 128   --n_downsample_G 2  --use_real_img  --how_many 8600
   
   
example of translated sequences from the first stage can be downloaded from [here](https://drive.google.com/drive/folders/14RfvHUSVPnFCpa1G-0mdLzPoi0r7YqtQ?usp=sharing, https://drive.google.com/drive/folders/1cq95D7rKGRXTRtK3JIaT0iThAkv3bybP?usp=sharing, https://drive.google.com/drive/folders/1fV8vodZBohacYsGJ-5io9b4NTLStd7UU?usp=sharing, https://drive.google.com/drive/folders/1y68d5BBFzLnm4ZkwOHb8ZPyuRkouYwCa?usp=sharing)

* Next, to perform the segmentation in the second stage, run:

# Acknowledgments
This code borrows heavily from vid2vid and pix2pixHD.

