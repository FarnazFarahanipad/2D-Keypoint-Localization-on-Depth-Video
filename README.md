# 2D-Keypoint-Localization-on-Depth-Video
2D Hand Keypoint Localization  on Depth Video through Video to Video Translation

This repository contains the code for method introduced in:
Markerless 2D Fingertip Localization on Depth Videos Using PairedVideo-To-Video Translation

# Requirements
To run the code you can, e.g., install the following requirements:

* python 3
* PyTorch 0.4
* NVIDIA GPU + CUDA cuDNN

# Testing with pretrained model
* Please first download test dataset from here and put it under "datasets/handpose_5_new/".
* Next, download pre-trained model from here.
* Compile a snapshot of FlowNet2 by running python src/download_flownet2.py.
* To test the model: \\
 python test.py --name handpose_5_new  --dataroot  datasets/handpose_5_new/  --label_nc  0  --loadSize 128   --n_downsample_G 2  --use_real_img  --how_many 8600


# Acknowledgments
This code borrows heavily from vid2vid and pix2pixHD.

