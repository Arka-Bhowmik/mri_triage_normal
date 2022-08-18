# INSTRUCTION FOR TRAINING U-NET

This folder contains a U-Net training script namely "training_unet.pynb" used for training a u-net network. To train the U-net, binary mask acts as a ground-truth which was created using a matlab application "create_mask.mlx". For creating the mask using the matlab code, one can follow the below steps

### I. MASK CREATION STEPS

Follow the steps in "mask_creation.mp4". Ensure the "sample.csv" with path of original image is available in the same format as in csv file provided in folder path "/mri_triage_normal/input/sample.csv" or else modify the "create_mask.mlx" script accordingly.

![screenshot_mask](https://user-images.githubusercontent.com/56223140/185296161-d4eaa8cf-5776-484d-89a9-e6dce99784b4.png)

### II. TRAINING U-Net Network

To run the u-net network below python packages can be installed in local computer or server. Refer PDF file for how to install neccessary packages.

Python package pre-requsite:
1. tensorflow (python deep learning package), 
2. tensorflow_addons (use pip install tensorflow-addons), 
3. matplotlib (use pip install matplotlib)

The main training function for u-net uses couple of sub-functions provided in path "/mri_triage_normal/support_function/". The main script is for execution and supporting functions are for executing suppoting tasks in the main script.
(a) Run the main script (only)    ---> training_unet.pynb [Note: all paths has to be re-defined according to local computer]
(b) Configuration for main script ---> config_unet.py [Note: Input changes are specified in config_unet.py]

Other supporting sub-functions for traingg u-net:
(c) 

