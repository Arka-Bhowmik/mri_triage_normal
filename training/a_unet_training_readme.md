# INSTRUCTION FOR TRAINING U-NET

This folder contains a U-Net training script namely "training_unet.pynb" used for training a u-net net work. In order to train the U-net, binary mask acts as a ground truth which is created using a matlab application "create_mask.mlx". For creating the mask using the matlab code, one can follow the below steps

### I. MASK CREATION STEPS

Follow the steps in "mask_creation.mp4". Ensure the "sample.csv" with path of original image is available in the same format as in csv file provided in folder path "/mri_triage_normal/input/sample.csv" or else modify the "create_mask.mlx" script accordingly.

![screenshot_mask](https://user-images.githubusercontent.com/56223140/185296161-d4eaa8cf-5776-484d-89a9-e6dce99784b4.png)


### II. TRAINING U-Net

