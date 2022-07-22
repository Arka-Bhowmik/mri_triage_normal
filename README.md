# Deep Learning Scripts for Traiging Normal Breast MRI

This repository consist scripts for training a deep learning model using breast MRI images as input and testing for traiging normal breast MRI to abbreviated list. The entire script can be runned using python and deep learning library tensorflow-Keras. The needed packages for training the algorithm on new data is detailed in the training folder. On the other hand, packages needed for testing only using weights trained by us is detailed in testing folder.

**Scripts are prepared using deep learning library tensorflow-Keras**

### TRAINING PIPELINE

The training pipeline consist of training two seperate networks in seperately (2D U-Net and VGG-16) as shown in **Fig. 1**. The input to the U-Net is substracted maximum intensity projection (MIP) images and output of U-Net is 2D binary mask. The U-Net is trained to segment the input image, i.e., breast from the thorax. Next, the mask is used to segment out the breast region. The segmented MIP images are divided from center into two halves and fed to the VGG 16 binary classifier to classify negative and positive labels. The classification network (VGG 16) is trained using BIRADS score given be radiologists (i.e., label "Neg/less suspicion": BIRADS 1&2, label "Pos/highly suspcion": BIRADS 3,4,5 & 6).   

![image](https://user-images.githubusercontent.com/56223140/180337017-2937f4ed-a70a-4608-b246-8270b879aad5.png)

**Figure 1: Deep learning training workflow.**

**NOTE**: Steps for training fresh MRI data along with installing the required packages are detailed in **training** folder


