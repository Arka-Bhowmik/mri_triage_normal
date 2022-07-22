# Deep Learning Scripts for Traiging Normal Breast MRI

This repository consist scripts for training a deep learning model using breast MRI images as input and testing for traiging normal breast MRI to abbreviated list. The entire script can be runned using python and deep learning library tensorflow-Keras. The needed packages for training the algorithm on new data is detailed in the training folder. On the other hand, packages needed for testing only using weights trained by us is detailed in testing folder.

### TRAINING PIPELINE

The training pipeline consist of training two seperate network (2D U-Net and VGG-16). The input to the U-Net is the substracted maximum intensity projection (MIP) images and out of U-Net is 2D binary mask. The U-Net is used here to segment the input image, i.e., breast from the thorax. Next, the mask is used to segment out the breast region. 
