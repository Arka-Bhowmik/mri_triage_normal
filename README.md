# Deep Learning Scripts for Triaging Normal Breast MRI

This repository consist scripts for training a deep learning model using breast MRI images as input and testing for triaging normal breast MRI to abbreviated list. The entire script can be runned using python and deep learning library tensorflow-Keras. The needed packages for training the algorithm on new data is detailed in the training folder. On the other hand, packages needed for testing only using weights trained by us is detailed in testing folder.

Use of source files and data, with or without modification, are permitted.

1. Please cite the source paper.

2. Source paper:
Automated Triage of Screening Breast MRI Examinations in High-Risk Women Using an Ensemble Deep Learning Model. Investigative Radiology. 2023; DOI: 10.1097/RLI.0000000000000976 <a href="https://journals.lww.com/investigativeradiology/Abstract/9900/Automated_Triage_of_Screening_Breast_MRI.104.aspx"> [View] </a>

4. Further information can be obtained by writing to Arka Bhowmik (arkabhowmik@yahoo.co.uk).

**Scripts are prepared using deep learning library tensorflow-Keras**

## Download ![5429470](https://github.com/Arka-Bhowmik/mri_triage_normal/assets/56223140/d07403fe-0f79-4720-af28-c275b312bde2) image to run inference (download link in testing folder)

### I. TRAINING PIPELINE

The training pipeline consist of training two seperate networks (2D U-Net and VGG-16) as shown in **Fig. 1**. The input to the U-Net is substracted maximum intensity projection (MIP) images and output of U-Net is 2D binary mask. The U-Net is trained to segment the input image, i.e., breast from the thorax. Next, the mask is used to segment out the breast region. The segmented MIP images are divided from center into two halves and fed to the VGG 16 binary classifier to classify negative and positive labels. The classification network (VGG 16) is trained using BI-RADS category given by radiologists (i.e., label "Neg/less suspicion": BIRADS 1&2, label "Pos/highly suspcion": BI-RADS 3,4,5 & 6).   

![image](https://user-images.githubusercontent.com/56223140/180337017-2937f4ed-a70a-4608-b246-8270b879aad5.png)

**Figure 1: Deep learning training workflow.**

**NOTE**: Steps for training fresh MRI data along with installing the required packages are detailed in **training** folder


### II. TESTING PIPELINE

The testing pipeline is completely automated wherein input are DICOM files and output is model prediction (see **Fig. 2**). During testing, the input DICOM files are used to create four subtracted MIPs. A single subtracted MIP with maximum projection of all the slices and three additional subtracted MIPS with maximum projection images of three sub-group slices. The additional three sub-group slices are extracted from the same DICOM files by dividing total slices by three parts. These extracted MIPs are segmented by the trained U-Net simultaneously, followed by dividing into four left and four right breasts corresponding to four subtracted MIPs. The segmented single breasts from all MIPs are then classified by the trained classifier simultaneously.

![Picture1](https://github.com/Arka-Bhowmik/mri_triage_normal/assets/56223140/5e331681-18ea-4be2-a84a-b705d1afa303)

**Figure 2: Deep learning testing (A) workflow, and (B) GRAD-CAM visualization.**
