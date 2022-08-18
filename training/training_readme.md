# INSTRUCTION FOR TRAINING U-NET

This folder contains a U-Net training script namely "training_unet.pynb" used for training a u-net net work. In order to train the U-net, binary mask acts as a ground truth which is created using a matlab application "create_mask.mlx". For creating the mask using the matlab code, one can follow the below steps

### I. MASK CREATION STEPS

The training pipeline consist of training two seperate networks in seperately (2D U-Net and VGG-16) as shown in **Fig. 1**. The input to the U-Net is substracted maximum intensity projection (MIP) images and output of U-Net is 2D binary mask. The U-Net is trained to segment the input image, i.e., breast from the thorax. Next, the mask is used to segment out the breast region. The segmented MIP images are divided from center into two halves and fed to the VGG 16 binary classifier to classify negative and positive labels. The classification network (VGG 16) is trained using BIRADS score given be radiologists (i.e., label "Neg/less suspicion": BIRADS 1&2, label "Pos/highly suspcion": BIRADS 3,4,5 & 6).   



### II. TRAINING U-Net

