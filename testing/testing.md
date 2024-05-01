![5429470](https://github.com/Arka-Bhowmik/mri_triage_normal/assets/56223140/c29ab39b-7476-4679-aa6e-473767374dc7)
## RUNNING STANDALONE INFERENCE WITH DOCKER (weights amd required packages pre-installed)

##### Download docker image (5 Gigabyte) :  https://drive.google.com/file/d/1BP3KdHD7O7c_89MCQ_hpmfH4H5dM3M5k/view?usp=sharing  
(use userguide for installation and running docker container in local machine)

Note: User need to convert nifti image of T1_Fat_Sat (i.e., T1_post1-T1_pre) prior to implementing the docker image



#

## RUNNING INFERENCE WITHOUT DOCKER

This folder contains python files for running inference. Here, Input files are substracted T1_Fat_Sat Nifti file.

a) User need to convert nifti image of T1_Fat_Sat (i.e., T1_post1-T1_pre) prior to implementing the inference code

b) The program works well for tensorflow version 2.5.3 since model uses lamda layer which is depricated in recent versions of tensorflow

c) Download the trained weight provided in output path and copy weights from download folder to output folder

### CODE FOR RUNNING ENSEMBLE MODEL INFERENCE
##### inference.ipynb   
(use userguide for installation of pre-requisite packages)

### CODE FOR RUNNING GRAD-CAM
##### grad_cam_2d.ipynb   
(use userguide for installation of pre-requisite packages)
