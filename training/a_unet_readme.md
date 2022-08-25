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
4. scikit-learn (use pip install scikit-learn)
5. pandas (use pip install pandas)



#### A. The main training function for u-net uses couple of sub-functions provided in path "/mri_triage_normal/support_function/". The main script is for execution and supporting functions are for executing suppoting tasks in the main script.


##### (a) Input                         ---> csv file with paths for image (without segmentation) and binary mask path (refer path "/mri_triage_normal/input/sample_unet.csv")

##### (b) Run the main script (only)    ---> "training_unet.pynb" [Note: all paths has to be re-defined according to local computer]

##### (c) Configuration for main script ---> "config_unet.py" [Note: Input changes are specified in config_unet.py]


#### B. Other supporting sub-functions for training u-net (refer path "/mri_triage_normal/support_function/"):


##### (d) read_and_split_unet.py       ----> For spliting the data into training and validation set

##### (e) create_dictionary_unet.py    ----> For creating training paths and validation paths into dictionary

##### (f) Data_gen_unet.py             ----> For generating training and validation generator

##### (g) custom_model_unet.py         ----> Includes the u_net architecture

##### (h) custom_metrics.py            ----> Includes functions for custom metrics (Dice similarity coefficient and Intersection over union)


