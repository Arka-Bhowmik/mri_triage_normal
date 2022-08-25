# INSTRUCTION FOR TRAINING VGG-16

This folder contains a VGG-16 binary classification training script namely "training_vgg16.pynb" used for training a classification network. To train the classification network, the radiologist BI-RADS category acted as a ground-truth and input to classification network is segmented single breast from thorax. Note, for training the classifier two classes were defined (i) BI-RADS 1&2 (as negative class or extremely low suspcion) and (ii) BI-RADS 3to6 (as positive class or possibly suspicious). The BI-RADS catagory was extracted from patient reoprt.

### I. TRAINING VGG-16 Network

To run the VGG-16 network below python packages can be installed in local computer or server. Refer PDF file for how to install neccessary packages.

Python package pre-requsite:
1. tensorflow (python deep learning package), 
2. tensorflow_addons (use pip install tensorflow-addons), 
3. matplotlib (use pip install matplotlib)
4. scikit-learn (use pip install scikit-learn)
5. pandas (use pip install pandas)



#### The main training function for Vgg-16 uses couple of sub-functions provided in path "/mri_triage_normal/support_function/". The main script is for execution and supporting functions are for executing suppoting tasks in the main script.


##### (a) Input                         ---> csv file with paths for image (without segmentation) and binary mask path (refer path "/mri_triage_normal/input/sample_unet.csv")

##### (b) Run the main script (only)    ---> "training_unet.pynb" [Note: all paths has to be re-defined according to local computer]

##### (c) Configuration for main script ---> "config_unet.py" [Note: Input changes are specified in config_unet.py]


### Other supporting sub-functions for training u-net (refer path "/mri_triage_normal/support_function/"):


##### (d) read_and_split_unet.py       ----> For spliting the data into training and validation set

##### (e) create_dictionary_unet.py    ----> For creating training paths and validation paths into dictionary

##### (f) Data_gen_unet.py             ----> For generating training and validation generator

##### (g) custom_model_unet.py         ----> Includes the u_net architecture

##### (h) custom_metrics.py            ----> Includes functions for custom metrics (Dice similarity coefficient and Intersection over union)
