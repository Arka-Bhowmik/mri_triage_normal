# INSTRUCTION FOR TRAINING VGG-16

This folder contains a VGG-16 binary classification training script namely "training_vgg16.pynb" used for training a classification network. To train the classification network, the radiologist BI-RADS category acted as a ground-truth and input to classification network is segmented MIP of single breast from thorax. Note, for training the classifier two classes were defined (i) BI-RADS 1&2 (as negative class or extremely low suspcion) and (ii) BI-RADS 3to6 (as positive class or possibly suspicious). The BI-RADS catagory was extracted from patient reoprt.

### I. TRAINING VGG-16 Network

To run the VGG-16 network below python packages can be installed in local computer or server. Refer PDF file for how to install neccessary packages.

Python package pre-requsite:
1. tensorflow (python deep learning package), 
2. tensorflow_addons, 
3. matplotlib,
4. scikit-learn,
5. pandas.



#### A. The main training function for VGG-16 uses couple of sub-functions provided in path "/mri_triage_normal/support_function/". The main script is for execution and supporting functions are for executing suppoting tasks in the main script.


##### (a) Input                         ---> csv file with paths for segmented breast MIPs  (refer path "/mri_triage_normal/input/sample_vgg16.csv")

##### (b) Run the main script (only)    ---> "training_vgg16.pynb" [Note: all paths has to be re-defined according to local computer]

##### (c) Configuration for main script ---> "config_vgg16.py" [Note: Input changes are specified in config_vgg16.py]


#### B. Other supporting sub-functions for training VGG-16  (refer path "/mri_triage_normal/support_function/"):


##### (d) read_and_split_vgg16.py       ----> For spliting the data into training and validation set

##### (e) create_dictionary_vgg16.py    ----> For creating training paths and validation paths into dictionary

##### (f) Data_gen_vgg16.py             ----> For generating training and validation generator
