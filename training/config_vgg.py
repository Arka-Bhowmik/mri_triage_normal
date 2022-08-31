#------------------------------------------------------------------
# This is a configuration file used for training vgg16
# (ONLY MODEIFY CONFIG FILE TO CHANGE IN TRAINING PROCESS)
#------------------------------------------------------------------
#
#
#
# Import neccessary packages
import os
#
#----------------------------------------------------------------------------
#              (A) ALL MODEL INPUTS FOR TRAINING
#----------------------------------------------------------------------------
#
# Define base path to input csv file
BASE_PATH = "/mri_triage_normal/input/"        			    # BASE PATH INPUT CSV FILES FOR VGG-16
#
CSV_NAME = "sample_vgg.csv"                                 # This csv file does not contain external validation dataset
CSV_NOTSPLIT = os.path.splitext(CSV_NAME)[0]                # FILENAME OF CSV FILE (TO BE SPLITTED INTO Train & valid set)
#
# CSV Name after spliting to train & validation set
CSV_TRAIN = "train.csv"                                     # CSV NAME OF ALREADY SPLITTED TRAIN FILE
train_filename=os.path.splitext(CSV_TRAIN)[0]               # FILENAME FOR TRAINING CSV
CSV_VALID = "valid.csv"                                     # CSV NAME OF ALREADY SPLITTED VALID FILE
valid_filename=os.path.splitext(CSV_VALID)[0]               # FILENAME FOR VALID CSV
#
# Spliting ratio to split entire training data int 80% training set and 20% validation set
train_ratio=0.80                                            # SPECIFY THE SPLIT RATIO FOR TRAINING SET
validation_ratio=0.20                                       # SPECIFY THE SPLIT RATIO FOR VALIDATION SET
split_ID='MRN'                                              # SPECIFY THE COLUMN NAME BY WHICH SPLIT IS PERFORMED
split_flg='split'                                           # SPECIFY 'split' OR 'nosplit' FOR SPLIT OPERATION TO EXECUTE
#
# Initialize our learning rate, number of epochs to train and the batch size
INIT_LR = 1e-5                                              # INITIAL LEARNING RATE FOR ADAM OPTIMIZER
NUM_EPOCHS = 100                                            # NUMBER OF ITERATION TRAINING DATA WILL BE TRAINED
BATCH_SIZE = 10                                             # TRAINING BATCHES
IMAGE_SIZE = 256                                            # SIZE OF ORIGINAL IMAGE
IMAGE_CHANNELS = 3                                          # IMAGE CHANNEL 3
CLASS_NUM=2                                                 # IMAGE CLASS NUMBER
AGUMENT_METH='fliponly'                                     # AGUMENTATION METHOD (AVAILABLE:--> None, fliponly)
FROZEN_LAYERS=0                                             # NUMBER OF FROZEN LAYERS
SHUFF=True                                                  # SPECIFY True or False FOR SHUFFLE DATA IN DATAGEN
#
h5file_name='vgg16'                                         # MODEL NAME KEY
plot_ACC_LOSS="N"                                           # Y/N TO PLOT LEARNING CURVE
#
# Define output path for model storage and testing of script
BASE_OUTPUT = "/mri_triage_normal/output/"
#