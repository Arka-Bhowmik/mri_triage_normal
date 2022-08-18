#------------------------------------------------------------------
# This is a configuration file used for training or testing u-net
# (ONLY MODEIFY CONFIG FILE TO CHANGE IN TRAING/TESTING PROCESS)
#------------------------------------------------------------------
#
#
#
# Import neccessary packages
import os
#
#----------------------------------------------------------------------------
#              (A) ALL MODEL INPUTS FOR TRAINING/TESTING
#----------------------------------------------------------------------------
#
# Define base path to input csv file
BASE_PATH = "/mri_triage_normal/input/"                               # BASE PATH INPUT CSV FILES FOR SEGMENTATION
#
CSV_NAME = "sample_unet.csv"                                          # This csv file does not contain external validation dataset
CSV_NOTSPLIT = os.path.splitext(CSV_NAME)[0]                          # FILENAME OF CSV FILE (TO BE SPLITTED INTO Train & valid set)
#
# CSV Name after spliting to train & validation set
CSV_TRAIN = "train.csv"                                               # CSV NAME OF ALREADY SPLITTED TRAIN FILE
train_filename=os.path.splitext(CSV_TRAIN)[0]                         # FILENAME FOR TRAINING CSV
CSV_VALID = "valid.csv"                                               # CSV NAME OF ALREADY SPLITTED VALID FILE
valid_filename=os.path.splitext(CSV_VALID)[0]                         # FILENAME FOR VALID CSV
CSV_TEST = "test.csv"                                                 # CSV NAME CALLED DURING TESTING PROCESS
test_filename=os.path.splitext(CSV_TEST)[0]                           # FILENAME FOR TEST CSV
#
# Spliting ratio to split entire training data int 80% training set and 20% validation set
train_ratio=0.80           # SPECIFY THE SPLIT RATIO FOR TRAINING SET
validation_ratio=0.20      # SPECIFY THE SPLIT RATIO FOR VALIDATION SET
split_ID='MRN'             # SPECIFY THE COLUMN NAME BY WHICH SPLIT IS PERFORMED
split_flg='nosplit'        # SPECIFY 'split' OR 'nosplit' FOR SPLIT OPERATION TO EXECUTE
#
# Initialize our learning rate, number of epochs to train and the batch size
INIT_LR = 1e-4            # OPTIMAL LEARNING RATE
NUM_EPOCHS = 200          # NUMBER OF ITERATION TRAINING DATA WILL BE TRAINED
BATCH_SIZE = 3            # TRAINING BATCHES
IMAGE_SIZE = 224          # SIZE OF ORIGINAL IMAGE AND MASK
IMAGE_CHANNELS = 1        # IMAGE CHANNEL TO BE SET AS 1 FOR SEGMENTATION
SHUFF=True                # SPECIFY True or False FOR SHUFFLE DATA IN DATAGEN
#
h5file_name='unet'
plot_ACC_LOSS="N"         # Plot training curves
#
# Define output path for model storage and testing of script
BASE_OUTPUT = "/mri_triage_normal/output/"
#
