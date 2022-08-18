def read_and_split(csv_path,csv_name,ID,train_ratio,validation_ratio,flag):
    #----------------------------------
    # IMPORT SOURCE LIBRARIES
    #----------------------------------
    from sklearn.model_selection import GroupShuffleSplit     # FUNCTION FOR SPLITTING IN GROUP
    # ---------------------------------
    # IMPORT HELPER LIBRARIES
    # ---------------------------------
    import os
    import os.path
    import pandas as pd
    import csv
    # ------------------------------------------------------------------------
    # THIS SCRIPT READ FILES FROM CSV PATH AND SPLITS INTO TRAIN/VALID SET
    #-------------------------------------------------------------------------
    # TO CALL THIS FUNCTION USE:
    # read_and_split(csv_path,csv_name,ID,train_ratio,validation_ratio,flag)
    #
    # 1.  csv_path:-        path to csv file
    # 2.  csv_name:-        name of the csv file
    # 3.  ID:-              name of group along which split will be performed (e.g., PatientID/MRNs)
    # 4.  train_ratio:-     Ratio of spliting training data from csv file
    # 5.  validation ratio:-Ratio of splitting validation data from csv file
    # 6.  flag:-            Flag specifies whether spliting is performed or not (if already splitted)
    #-------------------------------------------------------------------------------------
    # INSTRUCTION: THIS PROGRAM IS CALLED IN training_unet.ipynb or .py 
    # CHECK THE training PROGRAM to understand the calling method
    #
    # Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
    # Memorial Sloan Kettering Cancer Center, NY (2022)
    #-----------------------------------------------------------------------------------------------
    # OUTPUT OF THIS PROGRAM IS CSV FILES CALLED TRAIN AND VALID
    #-----------------------------------------------------------------------------------------------
    if flag == 'split':          # FLAG FOR SPLITING TO PROCEED
        #------------------------------------------------------------------------
        # READS THE CSV FILE TO DATAFRAME CALLED CSV_DATA FOR SPLITING
        #-------------------------------------------------------------------------
        filename=csv_name + '.csv'
        csv_data = pd.read_csv(os.path.join(csv_path, filename), index_col=0) # Reads the csv data to dataframe
        #------------- Inputs for the function-----------------
        tr_ratio=train_ratio;                      # DEFINES TRAINING DATA RATIO
        vl_ratio=validation_ratio;                 # DEFINES VALIDATION DATA RATIO
        groupID=ID                                 # DEFINES THE MRN USED FOR TO PERFORM GROUPSPLIT
        #--------------------------------------------------------------------------------------------
        #
        train_inds,test_inds=next(GroupShuffleSplit(n_splits=2, test_size = 1-tr_ratio,random_state=8).split(csv_data,groups=csv_data[groupID]))
        train=csv_data.iloc[train_inds]            # TRAIN DATA AFTER SPLITING
        validate=csv_data.iloc[test_inds]          # VALID DATA AFTER SPLITING
        #
        # ADDITIONAL RANDOM SHUFFLE OF DATASET ROWS
        train=train.sample(frac=1, random_state=42)
        validate=validate.sample(frac=1, random_state=42)
        #
        #
        #---------------------------------------------------
        # SAVE THE TRAIN AND VALID SET TO CSV FILES
        #---------------------------------------------------
        train.to_csv(os.path.join(csv_path, "train.csv"))         # Replace the files if already present
        validate.to_csv(os.path.join(csv_path, "valid.csv"))      # Replace the files if already present
        #
    if flag=='nosplit':  # FLAG TO NOT PROCEED FOR SPLITING SINCE TRAIN AND VALID DATA ALREADY EXIST IN THE PATH
        #-------------------------------------------------------------------------
        print("NO SPLIT PERFORMED SINCE TRAIN & VALID SET ALREADY EXIST")
        print("")
    #-------    
    return