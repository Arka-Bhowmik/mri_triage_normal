def create_dictionary(csv_name, csv_path, flag):
    # ---------------------------------
    # IMPORT HELPER LIBRARIES
    # ---------------------------------
    import os
    import os.path
    import pandas as pd
    import csv
    import numpy as np
    # ------------------------------------------------------------------------
    # THIS SCRIPT INSERT A UNIQUE ID ON THE LEFT AND CREATES PARTITION FILE
    #-------------------------------------------------------------------------
    # TO CALL THIS FUNCTION USE: create_dictionary(csv_name,csv_path,flag)
    # 1.  csv_path:-        path to csv file
    # 2.  csv_name:-        name of the csv file
    # 3.  flag:-            flag for shuffle 'random' or 'None' 
    #-------------------------------------------------------------------------------------
    # INSTRUCTION: THIS PROGRAM IS CALLED IN amian_training_vgg16.ipynb or .py 
    # CHECK THE TRAINING PROGRAM TO UNDERSTAND THE CALLING METHOD
    # NOTE: THIS PROGRAM DOES NOT CHANGE THE CSV FILE USEFUL FOR COUNTING AND DATAGEN
    #
    # Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
    # Memorial Sloan Kettering Cancer Center, NY (2022)
    #-------------------------------------------------------------------------------------
    # OUTPUT OF THIS PROGRAM IS A PARTITION FILES LINKED WITH UNIQUE ID
    #
    filename=csv_name + '.csv'
    # STORE CSV TO A DATAFRAME
    data_df = pd.read_csv(os.path.join(csv_path, filename))
    #
    # RANDOMIZE THE DATA
    if flag == 'random':
        data_df=data_df.sample(frac=1, random_state=10)
    elif flag == 'None':
        print("")
    #    
    data_df.insert(loc=0, column='selfID', value=np.arange(len(data_df)))  # INSERT ID CORRESPONDING TO EACH DATA
    #
    img_id=data_df.selfID.tolist()            # SELF DEFINED ID TO TRACK THE IMAGES
    img_labels=data_df.Split_class.tolist()   # STORES THE Split_class COLUMN IN IMAGE LABELS
    img_paths=data_df.File_path.tolist()      # STORES THE File_path COLUMN IN IMAGE PATH (FOR MIP)	
    #
    partition = {}                            # CELL INITIALIZATION
    labels = {}                               # CELL INITIALIZATION
    impath={}                                 # CELL INITIALIZATION
    partition[csv_name] = []                  # CELL INITIALIZATION
    #
    for idx in range(len(img_paths)):
        full_name=img_id[idx]
        partition[csv_name].append(full_name)
        labels[full_name]=img_labels[idx]
        impath[full_name]=img_paths[idx]
    #
    return partition[csv_name], labels, impath