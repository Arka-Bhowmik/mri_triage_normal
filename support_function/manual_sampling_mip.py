def manual_sampling(counter_name, csv_path, csv_file):
    #-----------------------------
    # IMPORT SUPPORTING LIBRARIES
    #----------------------------
    import os
    import os.path
    import sys
    import csv
    import pandas as pd
    import math
    #--------------------------
    # IMPORT PYTHON FUNCTION
    from collections import Counter
    #
    # ---------------------------------------------------------------------------------
    # THIS SCRIPT RANDOMLY OVERSAMPLE THE WEAK POPULATION/CLASS TO BALANCE THE CLASSES
    #---------------------------------------------------------------------------------
    # TO CALL THIS FUNCTION USE:
    # manual_sampling(counter_name, csv_path, csv_file)
    #
    # 1.  counter_name:-    Name of the counter either train/valid
    # 2.  csv_path:-        Path for csv files (be it train or valid)
    # 3.  csv_file:-        Name of the csv file
    #-------------------------------------------------------------------------------------
    # INSTRUCTION: THIS PROGRAM IS CALLED IN user_input_balancing.py 
    # CHECK THE user_input_balancing PROGRAM to understand the calling method
    #
    # Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
    # Memorial Sloan Kettering Cancer Center, NY (2022)
    #-----------------------------------------------------------------------------------------------
    # OUTPUT OF THIS PROGRAM IS MANUALLY OVERSAMPLED DATA AND A PRINT STATEMENT
    #-----------------------------------------------------------------------------------------------
    Neg_cl_no=counter_name[0]
    Pos_cl_no=counter_name[1]
    #
    if Neg_cl_no < Pos_cl_no:                  # IF STATEMENT TO SET DIFF, WEAK_CL, SMALL, AND FLAG
        diff=Pos_cl_no - Neg_cl_no
        weak_cl=Neg_cl_no
        flag= 0
        small= 0
        iteration=(math.floor(diff/weak_cl)+1)
    elif Neg_cl_no > Pos_cl_no:                # IF STATEMENT TO SET DIFF, WEAK_CL, SMALL, AND FLAG
        diff=Neg_cl_no - Pos_cl_no
        weak_cl=Pos_cl_no
        flag= 0
        small= 1
        iteration=(math.floor(diff/weak_cl)+1)
    elif Neg_cl_no==Pos_cl_no:                # ELSE STATEMENT TO SET DIFF, WEAK_CL, SMALL, AND FLAG
        flag=1
        iteration=0
    #
    filename=csv_file + '.csv'               # FILENAME THAT IS OVERSAMPLED
    #
    for it in range(iteration):
        #
        if flag==0:
            # IF STATEMENT INCASE OF CLASS IMBALANCE/RANDOM SAMPLING
            #-----------------------------------------------
            # LOAD CSV FILE WITH NAME PROVIDED BY CSV TYPE
            #-----------------------------------------------
            csv_data = pd.read_csv(os.path.join(csv_path, filename))
            #
            #----------------------------------------------
            # LIST THE ORIGINAL/MASTER DATA OF CSV FILE
            #----------------------------------------------
            img_birad=csv_data.BIRADS.tolist()
            img_class=csv_data.Split_class.tolist()
            img_mrn=csv_data.MRN.tolist()
            img_access=csv_data.Accession.tolist()
            img_file13=csv_data.File.tolist()
            img_path13=csv_data.File_path.tolist()
            img_patho=csv_data.Patho.tolist()
            img_later=csv_data.Laterity.tolist()
            #
            #------------------------------------------------------
            # SUFFLE THE MASTER DATA RANDOMLY
            #------------------------------------------------------
            #
            csv_data_1=csv_data.sample(frac=1, random_state=10)
            csv_data_2=csv_data_1.sample(frac=1, random_state=5)
            csv_data_3=csv_data_2.sample(frac=1, random_state=7)
            csv_data_4=csv_data_3.sample(frac=1, random_state=13)
            #
            #------------------------------------------
            # LIST THE DATA AFTER RANDOM SAMPLING
            #------------------------------------------
            #
            img_birad1=csv_data_4.BIRADS.tolist()
            img_class1=csv_data_4.Split_class.tolist()
            img_mrn1=csv_data_4.MRN.tolist()
            img_access1=csv_data_4.Accession.tolist()
            img_file113=csv_data_4.File.tolist()
            img_path113=csv_data_4.File_path.tolist()
            img_patho1=csv_data_4.Patho.tolist()
            img_later1=csv_data_4.Laterity.tolist()
            #
            #----------------------------------------------------------
            # SAVES NEW TRAINING/VALID DATASET AFTER MANUAL BALANCING
            #----------------------------------------------------------
            headerList = ['BIRADS', 'Split_class', 'MRN', 'Accession', 'File', 'File_path', 'Patho', 'Laterity']
            #
            with open(os.path.join(csv_path, filename), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headerList)    # first row is assigned to csv header
                #-----------------------------------------------------
                for idx in range(len(img_class)):
                    Birad=img_birad[idx]
                    cl=img_class[idx]
                    mrn=img_mrn[idx]
                    acc=img_access[idx]
                    file13=img_file13[idx]
                    filepath13=img_path13[idx]
                    pathol=img_patho[idx]
                    lateral=img_later[idx]
                    #
                    col = [Birad, cl, mrn, acc, file13, filepath13, pathol, lateral]
                    writer.writerow(col)
                    #
                #-----------------------------------------------------
                count=1
                #
                for idy in range(len(img_class1)):
                    if count<=diff:                 # IF STATEMENT TO RESTRICT WRITING TO CSV
                        if img_class1[idy]==small:  # IF STATEMENT TO ENTER ONLY WHEN SMALL VALUE SATISFIES
                            #
                            Birad1=img_birad1[idy]
                            cl1=img_class1[idy]
                            mrn1=img_mrn1[idy]
                            acc1=img_access1[idy]
                            file113=img_file113[idy]
                            filepath113=img_path113[idy]
                            pathol1=img_patho1[idy]
                            lateral1=img_later1[idy]
                            #
                            col=[Birad1, cl1, mrn1, acc1, file113, filepath113, pathol1, lateral1]
                            writer.writerow(col)
                            #
                            count=count+1
                            #
                        #
                    #
                #
            #---------------------------------------
            # CHECK THE NEW OUTPUT CSV FILE
            #---------------------------------------
            csv_new = pd.read_csv(os.path.join(csv_path, filename))
            list_n=csv_new.Split_class.tolist()
            shuffled = csv_new.sample(frac=1, random_state=10)
            shuffled.to_csv(os.path.join(csv_path, filename), index=False)
            counter=Counter(list_n)
            Neg_cl=counter[0]
            Pos_cl=counter[1]
            if Neg_cl < Pos_cl:
                diff=Pos_cl-Neg_cl
                flag=0
            elif Neg_cl > Pos_cl:
                diff=Neg_cl-Pos_cl
                flag=0
            elif Neg_cl==Pos_cl:
                flag=1
            #
        #
        elif flag==1:
            # IF STATEMENT INCASE OF CLASS BALANCE            
            #---------------------------------------
            # CHECK THE OUTPUT CSV FILE
            #---------------------------------------
            df = pd.read_csv(os.path.join(csv_path, filename))
            listt=df.Split_class.tolist()
            shuffled_df = df.sample(frac=1, random_state=10)
            shuffled_df.to_csv(os.path.join(csv_path, filename), index=False)
            counter1=Counter(listt)
            Neg_cl=counter1[0]
            Pos_cl=counter1[1]
            flag=1
        #
    return print("BALANCING COMPLETED :", "NEGATIVE CLASS -", Neg_cl, "POSTIVE CLASS -", Pos_cl)
        
            