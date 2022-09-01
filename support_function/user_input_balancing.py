def user_input_balancing(counter_tr, counter_vl):
    #
    # IMPORT PACKAGES
    import sys
    #
    # ADD PATH: For suppoting functions
    sys.path.append('/mri_triage_normal/support_function/')
    #
    # IMPORT INPUT ARGUMENTS
    import config                                         # Calls user-input function
    from manual_sampling_mip import manual_sampling       # Calls sub-function for manual oversampling of weaker class
    #
    # ------------------------------------------------------------------------
    # THIS SCRIPT SELECT THE PREFERENCE FOR CLASS BALANCING
    # CLASS BALANCING IS PERFORMED TO MAKE THE TRAINING BIAS FREE
    #-------------------------------------------------------------------------
    # TO CALL THIS FUNCTION USE:
    # user_input_balancing(counter_tr, counter_vl)
    # 
    # 1.  counter_tr:-      NO. OF TRAINING DATA CLASSES
    # 2.  counter_vl:-      NO. OF VALIDATION DATA CLASSES
    #-------------------------------------------------------------------------------------
    # INSTRUCTION: THIS PROGRAM IS CALLED IN main training_vgg16.pynb or .py 
    # CHECK THE TRAINING PROGRAM TO UNDERSTAND THE CALLING METHOD
    #
    # Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
    # Memorial Sloan Kettering Cancer Center, NY (2022)
    #-----------------------------------------------------------------------------------------------
    # OUTPUT OF THIS PROGRAM IS MANUAL OVERSAMPLED DATA
    #-----------------------------------------------------------------------------------------------
    if counter_tr[0]!=counter_tr[1]:                                  # IF STATEMENT FOR TRAIN SAMPLE
        bal_type='Y'
    elif counter_tr[0]==counter_tr[1]:
        bal_type='N'
    #
    if counter_vl[0]!=counter_vl[1]:                                  # IF STATEMENT FOR VALID SAMPLE
        balv_type='Y'
    elif counter_vl[0]==counter_vl[1]:
        balv_type='N'
    #    
    if bal_type=='Y' or balv_type=='Y':                               # IF STATEMENT FOR MANUAL SAMPLING
        bal_m='Y'
    elif bal_type=='N' or balv_type=='N':
        bal_m='N'
    else:
        bal_m='N'
    #
    #------------------MANUAL BALANCING BEGIN--------------------
    if bal_m=='Y':                                                    # IF STATEMENT FOR MANUAL BALANCING
        print("MANUAL BALANCING WILL RANDOMLY OVERSAMPLE THE WEAK POPULATION")
        #
        if bal_type=='Y':                                             # IF STATEMENT FOR MANUAL BALANCING OF TRAIN DATA
            manual_sampling(counter_tr, config.BASE_PATH, config.train_filename)   # FUNCTION FOR MANUAL BALANCING OF TRAIN SET
        else:
            pass
        #
        if balv_type=='Y':                                            # IF STATEMENT FOR MANUAL BALANCING OF VALID DATA
            manual_sampling(counter_vl, config.BASE_PATH, config.valid_filename)   # FUNCTION FOR MANUAL BALANCE OF VALID SET
        else:
            pass
        #
    #------------------MANUAL BALANCING END--------------------
    #
    else:
        print("NO BALANCING PERFORMED")
    #
    return