# ---------------------------------------------------------------------------------
# THIS SCRIPT PERFORMS A DATAGENERATION STRUCTURE (CALLED FOR TRAINING)
# PERFORMS AGUMENTATION ON USER INPUT IN **PARAMS
#----------------------------------------------------------------------------------
# TO CALL THIS CLASS USE: DataGenerator(partition_name, labels, path, **params)
# 1.  partition_name:-        Can be training/valid/test partition/Unique ID
# 2.  labels:-                Labels corresponding to that Unique ID
# 3.  Path:-                  Image path corresponding to that unique ID
# 4.  **params:-              Other default parameter transfered by user 
#
#-------------------------------------------------------------------------------------
# INSTRUCTION: THIS PROGRAM IS CALLED IN training_vgg16.ipynb or .py 
# CHECK THE training_script PROGRAM to understand the calling method
#
#
# Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
# Memorial Sloan Kettering Cancer Center, NY (2022)
#-----------------------------------------------------------------------------------------------
# OUTPUT OF THIS PROGRAM IS EITHER TRAINING/VALID GENERATOR
# 
# IMPORT HELPER LIBRARIES
import numpy as np
import PIL
import random
#
# IMPORT OTHER PYTHON LIBARAIES
import tensorflow
import tensorflow.keras
from tensorflow import keras

class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, list_IDs, labels, impaths, batch_size=32, dim=(224,224), n_channels=3, n_classes=2, shuffle=True, augmentation = 'None', imgsize=224):
        """Initialization
        :param list_IDs: list of all unique ID to be use in the generator
        :param labels: list of image labels (corresponding to unique ID)
        :param impaths: path to images location (corresponding to unique ID)
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: class number
        :param shuffle: True to shuffle unique indexes after every epoch
        :param agumentation: agumentation ON/OFF
        :param imgsize: image size
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.impaths = impaths
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.imgsize = imgsize
        self.on_epoch_end()
    
    #
    #_________on_epoch_end(self) shuffles the batch after each epoch________
    #
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    #
    #__________________PRINT BATCHES_PER_EPOCH__________________
    def __len__(self):
        batches_per_epoch=int(np.floor(len(self.list_IDs) / self.batch_size))
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return batches_per_epoch
    #
    #________________GETITEM CALL ALL THE FUNCTIONS_________________________
    #
    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y      
    
    #        
    #___________________DATA generation functions _____________________________
    #
    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))  # Note X has a shape of (32,224,224,3)
        Y = np.empty((self.batch_size), dtype=np.float32)            # Note the Y has a shape of (32)
        #
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            impaths_temp=self.impaths[ID]           # image path corresponding to ID
            A = self.__load_image(impaths_temp)     # NOTE X shape (32, 224, 224, 3), 32 is the batch size
            X[i,] = A
            # Store class
            Y[i] = self.labels[ID]                  # NOTE y has a shape (32,) where 32 is the batch size
        #
        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes) # For softmax activation in last layer
    #
    #    
    #_____________________LOAD IMAGE DATA_______________________________________
    #
    def __load_image(self, impaths_temp):
        """Load image
            :param impaths: path to image to load
            :return: image
        """
        #
        image_org1 = PIL.Image.open(impaths_temp)         # READS THE ORIGINAL IMAGE
        gray_im1 = PIL.ImageOps.grayscale(image_org1)     # CONVERTS TO AN GRAY SCALE IMAGE
        image1 = np.array(gray_im1)                       # CONVERTS TO AN ARRAY
        image1 = np.array(PIL.Image.fromarray(image1).resize((self.imgsize, self.imgsize), PIL.Image.ANTIALIAS))
        # RESIZE THE IMAGE (QUALITY DEGRADES) REDUCED TO (256, 256)
        #
        #-------------------------------------------------------
        # DATA AGUMENTATION STEPS BEGINS
        #-------------------------------------------------------
        if self.augmentation == 'fliponly':
            # AGUMENTATION OPERATION ONLY WITH LR FLIP WITH PROBABILITY < 0.5    
            if random.random() < 0.5:
                image1 = np.fliplr(image1)
            #   
        else:                                    # No Agumentation
            pass
        #
        #-----------------------------------------------
        # DATA AGUMENTATION STEP ENDS
        #-----------------------------------------------
        image1=np.array(image1, dtype=np.float32)                # Needed after agumentation
        image1 = image1/(np.amax(image1) - np.amin(image1))      # normalizes the image
        #
        A = np.zeros([self.imgsize, self.imgsize, self.n_channels], dtype=np.float32)             # Dummy varaible to store image data
        #
        A[:,:,0]=image1
        A[:,:,1]=image1
        A[:,:,2]=image1
        #
        return A