# ---------------------------------------------------------------------------------
# THIS SCRIPT PERFORMS A DATAGENERATION STRUCTURE (CALLED FOR TRAINING)
# PERFORMS AGUMENTATION ON USER INPUT IN **PARAMS
#----------------------------------------------------------------------------------
# TO CALL THIS CLASS USE: DataGenerator(partition_name, img_path, mask_path, **params)
# 1.  partition_name:-        Can be training or valid
# 2.  img_path:-              Image path corresponding to that unique ID
# 3.  mask_path:-             Image mask path corresponding to that unique ID
# 4.  **params:-              Other default parameter transfered by user 
#
#-------------------------------------------------------------------------------------
# INSTRUCTION: THIS PROGRAM IS CALLED IN training_unet.ipynb or .py 
# CHECK THE training script PROGRAM to understand the calling method
#
#
# Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
# Memorial Sloan Kettering Cancer Center, NY (2022)
#-----------------------------------------------------------------------------------------------
# OUTPUT OF THIS PROGRAM IS EITHER TRAINING OR VALIDATION GENERATOR
# 
# IMPORT HELPER LIBRARIES
import numpy as np
#
# IMPORT OTHER PYTHON LIBARAIES
import tensorflow
import tensorflow.keras
from tensorflow import keras

class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, list_IDs, impaths, maskpaths, batch_size=32, dim=(224,224), n_channels=3, shuffle=True, imgsize=224):
        """Initialization
        :param list_IDs: list of all unique ID to be use in the generator
        :param impaths: path to images location (corresponding to unique ID)
        :param maskpaths: list of image mask (corresponding to unique ID)
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle unique indexes after every epoch
        :param imgsize: image size
        """
        self.list_IDs = list_IDs
        self.impaths = impaths
        self.maskpaths = maskpaths
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
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
        #
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        #
        return X, Y
    #    
    #        
    #___________________DATA generation functions _____________________________
    #
    def __data_generation(self, list_IDs_temp):
        #
        # Initialization
        X = np.empty((self.batch_size, *self.dim, (self.n_channels)), dtype=np.uint8)  # Note X has a shape of (32,224,224,1)
        Y = np.empty((self.batch_size, *self.dim, (self.n_channels)), dtype=np.bool)   # Note the Y has a shape of (32,224,224,1) for binary image
        #
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            impaths_temp=self.impaths[ID]             # image path corresponding to ID
            X[i] = self.__load_image(impaths_temp)    # NOTE X shape (32, 224, 224, 1)
            #
            maskpaths_temp=self.maskpaths[ID]         # mask path corresponding to ID
            Y[i] = self.__load_mask(maskpaths_temp)   # NOTE Y shape (32, 224, 224, 1)
        #
        return X, Y
    #
    #    
    #_____________________LOAD ORIGINAL IMAGE DATA_______________________________________
    #
    def __load_image(self, impaths_temp):
        """Load image
            :param impaths: path to image to load
        """
        # READS THE ORIGINAL IMAGE
        image = tensorflow.keras.preprocessing.image.load_img(impaths_temp, target_size=(self.imgsize, self.imgsize), color_mode="grayscale")
        image= np.expand_dims(image, axis=-1)  # Expand array to (w, h, 1)
        #
        return image
    #
    #_____________________LOAD MASK IMAGE DATA_______________________________________
    #
    def __load_mask(self, maskpaths_temp):
        """Load image mask
            :param maskpaths: path to mask to load
        """
        # READS THE ORIGINAL MASK
        image_msk = tensorflow.keras.preprocessing.image.load_img(maskpaths_temp, target_size=(self.imgsize, self.imgsize), color_mode="grayscale")
        #
        image_msk= np.expand_dims(image_msk, axis=-1)  # Expand array to (w, h, 1)
        #
        return image_msk