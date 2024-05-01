""" FUNCTION FOR PREDICT PROBABILITY OF INDIVIDUAL IMAGES """
# Import libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow    as tf
import numpy         as np
import scipy
import PIL
from PIL             import Image, ImageOps
import nibabel       as nib
#
""" FUNCTION USE U-NET MODEL TO SEGMENT """
def segment_image(img_array, model):
    #
    image_arr =  ((img_array/(np.amax(img_array)))*255).clip(0,255) # converts to 0 to 255
    (w, h) = img_array.shape                              # STORE THE ORIGINAL SIZE OF THE IMAGE
    #-----------------------------------------
    #   PREDICTION BLOCK FOR U-Net (Start)
    #----------------------------------------
    image_resized=np.array(PIL.Image.fromarray(image_arr).resize((224, 224), PIL.Image.ANTIALIAS))
    # RESIZE THE IMAGE (QUALITY DEGRADES) REDUCED TO 224, 224
    imgunet= np.expand_dims(image_resized, axis=-1)       # Expand to make image shape from (224, 224) to (224, 224, 1)
    imgunet = np.expand_dims(imgunet, axis=0)             # Expand the Shape of image array into a format that u-net accepts = (1, 224, 224, 1)
    predictionunet=model.predict(imgunet, verbose = 0)    # Predicts the mask having shape (1, 224, 224, 1)
    mask=(predictionunet[0] >= 0.5).astype(np.uint8)      # Converts the prediction to binary mask having shape (224, 224, 1)
    mask=np.squeeze(mask, axis=2)                         # Essential to change the array shape (224, 224)
    mask=np.array(PIL.Image.fromarray(mask).resize((w, h), PIL.Image.ANTIALIAS)) # RESIZE THE MASK TO ORIGINAL SIZE
    mask=scipy.ndimage.binary_fill_holes(mask).astype(np.uint8)   # Fills the holes in the binary mask
    #------    SUBSTRACTED IMAGES --------
    imgunet=(np.array(image_arr)*mask).clip(0,255).astype(np.uint8) # Obtain substraction image of (original_image - mask)
    #
    #----------------------------------------------------
    #   DIVIDE THE SEGMENTED IMAGE INTO TWO HALVES
    #----------------------------------------------------
    #
    startRX, startRY, endRX, endRY, startLX, startLY, endLX, endLY = int(1), int(1), int(w/2), int(h), int(w/2), int(1), int(w), int(h)
    #
    # Cropping right breast
    right=np.float32(np.zeros([endRY-startRY, endRX-startRX]))         # Intialize the Crop image
    # Cropped the original image using bounding box coordinates
    for iy in range(startRY, endRY):
        for ix in range(startRX, endRX):
            right[iy-startRY,ix-startRX]=imgunet[iy,ix]
        #
    #
    right = np.array(PIL.Image.fromarray(right).resize((256, 256), PIL.Image.ANTIALIAS)) # Resize the right breast to (256, 256) input shape for classifier
    right = right/np.amax(right)
    right = (np.clip(right, 0, 1))*255.0                               # Force the image pixel between 0 to 255
    #
    img_r = np.zeros([256, 256, 3], dtype=np.float32) # 256 is the input size for classifier
    img_r[:,:,0]=right/255.0
    img_r[:,:,1]=right/255.0
    img_r[:,:,2]=right/255.0
    img_r = np.expand_dims(img_r, axis=0) # Changes the shape to (1, 256, 256, 3)
    #
    # Cropping left breast
    left=np.float32(np.zeros([endLY-startLY, endLX-startLX]))          # Intialize the Crop image
    # Cropped the original image using bounding box coordinates
    for iy in range(startLY, endLY):
        for ix in range(startLX, endLX):
            left[iy-startLY,ix-startLX]=imgunet[iy,ix]
        #
    #
    left = np.array(PIL.Image.fromarray(left).resize((256, 256), PIL.Image.ANTIALIAS)) # Resize the left to (256, 256) input shape for classifier
    left = left/np.amax(left)
    left = (np.clip(left, 0, 1))*255.0                                  # Force the image pixel between 0 to 255
    #
    img_l = np.zeros([256, 256, 3], dtype=np.float32) # 256 is the input size for classifier
    img_l[:,:,0]=(left/255.0)
    img_l[:,:,1]=(left/255.0)
    img_l[:,:,2]=(left/255.0)
    img_l = np.expand_dims(img_l, axis=0) # Changes the shape to (1, 256, 256, 3)
    #
    return img_r, img_l
#
""" FUNCTION TO USE CLASSIFICATION MODEL TO PROVIDE PROBABILITY """
def probability(img_array, model_unet, model_classifier):
    #
    # PROVIDES SEGMENTED AND SEPERATED IMAGES OF LEFT AND RIGHT BREAST
    img_right, img_left = segment_image(img_array, model_unet) # model u-net
    #
    # PROVIDES OUTPUT BY CLASSIFICATION NETWORK
    prediction_r=model_classifier.predict(img_right, verbose=0) # PROVIDES PROBABILITY
    prediction_l=model_classifier.predict(img_left, verbose=0) # PROVIDES PROBABILITY
    #
    return prediction_r, prediction_l
#
