#--------------------------------------------------------------------------
# DEFINES A CUSTOM METRICS FOR CALCULATION OF DSC loss, IOU, Ensemble loss
#--------------------------------------------------------------------------
#
# IMPORT HELPER LIBRARIES
#
import tensorflow as tf
import numpy as np
#
#
def dice_loss(y_true, y_pred):
    #
    # COMPUTE THE LOSS OF DICE COEFFICIENT
    #
    # Here the flatten converts multi-dimensional input tensors into a single dimension
    # Smooth is used to avoid dividibg by zero
    #
    smooth=tf.keras.backend.epsilon()
    #
    #
    y_true_flat = tf.keras.backend.flatten(tf.cast(y_true, dtype='float32'))
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat)
    dice_loss = 1 - tf.keras.backend.mean((2. * intersection + smooth)/(union + smooth))
    #
    return dice_loss
#
#
#
def IoU_acc(y_true, y_pred):
    #
    # metric determines intersection over union
    # Here the flatten converts multi-dimensional input tensors into a single dimension
    # Smooth is used to avoid dividibg by zero
    #
    smooth=tf.keras.backend.epsilon()
    #
    #
    y_true_flat = tf.keras.backend.flatten(tf.cast(y_true, dtype='float32'))
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    #
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    total = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat)
    union = total - intersection
    #
    IoU_acc = (intersection + smooth) / (union + smooth)             # For better overlap this value should increase with epooch
    #
    return IoU_acc
#
#
def dice_bce_loss(y_true, y_pred):
    # Ensemble loss assocaited with binary cross-entropy and loss of dice coefficient
    # Here the flatten converts multi-dimensional input tensors into a single dimension
    # Smooth is used to avoid dividibg by zero
    #
    smooth=tf.keras.backend.epsilon()
    #
    #
    y_true=tf.cast(y_true, dtype='float32')         # Essential to convert data type bool to float32
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    BCE =  tf.keras.backend.binary_crossentropy(y_true, y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat)
    dice_acc = tf.keras.backend.mean((2. * intersection + smooth)/(union + smooth))
    #
    dice_bce_loss=BCE+(1 - dice_acc)
    #
    return dice_bce_loss
#