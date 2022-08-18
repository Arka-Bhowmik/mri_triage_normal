def unet(img_size, channels):
    #
    # IMPORT NECCESSARY LIBRARIES
    import tensorflow as tf
    #
    #
    # -------------------------------------------------------------------------------------------------
    # THIS SCRIPT IS CALLED FOR U-Net MODEL TRAINING
    #--------------------------------------------------------------------------------------------------
    # TO CALL THIS FUNCTION USE: 
    # custom_model(img_size, channels)
    # 1.  img_size:-       Input image size (w,h)
    # 2.  channels:-       input image channels
    #-------------------------------------------------------------------------------------
    # INSTRUCTION: THIS PROGRAM IS CALLED IN unet_training.ipynb or .py 
    # CHECK THE training PROGRAM to understand the calling method
    #
    #
    # Created by: Arka Bhowmik and Sarah Eskreis-Winkler, 
    # Memorial Sloan Kettering Cancer Center, NY (2022)
    #-----------------------------------------------------------------------------------------------
    # THE SCRIPT PROVIDES MODEL OUTPUT
    #-----------------------------------------------------------------------------------------------
    #
    # DEFINE TWO LAYER OF 2D CONVOLUTION TO BE CALLED FOR DOWNSAMPLING AND UPSAMPLING
    # HERE n-filters is number of filters and x is input
    def double_conv_block(x, n_filters):
        # First Conv2D and elu activation
        x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        # Second Conv2D and elu activation
        x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "elu", kernel_initializer = "he_normal")(x)
        return x
    #
    # DEFINE DOWNSAMPLING BLOCK FOR ENCODER WITH SKIP CONNECTIONS TO CONNECT DECODER
    # HERE f is skip connections and p is downsmpling output
    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = tf.keras.layers.MaxPool2D(2)(f)
        p = tf.keras.layers.Dropout(0.3)(p)
        return f, p
    #
    # DEFINE UPSAMPLING BLOCK FOR DECODER THAT ACCEPTS SKIPPED FEATURES
    # HERE SKIPPED CONNECTIONS ARE MADE BY conv_features
    def upsample_block(x, conv_features, n_filters):
        # upsample
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = tf.keras.layers.concatenate([x, conv_features])
        # dropout
        x = tf.keras.layers.Dropout(0.3)(x)
        # Conv2D twice with elu activation
        x = double_conv_block(x, n_filters)
        return x
    #
    #
    #
    # FINALLY BUILD A U-Net model BY calling the sub-functions above
    #
    # inputs
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, channels)) # input image channel is one/grayscale
    #
    # Normalization layer
    NL = tf.keras.layers.Lambda(lambda x: x / 255) (inputs)
    #-----------------------------------------
    # Encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(NL, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    #--------------------------------------
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    #-------------------------------------
    # Decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    #-----------------------------------
    #
    # outputs
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation = "sigmoid")(u9)
    #
    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="U-Net")
    #
    return model
