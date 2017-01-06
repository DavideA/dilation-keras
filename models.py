from keras.models import Model
from keras.layers import Dropout, Activation, UpSampling2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, Input, AtrousConvolution2D
from keras.layers.core import Reshape, Permute
from datasets import CONFIG


# CITYSCAPES MODEL
def get_dilation_model_cityscapes(input_dim, output_dim, weights_file=None):

    h, w = input_dim
    h_out, w_out = output_dim

    model_in = Input(shape=(3, h, w))
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(19, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(19, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(19, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = AtrousConvolution2D(19, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(19, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(19, 1, 1, name='ctx_final')(h)

    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    h = UpSampling2D(size=(8, 8))(h)
    h = Convolution2D(19, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

    h = Permute((2, 3, 1))(h)
    h = Reshape((h_out*w_out, 19))(h)
    h = Activation('softmax', name='prob')(h)
    model_out = Reshape((h_out, w_out, 19))(h)

    model = Model(input=model_in, output=model_out)

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


# PASCAL VOC MODEL
def get_dilation_model_voc(input_dim, output_dim, weights_file=None):

    h, w = input_dim
    h_out, w_out = output_dim

    model_in = Input(shape=(3, h, w))
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(21, 1, 1, activation='relu', name='fc-final')(h)
    h = ZeroPadding2D(padding=(33, 33))(h)
    h = Convolution2D(42, 3, 3, activation='relu', name='ct_conv1_1')(h)
    h = Convolution2D(42, 3, 3, activation='relu', name='ct_conv1_2')(h)
    h = AtrousConvolution2D(84, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1')(h)
    h = AtrousConvolution2D(168, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1')(h)
    h = AtrousConvolution2D(336, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1')(h)
    h = AtrousConvolution2D(672, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1')(h)
    h = Convolution2D(672, 3, 3, activation='relu', name='ct_fc1')(h)
    h = Convolution2D(21, 1, 1, name='ct_final')(h)

    h = Permute((2, 3, 1))(h)
    h = Reshape((h_out * w_out, 21))(h)
    h = Activation('softmax')(h)
    model_out = Reshape((h_out, w_out, 21))(h)

    model = Model(input=model_in, output=model_out)

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


# KITTI MODEL
def get_dilation_model_kitti(input_dim, output_dim, weights_file=None):

    h, w = input_dim
    h_out, w_out = output_dim

    model_in = Input(shape=(3, h, w))
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(11, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(11, 1, 1, name='ctx_final')(h)

    h = Permute((2, 3, 1))(h)
    h = Reshape((h_out * w_out, 11))(h)
    h = Activation('softmax')(h)
    model_out = Reshape((h_out, w_out, 11))(h)

    model = Model(input=model_in, output=model_out)

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


# CAMVID MODEL
def get_dilation_model_camvid(input_dim, output_dim, weights_file=None):

    h, w = input_dim
    h_out, w_out = output_dim

    model_in = Input(shape=(3, h, w))
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(11, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(11, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(11, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(11, 1, 1, name='ctx_final')(h)
    h = Permute((2, 3, 1))(h)
    h = Reshape((h_out * w_out, 11))(h)
    h = Activation('softmax')(h)
    model_out = Reshape((h_out, w_out, 11))(h)

    model = Model(input=model_in, output=model_out)

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


# wrapper function
def get_dilation_model(dataset):

    # get dataset input dimension and infer output dimension
    h, w = CONFIG[dataset]['input_dim']
    h_out = (h - 2 * CONFIG[dataset]['conv_margin']) / CONFIG[dataset]['zoom']
    w_out = (w - 2 * CONFIG[dataset]['conv_margin']) / CONFIG[dataset]['zoom']

    # get weights file
    weights_file = CONFIG[dataset]['weights_file']

    if dataset == 'cityscapes':
        model = get_dilation_model_cityscapes(input_dim=(h, w), output_dim=(h_out, w_out), weights_file=weights_file)
    elif dataset == 'voc12':
        model = get_dilation_model_voc(input_dim=(h, w), output_dim=(h_out, w_out), weights_file=weights_file)
    elif dataset == 'kitti':
        model = get_dilation_model_kitti(input_dim=(h, w), output_dim=(h_out, w_out), weights_file=weights_file)
    elif dataset == 'camvid':
        model = get_dilation_model_camvid(input_dim=(h, w), output_dim=(h_out, w_out), weights_file=weights_file)
    else:
        raise 'Dataset not available: {}'.format(dataset)

    return model
