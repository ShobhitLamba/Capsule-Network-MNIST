# Capsule Network running over MNIST data.

# The code for the capsule layers have been taken from https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Reshape, Input
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128
num_classes = 10
num_routing = 3
epochs = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.
input_shape = (1, img_rows, img_cols)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

# Building the Capsule Network
# Initializing the network
x = Input(shape = (x_train.shape[1:]))

# Conventional Convolution 2D layer
conv1 = Conv2D(filters = 256, kernel_size = 9, strides = 1, padding = "valid", activation = "relu", name = "conv1")(x)

# Convolution layer with 'squash' activation and reshape to [None, num_capsule, dim_capsule]
primarycaps = PrimaryCap(conv1, dim_capsule = 8, n_channels = 32, kernel_size = 9, strides = 2, padding = "valid")

# Capsule Layer with routing algorithm
digitcaps = CapsuleLayer(num_capsule = num_classes, dim_capsule = 16, routings = num_routing, name = "digitcaps")(primarycaps)

# Auxilliary layer to replace each capsule with its length
out_caps = Length(name = "capsnet")(digitcaps)

# Building the Decoder Network
y = Input(shape = (num_classes,))
masked_by_y = Mask()([digitcaps, y]) # True label used to mask the output
masked = Mask()(digitcaps) # Mask using the capsule with maximum length

# Shared Decoder model in training and prediction
decoder = Sequential(name = 'decoder')
decoder.add(Dense(512, activation = "relu", input_dim = 16 * num_classes))
decoder.add(Dense(1024, activation = "relu"))
decoder.add(Dense(np.prod(x_train.shape[1:]), activation = "sigmoid"))
decoder.add(Reshape(target_shape = input_shape, name = "out_recon"))

# Models for training
model = Model([x, y], [out_caps, decoder(masked_by_y)])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

model.compile(optimizer = keras.optimizers.Adam(lr = 0.001),
                    loss = [margin_loss, "mse"],
                    loss_weights = [1., 0.392],
                    metrics = {"capsnet": "accuracy"})

model.fit([x_train, y_train], [y_train, x_train], batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = ([x_test, y_test], [y_test, x_test]))
