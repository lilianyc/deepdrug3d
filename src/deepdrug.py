#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:21:17 2019

@author: lyang_crosson


Visualize 3*3*3*14 voxel ?
What about change in 3d orientation of voxel ? should not because of CNN

different batch sizes of 16, 32, 64, and 128 yield a comparable
performance, and 50 epochs are sufficient to reach the convergence. A 5-fold cross-validation

see CAM for significant binding regions ?

TOUGH C1 dataset

# Pockets bind
nucleotide, heme and steroid

took representative sites with TC

!!!: it seems the steroid might not be considered as a class
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd


from keras import Input, Model
from keras.layers import Dense, Convolution3D, Flatten

#import keras
#from importlib import reload
#import os; os.environ['KERAS_BACKEND'] = 'theano_backend';reload(keras.backend)
# =============================================================================
# 
# =============================================================================

try:
    data_dir = Path(__file__).resolve().parent.parent.joinpath("data/")
except NameError:
    data_dir = Path(__name__).resolve().parent.parent.joinpath("data/")

# Raise error if deepdrug directory not found.
assert data_dir.joinpath("deepdrug3d_voxel_data/").is_dir(), \
       "No directory deepdrug3d_voxel_data/ found in data/"

# =============================================================================
# Loading the data
# =============================================================================

# numpy loading directly is horrendously slow.
# list(Path(".").glob("../data/deepdrug3d_voxel_data/[!._]*.npy"))
prot_list = [file for file in data_dir.glob("deepdrug3d_voxel_data/*.npy")
             if not file.name.startswith("._")]

# Assign class filenames to lists
with open(data_dir.joinpath("control.list"), "r") as file1, \
     open(data_dir.joinpath("heme.list"), "r") as file2, \
     open(data_dir.joinpath("nucleotide.list"), "r") as file3, \
     open(data_dir.joinpath("steroid.list"), "r") as file4:

    control_list = [filename.strip() for filename in file1.readlines()]
    heme_list = [filename.strip() for filename in file2.readlines()]
    nucleotide_list = [filename.strip() for filename in file3.readlines()]
    steroid_list = [filename.strip() for filename in file4.readlines()]

# =============================================================================
# Create x and y.
# =============================================================================

# Have balanced samples for training.
random.seed(0)
n_sample = min(len(control_list), len(heme_list),
               len(nucleotide_list), len(steroid_list))

# TODO: Think about naming ?
small_control_list = random.sample(control_list, 100)
small_heme_list = random.sample(heme_list, 10)
small_nucleotide_list = random.sample(nucleotide_list, 90)
small_steroid_list = random.sample(steroid_list, n_sample)


# TODO: Combine the class information from the files to the list in a func.
# !!!: Not resistant to reordering/index changes.
x = []
y = []
# Ugly, use a dict ?
for filename in prot_list:
    if filename.stem in small_control_list:
        y.append("control")
        x.append(np.load(filename))
    elif filename.stem in small_heme_list:
        y.append("heme")
        x.append(np.load(filename))
    elif filename.stem in small_nucleotide_list:
        y.append("nucleotide")
        x.append(np.load(filename))
#    elif filename.stem in small_steroid_list:
#        y.append("steroid")
#        x.append(np.load(filename))
    else:
#        print("Unrecognized file")
        pass

# =============================================================================
# Alternative method to get x.
# =============================================================================
# Inspired by train.py in the original deepdrug repo.
from keras.utils import np_utils
import os

voxel_folder = str(data_dir.joinpath("deepdrug3d_voxel_data/"))

atps = random.sample(nucleotide_list, 50)
hemes = random.sample(heme_list, 50)
controls = random.sample(control_list, 50)

L = len(atps) + len(hemes) + len(controls)
voxel = np.zeros(shape = (L, 14, 32, 32, 32),
        dtype = np.float64)
label = np.zeros(shape = (L,), dtype = int)
cnt = 0
print('...Loading the data')

for filename in prot_list:
    protein_name = filename.stem
#    full_path = voxel_folder + '/' + filename
    if protein_name in atps + hemes +controls:
        temp = np.load(filename)
        voxel[cnt,:] = temp
    if protein_name in atps:
        label[cnt] = 0
    elif protein_name in hemes:
        label[cnt] = 1
    elif protein_name in controls:
        label[cnt] = 2
    else:
#        print(protein_name)
        continue
    cnt += 1
    
y = np_utils.to_categorical(label, num_classes=3)
model = my_model()
model.fit(voxel, y, epochs=20, batch_size=20, validation_split=0.2)
pred = model.predict(voxel)

# =============================================================================
# END
# =============================================================================

# Shuffle the list before converting to array.
#!!!: Redim the y too then !
#random.shuffle(x)

x_redim = [np.squeeze(arr) for arr in x]
x_array = np.stack(x_redim)  #The problem does not seem to come from here
# However, can convert list directly to array

new_x_array = np.moveaxis(x_array, 1, -1)

# control, heme, nucleotide, steroid
one_hot_y = pd.get_dummies(pd.Series(y)).values


# Channel first simple model.
# It seems using too much filters in conv_1 or using another conv bugs.
def my_model():
    inputs = Input(shape=(14, 32, 32, 32))
    conv_1 = Convolution3D(
            input_shape=(14,32,32,32),
            filters=64,
            kernel_size=5,
            padding='valid',  # It seems using padding same causes problems
            activation="relu",
            data_format='channels_first',
        )(inputs)
#    conv_2 = Convolution3D(
#            filters=32,
#            kernel_size=3,
#            padding='valid',     # Padding method
#            data_format='channels_first',
#        )(conv_1)
    flat_1 = Flatten()(conv_1)
    output = Dense(3, activation="softmax")(flat_1)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


from keras.optimizers import Adam
from keras.layers import MaxPooling3D
def test_model():
    inputs = Input(shape=(32, 32, 32, 14))
    conv_1 = Convolution3D(
            input_shape=(32,32,32, 14),
            filters=64,
            kernel_size=5,
            padding='same',  # It seems using padding same causes problems
            activation="relu",
            kernel_initializer="he_normal",
#            data_format='channels_first',
        )(inputs)
#    conv_2 = Convolution3D(
#            filters=64,
#            kernel_size=3,
#            padding='valid',     # Padding method
#            data_format='channels_first',
#        )(conv_1)
    maxp_1 = MaxPooling3D(
            pool_size=(2,2,2),
            strides=None,
            padding='valid',    # Padding method
#            data_format='channels_first'
        )(conv_1)
    flat_1 = Flatten()(maxp_1)
    output = Dense(3, activation="softmax")(flat_1)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    adam = Adam(lr=0.1)

    model.compile(optimizer=adam,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = test_model()



np.random.seed(0)
model.fit(new_x_array, one_hot_y, batch_size=20, epochs=20, validation_split=0.2)
#model.fit(np.array(x_redim), one_hot_y, batch_size=32, epochs=30, validation_split=0.2)
pred = model.predict(new_x_array)
# It seems the model only predicts the second column
[ind_pred for ind_pred in pred if (ind_pred != np.array([0,1,0])).all()]

# Channel first.
np.random.seed(0)
model = my_model()
model.fit(x_array, one_hot_y, batch_size=20, epochs=20, validation_split=0.2)

model.predict(x_array)
#np.unique(x_train[0])

# =============================================================================
# See shape
# =============================================================================


from keras.models import Sequential
from keras.layers import LeakyReLU, Dropout, MaxPooling3D, Activation


from keras.layers import add, AveragePooling3D, Activation
def residual_module(lay_i, n_filters):
    # check if the number of filters needs to be increased, assumes
    # channels last format.
    if lay_i.shape[-1] != n_filters:
        lay_i = Convolution3D(n_filters, (1), padding="same", activation="relu",
                       kernel_initializer="he_normal", data_format='channels_first')(lay_i)
    save = lay_i
    conv_1 = Convolution3D(n_filters, (3), padding="same", activation="relu",
                    kernel_initializer="he_normal", data_format='channels_first')(lay_i)
    conv_2 = Convolution3D(n_filters, (3), padding="same", activation="linear",
                    kernel_initializer="he_normal", data_format='channels_first')(conv_1)
    conc_1 = add([conv_2, save])
    output = Activation("relu")(conc_1)

    return output

def my_model7():
    n_residual = 2
    print("Simple residual network with {} modules".format(n_residual))
    inputs = Input(shape=(32, 32, 32, 14))
    residual_i = inputs
    for _ in range(n_residual):
        residual_i = residual_module(residual_i, 20)

    # !! Padding to not lose dimension.
    gavg_1 = AveragePooling3D((2, 2, 2), strides=(1), padding="same")(residual_i)
    flat_1 = Flatten()(gavg_1)
    output = Dense(3, activation="softmax")(flat_1)

    model = Model(inputs=inputs, outputs=output)

    print(model.summary())

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model



def build():
        model = Sequential()
        # Conv layer 1
        model.add(Convolution3D(
            input_shape = (14,32,32,32),
            filters=64,
            kernel_size=5,
            padding='valid',     # Padding method
            data_format='channels_first',
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 1
        model.add(Dropout(0.2))
        # Conv layer 2
        model.add(Convolution3D(
            filters=64,
            kernel_size=3,
            padding='valid',     # Padding method
            data_format='channels_first',
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Maxpooling 1
        model.add(MaxPooling3D(
            pool_size=(2,2,2),
            strides=None,
            padding='valid',    # Padding method
            data_format='channels_first'
        ))
        # Dropout 2
        model.add(Dropout(0.4))
        # FC 1
        model.add(Flatten())
        model.add(Dense(128)) # TODO changed to 64 for the CAM
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 3
        model.add(Dropout(0.4))
        # Fully connected layer 2 to shape (2) for 2 classes
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
        return model
model = build()

# See how data changed
# d = c.reshape(32,32,32,14)
# np.squeeze + vstack

tmp = np.load(prot_list[0])
tmp_shp = tmp.reshape(14,32,32,32)
tmp_squ = np.squeeze(tmp)

tmp_shp.shape
tmp_squ.shape

(tmp_shp == tmp_squ).all()

# see reshape vs move axis

# dict to conv, (), mask ?