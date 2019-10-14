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
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

from keras import Input, Model
from keras.layers import Dense, Convolution3D, Flatten


try:
    data_dir = Path(__file__).resolve().parent.parent.joinpath("data/")
except NameError:
    data_dir = Path(__name__).resolve().parent.parent.joinpath("data/")

# Raise error if deepdrug directory not found.
assert data_dir.joinpath("deepdrug3d_voxel_data/").is_dir(), \
       "No directory deepdrug3d_voxel_data/ found in data/"

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

# Have balanced samples for training.
random.seed(0)
n_sample = min(len(control_list), len(heme_list),
               len(nucleotide_list), len(steroid_list))

# TODO: Think about naming ?
small_control_list = random.sample(control_list, n_sample)
small_heme_list = random.sample(heme_list, n_sample)
small_nucleotide_list = random.sample(nucleotide_list, n_sample)
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
    elif filename.stem in small_steroid_list:
        y.append("steroid")
        x.append(np.load(filename))
    else:
#        print("Unrecognized file")
        pass

# Shuffle the list before converting to array.
#!!!: Redim the y too then !
#random.shuffle(x)

x_redim = [np.squeeze(arr) for arr in x]
x_array = np.stack(x_redim)



# control, heme, nucleotide, steroid
one_hot_y = pd.get_dummies(pd.Series(y)).values


def my_model():
    inputs = Input(shape=(14, 32, 32, 32))
    conv_1 = Convolution3D(
            input_shape=(14,32,32,32),
            filters=4,
            kernel_size=5,
            padding='valid',  # It seems using padding same causes problems
            activation="relu",
            data_format='channels_first',
        )(inputs)
    flat_1 = Flatten()(conv_1)
    output = Dense(4, activation="softmax")(flat_1)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


from keras.optimizers import Adam
def test_model():
    inputs = Input(shape=(14, 32, 32, 32))
    conv_1 = Convolution3D(
            input_shape=(14,32,32,32),
            filters=4,
            kernel_size=5,
            padding='same',  # It seems using padding same causes problems
            activation="relu",
            kernel_initializer="he_normal",
            data_format='channels_first',
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
            data_format='channels_first'
        )(conv_1)
    flat_1 = Flatten()(maxp_1)
    output = Dense(4, activation="softmax")(flat_1)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    adam = Adam(lr=0.1)

    model.compile(optimizer=adam,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = test_model()


np.random.seed(0)
model.fit(x_array[:10], one_hot_y[:10], batch_size=20, epochs=30, validation_split=0.2)


np.random.seed(0)
model.fit(x_array, one_hot_y, batch_size=20, epochs=20, validation_split=0.2)

model.predict(x_array, one_hot_y)
#np.unique(x_train[0])

# =============================================================================
# See shape
# =============================================================================

from keras.models import Sequential
from keras.layers import LeakyReLU, Dropout, MaxPooling3D, Activation


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
        model.add(Dense(4))
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