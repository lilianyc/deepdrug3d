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


# Pockets bind
nucleotide, heme and steroid

took representative sites with TC

!!!: it seems the steroid might not be considered as a class
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import Input, Model
from keras.layers import Dense, Convolution3D, Flatten, Dropout, MaxPooling3D
from sklearn.metrics import roc_curve, auc

# Importing from keras.layers or keras.layers.core/convolutional
# has no effect.

# =============================================================================
# Functions definition.
# =============================================================================

def plot_roc(pred, y, title=None, col=None):
    """Plot a ROC curve.

    Inspired by t81_558_class_04_2_multi_class by Jeff Heaton.

    """
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    title = title if title else 'Receiver Operating Characteristic (ROC)'

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, color=col)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# Channel first simple model.
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
    conv_2 = Convolution3D(
            filters=32,
            kernel_size=3,
            padding='valid',     # Padding method
            data_format='channels_first',
        )(conv_1)
    maxp3d_1 = MaxPooling3D(
            pool_size=(2,2,2),
            strides=None,
            padding='valid',    # Padding method
            data_format='channels_first'
        )(conv_2)
    drop_1 = Dropout(0.4)(maxp3d_1)
    flat_1 = Flatten()(drop_1)
    output = Dense(3, activation="softmax", kernel_initializer='he_normal')(flat_1)
    model = Model(inputs=inputs, outputs=output)

    print(model.summary())
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# =============================================================================
# Verify if the data is there.
# =============================================================================

try:
    data_dir = Path(__file__).resolve().parent.parent.joinpath("data/")
except NameError:
    data_dir = Path(__name__).resolve().parent.parent.joinpath("data/")

# Raise error if deepdrug directory not found.
assert data_dir.joinpath("deepdrug3d_voxel_data/").is_dir(), \
       "No directory deepdrug3d_voxel_data/ found in data/"

# =============================================================================
# Load the data.
# =============================================================================

# Create the list of pockets.
# numpy loading directly is horrendously slow.
# list(Path(".").glob("../data/deepdrug3d_voxel_data/[!._]*.npy"))
prot_list = [file for file in data_dir.glob("deepdrug3d_voxel_data/*.npy")
             if not file.name.startswith("._")]

# Assign class filenames to lists.
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
random.seed(2)
n_sample = min(len(control_list), len(heme_list),
               len(nucleotide_list), len(steroid_list))

# Unbalance the sample a little bit.
small_control_list = random.sample(control_list, 150+random.randint(-15,15))
small_heme_list = random.sample(heme_list, 150+random.randint(-15,15))
small_nucleotide_list = random.sample(nucleotide_list, 150+random.randint(-15,15))
small_steroid_list = random.sample(steroid_list, n_sample)


# TODO: Combine the class information from the files to the list in a func.
# !!!: Not resistant to reordering/index changes.
x = []
y = []
# Load filenames in samples.
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
        pass


# Normalize dimension shape to (14, 32, 32, 32).
x_redim = [np.squeeze(arr) for arr in x]
# Convert array list to a numpy array, np.array can also fill the job.
x_array = np.stack(x_redim)  #The problem does not seem to come from here

# For use without channel_first
new_x_array = np.moveaxis(x_array, 1, -1)

# One hot encode, sorted in alphabetical order: control, heme, nucleotide.
one_hot_y = pd.get_dummies(pd.Series(y)).values

# =============================================================================
# Create and train the model.
# =============================================================================

# Channel first.
np.random.seed(0)
model = my_model()
np.random.seed(0)
#model.fit(x_array, one_hot_y, batch_size=20, epochs=20, validation_split=0.2)
history = model.fit(np.array(x_redim), one_hot_y, batch_size=20, epochs=20, validation_split=0.2)

#model.predict(x_array)
pred = model.predict(np.array(x_redim))

# The model only predicts one column/class.
print("numbers of classes predicted:\n"
      "control: {:.0f}, heme {:.0f}, nucleotide: {:.0f}".format(*pred.sum(axis=0)))

# =============================================================================
# Visualisation.
# =============================================================================

# Plot the loss.
plt.figure()
plt.plot(history.history["loss"], "-o")
plt.title("Loss by epoch")
plt.xticks(range(20))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot the accuracy.
plt.figure()
plt.plot(history.history["acc"], "-o")
plt.title("Accuracy by epoch")
plt.xticks(range(20))
plt.ylim(0, 1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
# It seems like the accuracy has small changes, at 1e-8 precision.


#from keras.utils import plot_model  # Needs pydot
#plot_model(model)


# ROC curves of control, heme and nucleotide.
plot_roc(pred[:, 0], one_hot_y[:, 0], "ROC of control")
plot_roc(pred[:, 1], one_hot_y[:, 1], "ROC of heme", col="green")
plot_roc(pred[:, 2], one_hot_y[:, 2], "ROC of nucleotide", col="brown")


# =============================================================================
# Alternative models. Unused
# =============================================================================


from keras.models import Sequential
from keras.layers import LeakyReLU, Activation
from keras.layers import add, AveragePooling3D

# Module for a RNN.
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

# RNN.
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


# Model from the article, readapted.
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
