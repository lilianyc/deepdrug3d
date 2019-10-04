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

TOGH C1 dataset

# Pockets bind
nucleotide, heme and steroid

took representative sites with TC
"""

from pathlib import Path
import numpy as np

data_dir = Path(__name__).resolve().parent.parent.joinpath("data/")
NB_PROT = 10

# numpy loading directly is horrendously slow.
prot_list = [file for file in data_dir.glob("deepdrug3d_voxel_data/*.npy")
             if not file.name.startswith("._")]

# list(Path(".").glob("../data/deepdrug3d_voxel_data/[!._]*.npy"))
x_train = [np.load(file) for file in prot_list[:NB_PROT]]

np.unique(x_train[0])

# See how data changed
# d = c.reshape(32,32,32,14)
# np.squeeze

# =============================================================================
# See shape
# =============================================================================

tmp = np.load(prot_list[0])
tmp_shp = tmp.reshape(14,32,32,32)
tmp_squ = np.squeeze(tmp)

tmp_shp.shape
tmp_squ.shape

(tmp_shp == tmp_squ).all()

# see reshape vs move axis