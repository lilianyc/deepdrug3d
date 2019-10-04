#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:21:17 2019

@author: lyang_crosson
"""

from pathlib import Path
import numpy as np

NB_PROT = 10

# MAke sure in right dir
# numpy loading directly is horrendously slow.
prot_list = [file for file in Path(".").glob("../data/deepdrug3d_voxel_data/*.npy")
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