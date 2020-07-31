# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:48:18 2020

@author: MA
"""
import os
import nibabel as nib
import numpy as np
join = os.path.join
from skimage import io

# set path
train_img_path = 'path to MICCAI2020/OriData/TNSCUI2020/TNSCUI2020_train/image'
train_gt_path = 'path to MICCAI2020/OriData/TNSCUI2020/TNSCUI2020_train/mask'
test_img_path = 'path to MICCAI2020/OriData/TNSCUI2020/tnscui2020_testset'
save_path = 'path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D'

# convert training png images to nii
for name in os.listdir(train_img_path):
    # image: png to nifti
    img_png = io.imread(join(train_img_path, name))
    img_3d = np.expand_dims(img_png, -1).repeat(3, -1)
    img_nii = nib.Nifti1Image(img_3d.astype(np.uint8), np.eye(4))
    # ground truth: png to nifti
    gt_png = io.imread(join(train_gt_path, name))>0 # label value should be 1
    gt_3d = np.expand_dims(gt_png, -1).repeat(3, -1)
    gt_nii = nib.Nifti1Image(gt_3d.astype(np.uint8), np.eye(4))
    # save results
    name_pre = name.split('.PNG')[0]
    nib.save(img_nii, join(save_path, '/imagesTr/'+name_pre+'_0000.nii.gz'))
    nib.save(gt_nii, join(save_path, 'labelsTr/'+name_pre+'.nii.gz'))

# convert testing png images to nii
for name in os.listdir(test_img_path):
    img_png = io.imread(join(test_img_path, name))
    img_3d = np.expand_dims(img_png, -1).repeat(3, -1)
    img_nii = nib.Nifti1Image(img_3d.astype(np.uint8), np.eye(4))
    name_pre = name.split('.PNG')[0]
    nib.save(img_nii, join(save_path, '/imagesTs/'+name_pre+'_0000.nii.gz'))








