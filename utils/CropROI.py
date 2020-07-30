# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:15:00 2020

@author: MA
"""

import os
import nibabel as nib
import numpy as np
from skimage import measure
join = os.path.join

img_path = r'path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D/imagesTr'
gt_path = r'path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D/labelsTr'
save_path = r'path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task603_ThyroidROI'

if os.path.exists(join(save_path, 'imagesTr')) is not True:
    os.mkdir(join(save_path, 'imagesTr'))
if os.path.exists(join(save_path, 'imagesTs')) is not True:
    os.mkdir(join(save_path, 'imagesTs'))
if os.path.exists(join(save_path, 'labelsTr')) is not True:
    os.mkdir(join(save_path, 'labelsTr'))


names = os.listdir(gt_path)
names.sort()
shift = 10
lowest_size = 360

for name in names:
    img_name = name.split('.nii.gz')[0] + '_0000.nii.gz'
    img = nib.load(join(img_path, img_name)).get_fdata()
    gt = nib.load(join(gt_path, name)).get_fdata()
    # compute bbox
    region_prop = measure.regionprops(gt.astype(np.uint8))
    img_bbox = region_prop[0].bbox
    width = img_bbox[3]-img_bbox[0]
    height = img_bbox[4]-img_bbox[1]
    
    if width<lowest_size:
        w_shift_temp = np.uint((lowest_size-width)/2)
        xmin = np.max([0, img_bbox[0]-w_shift_temp]);xmax = np.min([gt.shape[0], img_bbox[3]+w_shift_temp])
    else:
        xmin = np.max([0, img_bbox[0]-shift]); xmax = np.min([gt.shape[0], img_bbox[3]+shift])
    if height<lowest_size:
        h_shift_temp = np.uint((lowest_size-height)/2)
        ymin = np.max([0, img_bbox[1]-h_shift_temp]); ymax = np.min([gt.shape[1], img_bbox[4]+h_shift_temp])
    else:  
         ymin = np.max([0, img_bbox[1]-shift]); ymax = np.min([gt.shape[1], img_bbox[4]+shift])
    img_roi = img[xmin:xmax, ymin:ymax, :]; gt_roi = gt[xmin:xmax, ymin:ymax, :]

    nib.save(nib.Nifti1Image(img_roi, np.eye(4)), join(save_path, 'imagesTr/'+img_name))
    nib.save(nib.Nifti1Image(gt_roi.astype(np.uint8), np.eye(4)), join(save_path, 'labelsTr/'+name))  