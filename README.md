# 3 Steps Are All You Need to Achieve SOTA in MICCAI 2020 Thyroid Nodule Segmentation Challenge

Segmentation is the most popular tasks in MICCAI 2020 challenges, including 15 out of 24 challenges. In this tutorial, we focus on the segmentation task in  [thyroid nodule segmentation and classification challenge (TN-SCUI 2020)](https://tn-scui2020.grand-challenge.org/). In particular, we show how to use U-Net with 3 steps to achieve IoU 0.8093 on testing set, which is very close to the top 1 score (0.8254, 550+ participants) on the [leaderboard](https://tn-scui2020.grand-challenge.org/evaluation/leaderboard/).


## Task and Dataset

The target is to segment thyroid nodules from ultrasound (US) images. 

-  The training set consists of 3644 images with `png` format (1641 benign cases and 2003 malignant cases). The annotations are binary images with value `{0,255}`.
-  The testing set consists of 910 images where 400 images are randomly selected as validation set, and 510 images are used for final ranking.

Let's show two examples in the training set.

![benign-12_demo](https://github.com/JunMa11/TNSCUI2020/blob/master/Img/benign-12_demo.png)

![malignant-2369_demo](https://github.com/JunMa11/TNSCUI2020/blob/master/Img/malignant-2369_demo.png)

## Step 1. Preparing environment and training data

Our solution is based on U-Net with its great extension [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to the out-of-the-box and flexible nature of nnU-Net, we can easily adapt the training set to the required dataset format of nnU-Net.

### 1.1 Installation

- Ubuntu 16.04 or 18.04

- Install [PyTorch](https://pytorch.org/get-started/locally/) (1.3+)

- Install [Nvidia Apex](https://github.com/NVIDIA/apex). 

  ```python
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir ./
  ```

- Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

  ```python
  git clone https://github.com/MIC-DKFZ/nnUNet.git
  cd nnUNet
  pip install -e .
  ```

- Set path in `nnUNet/nnunet/paths.py`

  ```python
  # line 29: 
  base = 'your path to store training data, e.g., ./MICCAI2020/nnUNetData'
  # line 30
  preprocessing_output_dir = 'your path to store preprocessing data, e.g., ./MICCAI2020/nnUNetData/pre_data' # SSD is highly recommanded
  # line 31
  network_training_output_dir_base = 'your path to save trained models, e.g., ./MICCAI2020/Models'
  ```

### 1.2 Preparing dataset

Create following folders

```python
    MICCAI2020/
    ├── nnUNetData
    │   └── nnUNet_raw_data
    │       └── Task600_Thyroid2D
    │           └── imagesTr
    │           └── imagesTs
    │           └── labelsTr

    ├── Models
    ├── OriData
    │   └── TNSCUI2020
    │       └── TNSCUI2020_train # directly decompressing TNSCUI2020_train.rar
    │       └── tnscui2020_testset # directly decompressing tnscui2020_testset.rar
```

nnU-Net is designed for 3D images with `nifti` format, while the data format in thyroid nodule task is the 2D image with `png` format. We can expand all the 2D images with an additional dimension and convert them to `nifti` format with this [code](https://github.com/JunMa11/TNSCUI2020/blob/master/utils/Step1_preparing_data.py). Now, the files in `MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D` are

```python
    MICCAI2020/
    ├── nnUNetData
    │   └── nnUNet_raw_data
    │       └── Task600_Thyroid2D
    │           └── imagesTr
    │               ├── 2_0000.nii.gz
    │               ├── 4_0000.nii.gz
    │               ├── ..._0000.nii.gz
    │           └── imagesTs
    │               ├── test_1_0000.nii.gz
    │               ├── test_2_0000.nii.gz
    │               ├── test_..._0000.nii.gz
    │           └── labelsTr
    │               ├── 2.nii.gz
    │               ├── 4.nii.gz
    │               ├── ....nii.gz
    │           └── dataset.json # download from https://github.com/JunMa11/TNSCUI2020/blob/master/Task600_Thyroid2D/dataset.json
```

Open terminal and run
`nnUNet_plan_and_preprocess -t 600 --verify_dataset_integrity`

Data ready! 

Next, we can train 2D U-Net models.

## Step 2. Training five-fold cross validation models

- Network hyperparameters

![Network hyperparameters](https://github.com/JunMa11/TNSCUI2020/blob/master/Img/NetworkHyperparameters.PNG)

- Train five models for cross validation. Open terminal and run

```python
nnUNet_train 2d nnUNetTrainerV2 Task600_Thyroid2D 0
nnUNet_train 2d nnUNetTrainerV2 Task600_Thyroid2D 1
nnUNet_train 2d nnUNetTrainerV2 Task600_Thyroid2D 2
nnUNet_train 2d nnUNetTrainerV2 Task600_Thyroid2D 3
nnUNet_train 2d nnUNetTrainerV2 Task600_Thyroid2D 4
```

The five trained models will be automatically saved in `MICCAI2020/Models/nnUNet/2d/Task600_Thyroid2D/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0,1,2,3,4`

    MICCAI2020/
    ├── Models
    │   └── nnUNet
    │       └── 2d
    │           └── Task600_Thyroid2D
    │               └── nnUNetTrainerV2__nnUNetPlansv2.1
    │                   └── fold_0
    │                       ├── model_final_checkpoint.model
    │                       ├── model_final_checkpoint.model.pkl
    │                   └── fold_1
    │                       ├── model_final_checkpoint.model
    │                       ├── model_final_checkpoint.model.pkl
    │                   └── fold_2
    │                       ├── model_final_checkpoint.model
    │                       ├── model_final_checkpoint.model.pkl
    │                   └── fold_3
    │                       ├── model_final_checkpoint.model
    │                       ├── model_final_checkpoint.model.pkl
    │                   └── fold_4
    │                       ├── model_final_checkpoint.model
    │                       ├── model_final_checkpoint.model.pkl
    │                   └── plans.pkl # 



## Step 3. Inferring testing set and Submission

Inferring testing set by five-model ensemble

```python
nnUNet_predict -i path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D/imagesTs -o path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D/UNet_Submission_NII/ -t Task600_Thyroid2D -m 2d
```

The inference results are in

```python
    MICCAI2020/
    ├── nnUNetData
    │   └── nnUNet_raw_data
    │       └── Task600_Thyroid2D
    │           └── UNet_Submission_NII
    │               ├── test_1.nii.gz
    │               ├── test_2.nii.gz
    │               ├── test_....nii.gz
```

Then, we convert the `nifti` files to `PNG` format.

```python
import nibabel as nib
from skimage import io
import os
join = os.path.join

seg_path = 'path to MICCAI2020/nnUNetData/nnUNet_raw_data/Task600_Thyroid2D/TestSet_NII_Results_NaiveUNet_Task600/'
save_path = 'path to MICCAI2020/OriData/TNSCUI2020/TestSet_PNG_Results_NaiveUNet_Task600/'

for i in range(1, 911):
    seg = nib.load(join(seg_path, 'test_'+str(i)+'.nii')).get_fdata()
    seg_2d = seg_data[:,:,1]
    io.imsave(join(save_path, 'test_'+str(i)+'.PNG'), seg_2d)
```

**The most exciting moment comes!**

Zip the folder `TestSet_PNG_Results_NaiveUNet_Task600` and submit it to the [official portal](https://tn-scui2020.grand-challenge.org/evaluation/submissions/create/).

The results obtain IoU 0.8093, which is very close to the Top 1 IoU 0.8254 on the [leaderboard](https://tn-scui2020.grand-challenge.org/evaluation/leaderboard/).



## Two unsuccessful attempts

### Cascaded pipeline

Two U-Nets are employed.
- Step 1. Train U-Net to segment thyroid nodules from original ultrasound images.
- Step 2. Crop the nodule region of interest (ROI).
- Step 3. Train the new U-Net based on ROI images.

This strategy obtains remarkable improvements during five-fold cross validation. However, the performance gains **do not** generalize to testing set.

### Pesudo label learning

- Step 1. Generate pseudo test set labels: using the baseline U-Net to segment the testing set.
- Step 2. Train a new U-Net based on test set images and the pseudo labels.
- Step 3. Finetune the trained U-Net on training set.



## Five-fold cross validation results (Dice)

| Fold | Naive U-Net | Cascaded U-Net | Pseudo label |
| :--: | :---------: | :------------: | :----------: |
|  0   |   0.8747    |     0.9132     |    0.8770    |
|  1   |   0.8873    |     0.9147     |    0.8882    |
|  2   |   0.8808    |     0.9123     |    0.8799    |
|  3   |   0.8701    |     0.9147     |    0.8771    |
|  4   |   0.8719    |     0.9105     |    0.8735    |

> Trained models are publicly available at [here](http://doi.org/10.5281/zenodo.3965648).

## Future

When beginning with a medical image segmentation challenge, many participants would take U-Net as their first try. Currently, such trained baseline models are not widely shared between participants due to privacy considerations. This might be a great waste of time and energy.

In the future segmentation challenges, we hope the trained baseline models can be freely shared by the organizers and participants at the beginning of the challenge, which could reduce repeatedly training the same models.


## Acknowledgment

We highly appreciate TN-SCUI 2020 organizers for the great challenge, and all authors of nnU-Net for the out-of-the-box method.


