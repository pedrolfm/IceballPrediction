#! /usr/bin/python

import torch
from configparser import ConfigParser
from monai.transforms import (
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandAdjustContrastd,
    RandFlipd,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,    
    Spacingd,
    ToTensord,
    DataStatsd,
)

from monai.utils import first, set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import numpy as np
import glob
import os
import shutil

from monai.data import CacheDataset, DataLoader, Dataset


#--------------------------------------------------------------------------------
# Load configurations
#--------------------------------------------------------------------------------

class Param():

    def __init__(self, filename='config.ini'):
        self.config = ConfigParser()
        self.config.read(filename)
        self.readParameters()

    def getvector(self, config, section, key, default=None):
        value = None
        if default:
            value = config.get(section, key, fallback=default)
        else:
            value = config.get(section, key)
        if value:
            value = value.split(',')
            value = [float(s) for s in value]
            value = tuple(value)
            return value
        else:
            return None

    def readParameters(self):
 
        self.data_dir = self.config.get('common', 'data_dir')
        self.root_dir = self.config.get('common', 'root_dir')
        
        self.pixel_dim = self.getvector(self.config, 'common', 'pixel_dim')
        if self.pixel_dim == None:
            self.pixel_dim = (1.0,1.0,1.0)
        
        self.window_size = self.getvector(self.config, 'common', 'window_size')
        if self.window_size:
            self.window_size = [int(s) for s in self.window_size]
            self.window_size = tuple(self.window_size)
        else:
            self.window_size = (160,160,16)

        self.pixel_intensity_scaling = self.config.get('common', 'pixel_intensity_scaling')
        self.pixel_intensity_min = self.config.getfloat('common', 'pixel_intensity_min')
        self.pixel_intensity_max = self.config.getfloat('common', 'pixel_intensity_max')
        self.pixel_intensity_percentile_min = self.config.getfloat('common', 'pixel_intensity_percentile_min')
        self.pixel_intensity_percentile_max = self.config.getfloat('common', 'pixel_intensity_percentile_max')
        
        self.model_file = self.config.get('common', 'model_file')

        self.in_channels = int(self.config.get('common', 'in_channels'))
        self.out_channels = int(self.config.get('common', 'out_channels'))


class TrainingParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.use_tensorboard = int(self.config.get('training', 'use_tensorboard'))
        self.use_matplotlib = int(self.config.get('training', 'use_matplotlib'))
        self.training_name = self.config.get('training', 'training_name')
        self.max_epochs = int(self.config.get('training', 'max_epochs', fallback='200'))
        self.training_device_name = self.config.get('training', 'training_device_name')
        self.training_rand_rot = int(self.config.get('training', 'random_rot', fallback='0'))
        if self.training_rand_rot ==1:
            self.training_rand_rot_angle = tuple(self.getvector(self.config, 'training', 'random_rot_angle', '0.0,0.0,0.2617993877991494'))
            self.training_rand_rot_scale = tuple(self.getvector(self.config, 'training', 'random_rot_scale', '0.1,0.1,0.1'))
        self.training_rand_flip = int(self.config.get('training', 'random_flip', fallback='0'))
        self.training_rand_shift_intensity = float(self.config.get('training', 'random_shift_intensity', fallback='0.0'))
        self.training_rand_contrast = int(self.config.get('training', 'random_contrast', fallback='0'))
        self.training_rand_scale = float(self.config.get('training', 'random_scale', fallback='0.0'))

class TestParam(Param):

    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.test_device_name = self.config.get('test', 'test_device_name')

        
class TransferParam(TrainingParam):
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()
        self.tl_model_file = self.config.get('transfer', 'tl_model_file')
        self.tl_name = self.config.get('transfer', 'tl_name', fallback='transfer_learning_1')
        self.tl_data_dir = self.config.get('transfer', 'tl_data_dir')
    
        
class InferenceParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.inference_device_name = self.config.get('inference', 'inference_device_name')


#--------------------------------------------------------------------------------
# Load Transforms
#--------------------------------------------------------------------------------

def loadTrainingTransforms(param):

    scaleIntensity = None
    if param.pixel_intensity_scaling == 'absolute':
        print('Intensity scaling by max/min')
        scaleIntensity = ScaleIntensityRanged(
            keys=["image"], a_min=param.pixel_intensity_min, a_max=param.pixel_intensity_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    elif param.pixel_intensity_scaling == 'percentile':
        print('Intensity scaling by percentile')
        scaleIntensity = ScaleIntensityRangePercentilesd(
            keys=["image"], lower=param.pixel_intensity_percentile_min, upper=param.pixel_intensity_percentile_max,
            b_min=0.0, b_max=1.0, clip=True,
            )
    else: # 'normalize
        scaleIntensity = NormalizeIntensityd(keys=["image"])

    transform_array = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        scaleIntensity,
        DataStatsd(keys=['image', 'label'], data_value=False),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=param.window_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        ToTensord(keys=["image", "label"]),
    ]

    print("PEDRO")
    print(param.training_rand_rot)
    if param.training_rand_rot == 1:
        transform_array.append(
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0,
                spatial_size=param.window_size,
                rotate_range=param.training_rand_rot_angle,
                scale_range=param.training_rand_rot_scale)
        )
        
    if param.training_rand_flip == 1:
        transform_array.append(
            RandFlipd(
                keys=['image', 'label'],
                prob=0.5,
                spatial_axis=0 # TODO: Make sure that the axis corresponds to L-R
            )
        )

    if param.training_rand_shift_intensity > 0.0:
        transform_array.append(
            RandStdShiftIntensityd(
                keys=['image'],
                prob=1.0,
                factors=param.training_rand_shift_intensity
            )
        )
        
    if param.training_rand_contrast == 1:
        transform_array.append(
            RandAdjustContrastd(
                keys=['image'],
                prob=1.0,
                gamma=(0.5,1.5)
            )
        )

    if param.training_rand_scale > 0.0:
        transform_array.append(
            RandScaleIntensityd(
                keys=['image'],
                prob=1.0,
                factors=param.training_rand_scale
            )
        )
        
    
    train_transforms = Compose(transform_array)

    return train_transforms



def loadValidationTransforms(param):
    
    if param.pixel_intensity_scaling == 'absolute':
        print('Intensity scaling by max/min')
        scaleIntensity = ScaleIntensityRanged(
            keys=["image"], a_min=param.pixel_intensity_min, a_max=param.pixel_intensity_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    elif param.pixel_intensity_scaling == 'percentile':
        print('Intensity scaling by percentile')
        scaleIntensity = ScaleIntensityRangePercentilesd(
            keys=["image"], lower=param.pixel_intensity_percentile_min, upper=param.pixel_intensity_percentile_max,
            b_min=0.0, b_max=1.0, clip=True,
            )
    else: # 'normalize
        scaleIntensity = NormalizeIntensityd(keys=["image"])
        
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="LPS"),
            scaleIntensity,
            DataStatsd(keys=['image', 'label'], data_value=False),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=param.window_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return val_transforms


def loadInferenceTransforms(param, output_path):
    
    if param.pixel_intensity_scaling == 'absolute':
        print('Intensity scaling by max/min')
        scaleIntensity = ScaleIntensityRanged(
            keys=["image"], a_min=param.pixel_intensity_min, a_max=param.pixel_intensity_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    elif param.pixel_intensity_scaling == 'percentile':
        print('Intensity scaling by percentile')
        scaleIntensity = ScaleIntensityRangePercentilesd(
            keys=["image"], lower=param.pixel_intensity_percentile_min, upper=param.pixel_intensity_percentile_max,
            b_min=0.0, b_max=1.0, clip=True,
            )
    else: # 'normalize
        scaleIntensity = NormalizeIntensityd(keys=["image"])
        
    pre_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=param.pixel_dim, mode=("bilinear")),
            Orientationd(keys=["image"], axcodes="LPS"),
            scaleIntensity,
            CropForegroundd(keys=["image"], source_key="image"),
            #ToTensord(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )


    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=pre_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                              # then invert `pred` based on this information. we can use same info
                              # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                             # for example, may need the `affine` to invert `Spacingd` transform,
                                             # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                           # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                           # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        Activationsd(keys="pred", sigmoid=True),
        #AsDiscreted(keys="pred", threshold_values=True),
        AsDiscreted(keys="pred", argmax=True, num_classes=param.out_channels),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False),
    ])

    
    return (pre_transforms, post_transforms)

    

#--------------------------------------------------------------------------------
# Generate a file list
#--------------------------------------------------------------------------------

def generateLabeledFileList(srcdir, prefix):
    
    print('Reading labeled images from: ' + srcdir)
    images = sorted(glob.glob(os.path.join(srcdir, prefix + "_images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(srcdir, prefix + "_labels", "*.nii.gz")))
    
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    return data_dicts


def generateFileList(srcdir):
    
    print('Reading images from: ' + srcdir)
    images = sorted(glob.glob(os.path.join(srcdir, "*.nii.gz")))
    
    data_dicts = [
        {"image": image_name} for image_name in images
    ]

    return data_dicts
    

#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

def setupModel(param):

    model_unet = UNet(
        spatial_dims=3,
        in_channels=param.in_channels,
        out_channels=param.out_channels,
        channels=(16, 32, 64, 128,256),
        #channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        #norm=Norm.BATCH,
    )
    
    post_pred = AsDiscrete(argmax=True, to_onehot=param.out_channels, n_classes=param.out_channels)
    post_label = AsDiscrete(to_onehot=param.out_channels, n_classes=param.out_channels)

    return (model_unet, post_pred, post_label)
