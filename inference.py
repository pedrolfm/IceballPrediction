#! /usr/bin/python

from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.metrics import compute_meandice, DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, NiftiSaver, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

import numpy as np
import torch
import tempfile
import shutil
import os
import glob
import sys
import argparse
import time


from common import *


class prediction:

  def __init__(self, config, inputPath, outputPath,imgType,model):
    
    self.config_file = config
    self.input_path = inputPath
    self.output_path = outputPath
    self.image_type = imgType
    self.model_file = model


  def boundaryPrediction(self):

    print('Loading parameters from: ' + self.config_file)
    param = InferenceParam(self.config_file)
    files = generateFileList(self.input_path)
    n_files = len(files)
    print('# of images: ' + str(n_files))
    

    st = time.time()
    self.run(param, self.output_path, self.image_type, files, self.model_file)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

  def run(self,param, output_path, image_type, val_files, model_file):

    device = torch.device(param.inference_device_name)

    (pre_transforms, post_transforms) =  loadInferenceTransforms(param, output_path)

    val_ds = CacheDataset(data=val_files, transform=pre_transforms, cache_rate=1.0, num_workers=4)
    #val_ds = Dataset(data=val_files, transform=pre_transforms)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
 
    
    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------
    
    (model_unet, post_pred, post_label) = setupModel(param)
    
    model = model_unet.to(device)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    model.load_state_dict(torch.load(os.path.join(param.root_dir, model_file), map_location=device))


    
    #--------------------------------------------------------------------------------
    # Validate
    #--------------------------------------------------------------------------------
    
    model.eval()
    
    with torch.no_grad():
    
        #saver = NiftiSaver(output_dir=output_path, separate_folder=False)
        metric_sum = 0.0
        metric_count = 0
        
        for i, val_data in enumerate(val_loader):
            roi_size = param.window_size
            sw_batch_size = 4
            
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)
            #val_output_label = torch.argmax(val_outputs, dim=1, keepdim=True)
            #saver.save_batch(val_output_label, val_data['image_meta_dict'])            

            



