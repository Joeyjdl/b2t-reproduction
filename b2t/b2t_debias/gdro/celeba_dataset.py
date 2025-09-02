import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # allow us to find the celeba file
from data.celeba import CelebA as CelebA_parentfolder # instead of using datasets.CelebA (not compatible in this torch version) 
from PIL import Image

#class CustomCelebA(datasets.CelebA):
class CustomCelebA(CelebA_parentfolder):
    def __init__(self, root, split, target_attr, bias_attr, transform, pseudo_bias=None):
        super(CustomCelebA, self).__init__(root, split, transform=transform) # changed download to false, use downloader.py/experiment.py to download
        
        #self.targets = self.attr[:, target_attr]
        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias)
        else:
            #self.biases = self.attr[:, bias_attr]
            self.biases = self.confounder_array
        
    def __getitem__(self, index):
        img_filename = os.path.join(self.data_dir, 'img_align_celeba', self.filename_array[index])
        X = Image.open(img_filename).convert('RGB')
        y = self.targets[index]
        a = self.biases[index]
        
        if self.transform is not None:
            X = self.transform(X)
            
        ret_obj = {'x': X,
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   'filename': self.filename_array[index],
                   }

        return ret_obj
