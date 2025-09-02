#ensure you're in parent directory of b2T when running
import torchvision.models as models


import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import clip

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.waterbirds import Waterbirds, get_transform_cub

from tqdm import tqdm
import os
import time
import pandas as pd
import math

import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_Waterbirds_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_Waterbirds_erm.pth, best_model_Waterbirds_dro.pth
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
args = parse_args()

if args.dataset == 'waterbird':
    preprocess = get_transform_cub()
    class_names = ['landbird', 'waterbird']
    # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
    image_dir = '../data/cub/data/waterbird_complete95_forest2water2/'
    val_dataset = Waterbirds(data_dir='../data/cub/data/waterbird_complete95_forest2water2', split='test', transform=preprocess) # test for fig 5
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    class_names = ['not blond', 'blond']
    # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
    image_dir = '../data/celebA/data/img_align_celeba/'
    val_dataset = CelebA(data_dir='../data/celebA/data/', split='test', transform=preprocess) # test for fig 5
else:
    print("Invalid dataset!")

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)




result_dir = ''
model_dir = '../model/'

# correctify dataset
result_path = result_dir + "raw_logits" +"_"+ args.dataset + ".csv"
if args.model.endswith(".pth"):
    if not args.model.endswith("erm.pth"):
        print("Please use ERM model to replicate figure 5")
    else:
        model = torch.load(model_dir + args.model,map_location=device)
else:
    print(f"invalid model {args.model}")
    pass

model = model.to(device)
model.eval()
start_time = time.time()
result = {"image":[],
        "pred":[],
        "logit_0":[],
        "logit_1":[],
        "actual":[],
        "group":[],
        "spurious":[],                
        "correct":[],
        }

with torch.no_grad():
    running_corrects = 0
    for (images, (targets, targets_g, targets_s), index, paths) in tqdm(val_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(len(preds)):
            image = paths[i]
            pred = preds[i]
            actual = targets[i]
            output = outputs[i]
            logit_0 = output[0]
            logit_1= output[1]
            group = targets_g[i]
            spurious = targets_s[i]
            result['image'].append(image)
            result['pred'].append(pred.item())


            result['logit_0'].append(logit_0.item()) # 0 class label logit
            result['logit_1'].append(logit_1.item()) # 1 class label logit


            result['actual'].append(actual.item())
            result['group'].append(group.item())
            result['spurious'].append(spurious.item())
            if pred == actual:
                    result['correct'].append(1)
                    running_corrects += 1
            else:
                    result['correct'].append(0)

df = pd.DataFrame(result)
df.to_csv(result_path)
print("Extracted logits to ",result_path)
