import torchvision.models as models


import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import clip

# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.fairfaces import Fairfaces
from data.waterbirds import Waterbirds, get_transform_cub

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity

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
    parser.add_argument("--extract_caption", action='store_true',help="extract captions from images")
    parser.add_argument("--save_result", default = True)
    parser.add_argument("--minority_ratio", type = float,default=0.05,help="fairfaces minority ratio in data.")
    parser.add_argument("--minority_group", type = int,default=3,help="fairfaces minority group that will be 'mispredicted' ")
    parser.add_argument("--override_result", action='store_true',help="override stored result, without necessarily having to extract caption(still can though)")
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
args = parse_args()

if args.dataset == 'waterbird':
    preprocess = get_transform_cub()
    class_names = ['landbird', 'waterbird']
    # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
    image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
    caption_dir = 'data/cub/caption/'
    val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    class_names = ['not blond', 'blond']
    # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
    image_dir = 'data/celebA/data/img_align_celeba/'
    caption_dir = 'data/celebA/caption/'
    val_dataset = CelebA(data_dir='data/celebA/data/', split='val', transform=preprocess)
elif args.dataset == 'fairfaces':
    preprocess = get_transform_cub() # cub transform is general
    print("Recommend running fairfaces initially with -1 as minority_ratio and extract_caption , allowing you to only have to extract_caption once! (all data will be captioned). Runs after this may use --override_result")
    class_names = ['male', 'female']
    image_dir = 'data/fairfaces_data/' 
    caption_dir = 'data/fairfaces_data/caption/'
    val_dataset = Fairfaces(data_dir=image_dir, split='val', transform=preprocess,minority_ratio=args.minority_ratio,minority_group=args.minority_group)
elif args.dataset == 'imageNet':
    print("Imagenet")
else:
    print(f"Invalid dataset {args.dataset}")
    pass

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)




result_dir = 'result/'
model_dir = 'model/'
diff_dir = 'diff/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(diff_dir):
    os.makedirs(diff_dir)

# extract caption
if args.extract_caption:
    print("Start extracting captions..")
    for x, (y, y_group, y_spurious), idx, path in tqdm(val_dataset):
        image_path = image_dir + path
        caption = extract_caption(image_path)
        if not os.path.exists(caption_dir):
            os.makedirs(caption_dir)
        caption_path = caption_dir + path.split("/")[-1][:-4] + ".txt"
        with open(caption_path, 'w') as f:
            f.write(caption)
    print("Captions of {} images extracted".format(len(val_dataset)))

# correctify dataset
result_path = result_dir + args.dataset +"_" +  args.model.split(".")[0] + ".csv"
if not os.path.exists(result_path)  or args.extract_caption:
    if args.model.endswith(".pth"):
        model = torch.load(model_dir + args.model)
    elif args.model == "Resnet50":
        model = models.resnet50(pretrained=True)
    else:
        print(f"invalid model {args.model}")
        pass

    model = model.to(device)
    model.eval()
    start_time = time.time()
    print("Pretrained model \"{}\" loaded".format(args.model))

    result = {"image":[],
            "pred":[],
            "actual":[],
            "group":[],
            "spurious":[],                
            "correct":[],
            "caption":[],
            }

    with torch.no_grad():
        running_corrects = 0
        for (images, (targets, targets_g, targets_s), index, paths) in tqdm(val_dataloader):
            if args.dataset != 'fairfaces':
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            else:
                preds = targets # no model for fairfaces -> use targets as preds
            for i in range(len(preds)):
                image = paths[i]
                pred = preds[i]
                actual = targets[i]
                group = targets_g[i]
                spurious = targets_s[i]
                caption_path = caption_dir + image.split("/")[-1][:-4] + ".txt"
                with open(caption_path, "r") as f:
                    caption = f.readline()
                result['image'].append(image)
                result['pred'].append(pred.item())
                result['actual'].append(actual.item())
                result['group'].append(group.item())
                result['spurious'].append(spurious.item())
                result['caption'].append(caption)
                if pred == actual:
                        result['correct'].append(1)
                        running_corrects += 1
                else:
                        result['correct'].append(0)

        print("# of correct examples : ", running_corrects)
        print("# of wrong examples : ", len(val_dataset) - running_corrects)
        print("# of all examples : ", len(val_dataset))
        print("Accuracy : {:.2f} %".format(running_corrects/len(val_dataset)*100))

    df = pd.DataFrame(result)
    df.to_csv(result_path)
    print("Classified result stored")
else: 
    df = pd.read_csv(result_path)
    print("Classified result \"{}\" loaded".format(result_path))
    if args.override_result:    # override known results 
        result = {"indices":[]}
        with torch.no_grad():
            running_corrects = 0
            for (_, (_, _, _), index, _) in tqdm(val_dataloader):
                for i in range(len(index)):
                    idx = index[i]
                    result['indices'].append(idx)

        # Check if original
        try:
            df = df.loc[result['indices']]
        except KeyError as e:  # KeyError occurs if indices are not found
            raise ValueError(
            "Overriding stored predictions failed, please first run fairfaces with -1 as minority_ratio and extract_caption to caption all data first."
            ) from e
        print("Session will use filtered indices based on input arguments")


# fairfaces labelling
if args.dataset == "fairfaces":
    indices_minority = df[df['group'] == val_dataloader.dataset.minority_group].index
    minority_class = math.ceil((val_dataloader.dataset.minority_group - 1) / 2)
    indices_other_class = df[df['actual'] != minority_class].index
    df.loc[indices_minority, "correct"] = 0 # maybe random set if want only subset
    df.loc[indices_other_class[0], "correct"] = 0 # placeholder such that every label has one mispred, so regular still code works
    

# extract keyword
df_class_0 = df[df['actual'] == 0] # not blond, landbird
df_class_1 = df[df['actual'] == 1] # blond, waterbird

caption_class_0 = ' '.join(df_class_0['caption'].tolist())
caption_class_1 = ' '.join(df_class_1['caption'].tolist())
keywords_class_0 = extract_keyword(caption_class_0, num_keywords=40)
keywords_class_1 = extract_keyword(caption_class_1, num_keywords=40)
    


################## From here 

# Similarity per keyword
similarity_class_0 = calc_similarity(image_dir, df_class_0['image'], keywords_class_0) if df_class_0['image'].count() != 0 else 0

similarity_class_1 = calc_similarity(image_dir, df_class_1['image'], keywords_class_1) if df_class_1['image'].count() != 0 else 0

# New clip score 
dist_class_0 = similarity_class_0 - similarity_class_0.mean().item() 
dist_class_1 = similarity_class_1 - similarity_class_1.mean().item() 

################## To here need to implement our own scoring system
if (args.dataset != "fairfaces")  or (args.dataset == "fairfaces" and minority_class == 0): # only print the minority group stats
    print("Result for class :", class_names[0])
    diff_0 = print_similarity(keywords_class_0, keywords_class_1, dist_class_0, dist_class_1, df_class_0)
    print("*"*60)

    if args.save_result:
        diff_path_0 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0] + ".csv"
        diff_0.to_csv(diff_path_0)

if (args.dataset != "fairfaces")  or (args.dataset == "fairfaces" and minority_class == 1):
    print("Result for class :", class_names[1])
    diff_1 = print_similarity(keywords_class_1, keywords_class_0, dist_class_1, dist_class_0, df_class_1)
    
    if args.save_result:
        diff_path_1 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[1] + ".csv"
        diff_1.to_csv(diff_path_1)
