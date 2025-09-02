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
import os
import clip
import torch
from pathlib import Path
import skimage.io as io
from PIL import Image
from tqdm import tqdm

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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
    val_dataset = Waterbirds(data_dir='../data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)  # val & test for fig 5 failure direction
    test_dataset = Waterbirds(data_dir='../data/cub/data/waterbird_complete95_forest2water2', split='test', transform=preprocess)  # val & test for fig 5 failure direction
    image_dir = '../data/cub/data/waterbird_complete95_forest2water2/'
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    test_dataset = CelebA(data_dir='../data/celebA/data/', split='test', transform=preprocess) # val & test for fig 5 failure direction
    val_dataset = CelebA(data_dir='../data/celebA/data/', split='val', transform=preprocess) # val & test for fig 5 failure direction
    image_dir = '../data/celebA/data/img_align_celeba/'
else:
    print("Invalid dataset!")

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=4, drop_last=False)




result_dir = ''
model_dir = '../model/'

# correctify dataset
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
            result['actual'].append(actual.item())
            result['group'].append(group.item())
            result['spurious'].append(spurious.item())
            if pred == actual:
                    result['correct'].append(1)
                    running_corrects += 1
            else:
                    result['correct'].append(0)

df = pd.DataFrame(result)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load('ViT-B/32', device)

svms = []
val_avgs = []
val_stds = []
for i in np.unique(df['actual']):
    class_indices = df[df['actual'] == i].index
    val_embed_list = []
    images = val_dataset.filename_array[class_indices]
    images = [image_dir + image for image in images]
    images = [Image.fromarray(io.imread(image)) for image in images]
    image_list_chunked = list_chunk(images, 2000)

    for image_list in tqdm(image_list_chunked):


        # Prepare the inputs
        image_inputs = torch.cat([preprocess_clip(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)
        # Calculate features
        with torch.no_grad():
            image_features = model_clip.encode_image(image_inputs)
        val_embed_list.append(image_features)

    val_embed_list = torch.cat(val_embed_list,dim=0).cpu().to(torch.float32)
    val_embed_list_avg = val_embed_list.mean(dim=0)
    val_embed_list_std = val_embed_list.std(dim=0)

    epsilon = 1e-8
    # Standardize
    val_embed_list = (val_embed_list - val_embed_list_avg) / (val_embed_list_std + epsilon)
    val_embed_list /= val_embed_list.norm(dim=-1, keepdim=True)




    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV

    # Define SVM with hyperparameter tuning
    param_grid = {'C': np.logspace(-6, 0, 7)} 
    svm = LinearSVC(class_weight='balanced',loss='hinge')

    grid_search = GridSearchCV(svm, param_grid, scoring='balanced_accuracy', cv=2,n_jobs=-1)
    grid_search.fit(val_embed_list.numpy(), df['correct'][class_indices])

    best_svm = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    print("Best C param found : ", best_C)
    print("Best SVM score found : ", grid_search.best_score_)
    svms.append(best_svm)
    val_avgs.append(val_embed_list_avg)
    val_stds.append(val_embed_list_std)




test_embed_list = []

images = test_dataset.filename_array
images = [image_dir + image for image in images]
images = [Image.fromarray(io.imread(image)) for image in images]

similarity_list = []
image_list_chunked = list_chunk(images, 2000)

for image_list in tqdm(image_list_chunked):


    # Prepare the inputs
    image_inputs = torch.cat([preprocess_clip(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)
    # Calculate features
    with torch.no_grad():
        image_features = model_clip.encode_image(image_inputs)
    
                                        #image_features /= image_features.norm(dim=-1, keepdim=True)
    test_embed_list.append(image_features)

test_embed_list = torch.cat(test_embed_list,dim=0).cpu().to(torch.float32)


for i in np.unique(df['actual']):
    test_embed_list[test_dataloader.dataset.targets == i] = (test_embed_list[test_dataloader.dataset.targets == i] - val_avgs[i]) / (val_stds[i] + epsilon)

test_embed_list /= test_embed_list.norm(dim=-1, keepdim=True)

test_embed_list = test_embed_list.numpy()

test_decision_values = np.zeros_like(test_embed_list[:,0])
for i in np.unique(df['actual']):
    test_decision_values[test_dataset.targets == i] = svms[i].decision_function(test_embed_list[test_dataset.targets == i])

result = {"image":[],
        "actual":[],
        "group":[],
        "spurious":[]                
        }

with torch.no_grad():
    running_corrects = 0
    for (images, (targets, targets_g, targets_s), index, paths) in tqdm(test_dataloader):
        _, preds = torch.max(outputs, 1)
        for i in range(len(paths)):
            image = paths[i]
            actual = targets[i]
            group = targets_g[i]
            spurious = targets_s[i]
            result['image'].append(image)
            result['actual'].append(actual.item())
            result['group'].append(group.item())
            result['spurious'].append(spurious.item())

df = pd.DataFrame(result)
df['svm_score'] = test_decision_values

result_path = result_dir + "failure_direction" +"_"+ args.dataset + ".csv"
df.to_csv(result_path)
print("Extracted failure direction SVM scores to ",result_path)
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
import os
import clip
import torch
from pathlib import Path
import skimage.io as io
from PIL import Image
from tqdm import tqdm

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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
    val_dataset = Waterbirds(data_dir='../data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)  # val & test for fig 5 failure direction
    test_dataset = Waterbirds(data_dir='../data/cub/data/waterbird_complete95_forest2water2', split='test', transform=preprocess)  # val & test for fig 5 failure direction
    image_dir = '../data/cub/data/waterbird_complete95_forest2water2/'
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    test_dataset = CelebA(data_dir='../data/celebA/data/', split='test', transform=preprocess) # val & test for fig 5 failure direction
    val_dataset = CelebA(data_dir='../data/celebA/data/', split='val', transform=preprocess) # val & test for fig 5 failure direction
    image_dir = '../data/celebA/data/img_align_celeba/'
else:
    print("Invalid dataset!")

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=4, drop_last=False)




result_dir = ''
model_dir = '../model/'

# correctify dataset
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
            result['actual'].append(actual.item())
            result['group'].append(group.item())
            result['spurious'].append(spurious.item())
            if pred == actual:
                    result['correct'].append(1)
                    running_corrects += 1
            else:
                    result['correct'].append(0)

df = pd.DataFrame(result)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load('ViT-B/32', device)

svms = []
val_avgs = []
val_stds = []
for i in np.unique(df['actual']):
    class_indices = df[df['actual'] == i].index
    val_embed_list = []
    images = val_dataset.filename_array[class_indices]
    images = [image_dir + image for image in images]
    images = [Image.fromarray(io.imread(image)) for image in images]
    image_list_chunked = list_chunk(images, 2000)

    for image_list in tqdm(image_list_chunked):


        # Prepare the inputs
        image_inputs = torch.cat([preprocess_clip(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)
        # Calculate features
        with torch.no_grad():
            image_features = model_clip.encode_image(image_inputs)
        val_embed_list.append(image_features)

    val_embed_list = torch.cat(val_embed_list,dim=0).cpu().to(torch.float32)
    val_embed_list_avg = val_embed_list.mean(dim=0)
    val_embed_list_std = val_embed_list.std(dim=0)

    epsilon = 1e-8
    # Standardize
    val_embed_list = (val_embed_list - val_embed_list_avg) / (val_embed_list_std + epsilon)
    val_embed_list /= val_embed_list.norm(dim=-1, keepdim=True)




    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV

    # Define SVM with hyperparameter tuning
    param_grid = {'C': np.logspace(-6, 0, 7)} 
    svm = LinearSVC(class_weight='balanced',loss='hinge')

    grid_search = GridSearchCV(svm, param_grid, scoring='balanced_accuracy', cv=2,n_jobs=-1)
    grid_search.fit(val_embed_list.numpy(), df['correct'][class_indices])

    best_svm = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    print("Best C param found : ", best_C)
    print("Best SVM score found : ", grid_search.best_score_)
    svms.append(best_svm)
    val_avgs.append(val_embed_list_avg)
    val_stds.append(val_embed_list_std)




test_embed_list = []

images = test_dataset.filename_array
images = [image_dir + image for image in images]
images = [Image.fromarray(io.imread(image)) for image in images]

similarity_list = []
image_list_chunked = list_chunk(images, 2000)

for image_list in tqdm(image_list_chunked):


    # Prepare the inputs
    image_inputs = torch.cat([preprocess_clip(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)
    # Calculate features
    with torch.no_grad():
        image_features = model_clip.encode_image(image_inputs)
    
                                        #image_features /= image_features.norm(dim=-1, keepdim=True)
    test_embed_list.append(image_features)

test_embed_list = torch.cat(test_embed_list,dim=0).cpu().to(torch.float32)


for i in np.unique(df['actual']):
    test_embed_list[test_dataloader.dataset.targets == i] = (test_embed_list[test_dataloader.dataset.targets == i] - val_avgs[i]) / (val_stds[i] + epsilon)

test_embed_list /= test_embed_list.norm(dim=-1, keepdim=True)

test_embed_list = test_embed_list.numpy()

test_decision_values = np.zeros_like(test_embed_list[:,0])
for i in np.unique(df['actual']):
    test_decision_values[test_dataset.targets == i] = svms[i].decision_function(test_embed_list[test_dataset.targets == i])

result = {"image":[],
        "actual":[],
        "group":[],
        "spurious":[]                
        }

with torch.no_grad():
    running_corrects = 0
    for (images, (targets, targets_g, targets_s), index, paths) in tqdm(test_dataloader):
        _, preds = torch.max(outputs, 1)
        for i in range(len(paths)):
            image = paths[i]
            actual = targets[i]
            group = targets_g[i]
            spurious = targets_s[i]
            result['image'].append(image)
            result['actual'].append(actual.item())
            result['group'].append(group.item())
            result['spurious'].append(spurious.item())

df = pd.DataFrame(result)
df['svm_score'] = test_decision_values

result_path = result_dir + "failure_direction" +"_"+ args.dataset + ".csv"
df.to_csv(result_path)
print("Extracted failure direction SVM scores to ",result_path)
