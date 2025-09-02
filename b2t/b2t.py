# from concurrent.futures import ThreadPoolExecutor
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
from data.imagenet import ImageNetDataset, get_transform_imagenet

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_caption import extract_caption_batch ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity
from data.imagenet import IMAGENET2012_CLASSES

from tqdm import tqdm
import os
import time
import pandas as pd
import math
from tabulate import tabulate


import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import cProfile
import pstats
import io

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    # Need class for datasets with multiple classes
    parser.add_argument("--class_names",type=str,nargs="+", 
                        help="Names of the classes for running (provide multiple class names separated by space)")
    parser.add_argument("--class_name", type = str, default = 'bee', help="Name of the class for running")
    parser.add_argument("--model", type=str, default='best_model_Waterbirds_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_Waterbirds_erm.pth, best_model_Waterbirds_dro.pth
    parser.add_argument("--extract_caption", action='store_true',help="extract captions from images")
    parser.add_argument("--save_result", action='store_true')
    parser.add_argument("--minority_ratio", type = float,default=0.05,help="fairfaces minority ratio in data.")
    parser.add_argument("--minority_group", type = int,default=3,help="fairfaces minority group that will be 'mispredicted' ")
    parser.add_argument("--rerun_result", action='store_true',help="rerun the code for getting the results")
    parser.add_argument("--override_result", action='store_true',help="override stored result, without necessarily having to extract caption(still can though)")
    parser.add_argument("--no_model", action='store_true',help="uses no classification model and just uses the target as the prediction")
    args = parser.parse_args()
    return args

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'waterbird':
        preprocess = get_transform_cub()
        class_names = [('landbird', 'waterbird')]
        # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
        image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
        caption_dir = 'data/cub/caption/'
        val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)
        class_labels = [('landbird', 'waterbird')]
        classes = [1]
    elif args.dataset == 'celeba':
        preprocess = get_transform_celeba()
        class_names = [('not blond', 'blond')]
        # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
        image_dir = 'data/celebA/data/img_align_celeba/'
        caption_dir = 'data/celebA/caption/'
        val_dataset = CelebA(data_dir='data/celebA/data/', split='val', transform=preprocess)
        class_labels = [('not blond', 'blond')]
        classes = [1]
    elif args.dataset == 'fairfaces':
        preprocess = get_transform_cub() # cub transform is general
        print("Recommend running fairfaces initially with -1 as minority_ratio and extract_caption , allowing you to only have to extract_caption once! (all data will be captioned). Runs after this may use --override_result")
        class_names = ['male', 'female']
        class_labels = [('male', 'female')]
        image_dir = 'data/fairfaces_data/' 
        caption_dir = 'data/fairfaces_data/caption/'
        val_dataset = Fairfaces(data_dir=image_dir, split='val', transform=preprocess,minority_ratio=args.minority_ratio,minority_group=args.minority_group)
        classes = [1]
    elif args.dataset == 'imagenet':
        preprocess = get_transform_imagenet()
        class_names = [f'not {args.class_name}', f'{args.class_name}']
        # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
        image_dir = 'data/imagenet/data/imagenet-val/'
        caption_dir = 'data/imagenet/caption/'
        val_dataset = ImageNetDataset(data_dir='data/imagenet/data/imagenet-val', split='val' ,transform=preprocess)
        if args.class_names is None:
            class_labels = [
                (f"not {full_name}", full_name)
                for class_tag, full_name in IMAGENET2012_CLASSES.items()
            ]
            classes = range(0,1000)
        else:
            class_labels = []
            classes = []

            for class_name in args.class_names:
                class_info = ImageNetDataset.get_class_info(class_name)
                if class_info is not None:
                    class_tag, index, full_name = class_info  # Unpack values
                    class_labels.append((f"not {full_name}", full_name))  # Keep class_labels the same
                    classes.append(index)  # Store only indices

                else:
                    print(f"Invalid dataset {args.dataset}")
                    return

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers= os.cpu_count() // 2, drop_last=False)




    result_dir = 'result/'
    model_dir = 'model/'
    diff_dir = 'diff/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)

    if args.extract_caption:
        print("Start extracting captions...")
        
        # Collect all image paths
        if args.dataset == "imagenet":
            image_paths = [
                os.path.join(image_dir, filename.split('_')[-1].split('.')[0], '_'.join(filename.split('_')[:-1]) + ".JPEG")
                for filename in val_dataset.filename_array
            ]
        else:
            image_paths = [os.path.join(image_dir, path) for path in val_dataset.filename_array]

        # Process images in batches
        captions = extract_caption_batch(
            image_paths=image_paths,
            batch_size=512  # Adjust the batch size as per available memory
        )

        # Save captions to files
        for image_path, caption in zip(image_paths, captions):
            if not os.path.exists(caption_dir):
                os.makedirs(caption_dir)
            caption_path = caption_dir + os.path.basename(image_path)[:-4] + ".txt"
            with open(caption_path, 'w') as f:
                f.write(caption)
        
        print(f"Captions for {len(image_paths)} images have been extracted.")


    # correctify dataset
    if args.no_model:
        result_path = result_dir + args.dataset + "_no_model.csv"
    else:
        result_path = result_dir + args.dataset + "_" +  args.model.split(".")[0] + ".csv"
    if not os.path.exists(result_path) or args.rerun_result == True:
        if args.no_model == False:
            if args.model.endswith(".pth"):
                model = torch.load(model_dir + args.model)
            elif args.model == "Resnet50":
                # model = models.resnet50(pretrained=True)
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                model = models.resnet50(weights=weights)   
            else:
                print(f"invalid model {args.model}")
                pass

            model = model.to(device)
            model.eval()
            print("Pretrained model \"{}\" loaded".format(args.model))
        start_time = time.time()

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
                images = images.to(device)
                targets = targets.to(device)
                if args.no_model == True:
                    preds = targets
                else:
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                for i in range(len(preds)):
                    image = paths[i]
                    pred = preds[i]
                    actual = targets[i]
                    group = targets_g[i]
                    spurious = targets_s[i]
                    if args.dataset == "imagenet":
                        caption_path = os.path.join(caption_dir, '_'.join(image.split('_')[:3]) + "..txt")
                    else:
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
    

    
    if args.dataset == "imagenet":
        for ind in range(0,len(classes)):
            class_1 = classes[ind]
            class_tag, index, full_class_name = ImageNetDataset.get_class_info(class_1)
            image_path = os.path.join(image_dir, class_tag) + "/"
            print(f"Running scoring for {class_labels[ind][1]}")
            print(f"progress at {ind}/{len(classes)}")

            if args.no_model:
                diff_path_1 = diff_dir + args.dataset + "_no_model_" +  class_labels[ind][1] + ".csv"
            else:
                diff_path_1 = diff_dir + args.dataset + "_" +  args.model.split(".")[0] + "_" +  class_labels[ind][1] + ".csv"
            
            if not os.path.exists(diff_path_1):
                df_wrong = df[df['correct'] == 0]
                df_correct = df[df['correct'] == 1]
                keywords_class_0 = list()
                dist_class_0 = list()

            if not os.path.exists(diff_path_1):
                df_class_1 = df[df['actual'] == class_1] # blond, waterbird
                df_wrong_class_1 = df_wrong[df_wrong['actual'] == class_1]
                df_correct_class_1 = df_correct[df_correct['actual'] == class_1]
                
                if args.no_model:
                    caption_correct_class_1 = ' '.join(df_correct_class_1['caption'].tolist())
                    keywords_class_1 = extract_keyword(caption_correct_class_1)
                else:
                    caption_wrong_class_1 = ' '.join(df_wrong_class_1['caption'].tolist())
                    keywords_class_1 = extract_keyword(caption_wrong_class_1)

                print("Start calculating scores..")

                if df_wrong_class_1['image'].count() != 0:
                    df_wrong_class_1['updated_image'] = df_wrong_class_1['image'].apply(
                        lambda x: '/'.join(x.split('/')[:-1]) + '/' + '_'.join(x.split('/')[-1].split('_')[:3]) + '.JPEG'
                    )
                    similarity_wrong_class_1 = calc_similarity(image_path, df_wrong_class_1['updated_image'], keywords_class_1)
                else:
                    similarity_wrong_class_1 = 0
                
                if df_correct_class_1['image'].count() != 0:
                    df_correct_class_1['updated_image'] = df_correct_class_1['image'].apply(
                        lambda x: '/'.join(x.split('/')[:-1]) + '/' + '_'.join(x.split('/')[-1].split('_')[:3]) + '.JPEG'
                    )
                    similarity_correct_class_1 = calc_similarity(image_path, df_correct_class_1['updated_image'], keywords_class_1)
                else:
                    similarity_correct_class_1 = 0

                dist_class_1 = similarity_wrong_class_1 - similarity_correct_class_1
                
                print("Result for class :", class_labels[ind][1])
                diff_1 = print_similarity(keywords_class_1, keywords_class_0, dist_class_1, dist_class_0, df_class_1)

                if args.save_result:
                    diff_1.to_csv(diff_path_1)
            else:
                diff_1 = pd.read_csv(diff_path_1)
                print(tabulate(diff_1, headers='keys', showindex=False))

    else:
        class_1 = classes[0]
        # extract keyword
        df_wrong = df[df['correct'] == 0]
        df_correct = df[df['correct'] == 1]
        df_class_0 = df[df['actual'] != class_1] # not blond, landbird
        df_class_1 = df[df['actual'] == class_1] # blond, waterbird
        df_wrong_class_0 = df_wrong[df_wrong['actual'] != class_1]
        df_wrong_class_1 = df_wrong[df_wrong['actual'] == class_1]
        df_correct_class_0 = df_correct[df_correct['actual'] != class_1]
        df_correct_class_1 = df_correct[df_correct['actual'] == class_1]

        if args.no_model:
            caption_correct_class_0 = ' '.join(df_correct_class_0['caption'].tolist())
            caption_correct_class_1 = ' '.join(df_correct_class_1['caption'].tolist())
            keywords_class_0 = extract_keyword(caption_correct_class_0)
            keywords_class_1 = extract_keyword(caption_correct_class_1)
        else:
            caption_wrong_class_0 = ' '.join(df_wrong_class_0['caption'].tolist())
            caption_wrong_class_1 = ' '.join(df_wrong_class_1['caption'].tolist())
            keywords_class_0 = extract_keyword(caption_wrong_class_0)
            keywords_class_1 = extract_keyword(caption_wrong_class_1)

        # calculate similarity
        print("Start calculating scores..")
        if df_wrong_class_0['image'].count() != 0:
            similarity_wrong_class_0 = calc_similarity(image_dir, df_wrong_class_0['image'], keywords_class_0)
        else:
            similarity_wrong_class_0 = 0
        if df_correct_class_0['image'].count() != 0:
            similarity_correct_class_0 = calc_similarity(image_dir, df_correct_class_0['image'], keywords_class_0)
        else:
            similarity_correct_class_0 = 0
        if df_wrong_class_1['image'].count() != 0:
            similarity_wrong_class_1 = calc_similarity(image_dir, df_wrong_class_1['image'], keywords_class_1)
        else:
            similarity_wrong_class_1 = 0
        if df_correct_class_1['image'].count() != 0:
            similarity_correct_class_1 = calc_similarity(image_dir, df_correct_class_1['image'], keywords_class_1)
        else:
            similarity_correct_class_1 = 0

        if args.no_model:
            dist_class_0 = similarity_correct_class_0 - similarity_correct_class_0.mean().item() 
            dist_class_1 = similarity_correct_class_1 - similarity_correct_class_1.mean().item() 
        else: 
            dist_class_0 = similarity_wrong_class_0 - similarity_correct_class_0
            dist_class_1 = similarity_wrong_class_1 - similarity_correct_class_1

        if (args.dataset != "fairfaces")  or (args.dataset == "fairfaces" and minority_class == 0): # only print the minority group stats
            print("Result for class :", class_labels[0][0])
            diff_0 = print_similarity(keywords_class_0, keywords_class_1, dist_class_0, dist_class_1, df_class_0)
            print("*"*60)

            if args.save_result:
                diff_path_0 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0][0] + ".csv"
                diff_0.to_csv(diff_path_0)

        if (args.dataset != "fairfaces")  or (args.dataset == "fairfaces" and minority_class == 1):
            print("Result for class :", class_labels[0][1])
            diff_1 = print_similarity(keywords_class_1, keywords_class_0, dist_class_1, dist_class_0, df_class_1)
            
            if args.save_result:
                diff_path_1 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0][1] + ".csv"
                diff_1.to_csv(diff_path_1)


if __name__ == "__main__":
    # Initialize the profiler
    profiler = cProfile.Profile()
    args = parse_args()

    profiler.enable()  # Start profiling

    main(args)  # Run the main script

    profiler.disable()  # Stop profiling
    if args.dataset != "imagenet":
        profile_output_file = f"profiling_results_{args.dataset}.prof"
    else:
        profile_output_file = f"profiling_results_{args.dataset}_{args.class_name}.prof"
    profiler.dump_stats(profile_output_file)

    # Save the profiling results to a human-readable format
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats()

