import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import clip

from sklearn.metrics import classification_report
from tqdm import tqdm

from data.celeba import CelebA
from data.waterbirds import Waterbirds

import celeba_templates
import waterbirds_templates

def main(args):
    model, preprocess = clip.load('RN50', 'cuda', jit=False)

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])


    if args.dataset == 'waterbirds':
        data_dir = os.path.join(args.data_dir, 'cub/data/waterbird_complete95_forest2water2')
        test_dataset = Waterbirds(data_dir=data_dir, split='test', transform=transform)
        templates = waterbirds_templates.templates
        if args.score == 'positive':
            class_templates = waterbirds_templates.pos_prompt_templates
        elif args.score == 'negative':
            class_templates = waterbirds_templates.neg_prompt_templates
        class_keywords_all = waterbirds_templates.pos_prompt_classes
    elif args.dataset == 'celeba':
        data_dir = os.path.join(args.data_dir, 'celebA/data')
        test_dataset = CelebA(data_dir=data_dir, split='test', transform=transform)
        templates = celeba_templates.templates
        if args.score == 'positive':
            class_templates = celeba_templates.pos_prompt_templates
        elif args.score == 'negative':
            class_templates = celeba_templates.neg_prompt_templates
        class_keywords_all = celeba_templates.pos_prompt_classes
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter

    # calculate zero-shot weights for clip classifier
    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_keywords_all:
            if args.dataset == 'waterbirds':
                texts = [template.format(class_template.format(class_keywords)) for template in templates for class_template in class_templates]
            elif args.dataset == 'celeba':
                texts = [template.format(class_template.format(class_keyword)) for template in templates for class_template in class_templates for class_keyword in class_keywords]
            print(f"example text: {texts[1]}")
            texts = clip.tokenize(texts).cuda()

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    preds_minor, preds, targets_minor = [], [], []
    probs_list, targets_list,target_spur_list = [], [], [] # edited
    with torch.no_grad():
        for (image, (target, target_g, target_s), _) in tqdm(train_dataloader):
            image = image.cuda()
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights / temperature

            probs = logits.softmax(dim=-1).cpu()
            conf, pred = torch.max(probs, dim=1)

            if args.dataset == 'waterbirds':
                # minor group if
                # (target, target_s) == (0, 1): landbird on water background
                # (target, target_s) == (1, 0): waterbird on land background
                is_minor_pred = (((target == 0) & (pred == 1)) | ((target == 1) & (pred == 0))).long()
                is_minor = (((target == 0) & (target_s == 1)) | ((target == 1) & (target_s == 0))).long()
            if args.dataset == 'celeba':
                # minor group if
                # (target, target_s) == (1, 1): blond man
                is_minor_pred = ((target == 1) & (pred == 1)).long()
                is_minor = ((target == 1) & (target_s == 1)).long()

            preds_minor.append(is_minor_pred)
            preds.append(pred)
            targets_minor.append(is_minor)
            probs_list.append(probs) # edited
            targets_list.append(target) # edited
            target_spur_list.append(target_s)  #edited


    preds_minor, preds, targets_minor = torch.cat(preds_minor), torch.cat(preds), torch.cat(targets_minor)
    
    probs_list = torch.cat(probs_list,dim=0) # waterbird: col 1 is prob for water background, col 2 for land. CelebA: col 1 voor "women" (just nothing in keywords), col 2 voor man
    target_spur_list = torch.cat(target_spur_list) # Waterwird: 0 == water background , 1 == land background CelebA: 1 voor man , 0 voor northing
    targets_list = torch.cat(targets_list) # Waterbird: 0 ==  waterbird class , 1 == landbird class; Celeba: 1 == blond class 

    # sort groups and correct predictions
    group_list = targets_list *2 + target_spur_list 
    num_groups = 4
    corrects = preds.eq(targets_list)
    group_indices = dict()
    for i in range(num_groups):
        group_indices[i] = np.where(group_list == i)[0]

    print("total", len(preds))
    print("total corrects:", corrects.sum())

    # calculate worst-group and average accuracies
    worst_accuracy = 100
    for i in range(num_groups):
        print("group: ", i)
        print("total in group", len(group_indices[i]))
        correct = corrects[group_indices[i]].sum().item()
        print("correct in group", correct)
        accuracy = 100. * correct / len(group_indices[i])
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy

        accuracy = 100. * corrects.sum().item() / len(preds)

    print("worst group accuracy:", worst_accuracy)
    print("average accuracy:", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--score', default='positive')
    parser.add_argument('--dataset', default='waterbirds')
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--save_path', default='./prompt_engineering/waterbirds.pt')

    args = parser.parse_args()
    main(args)