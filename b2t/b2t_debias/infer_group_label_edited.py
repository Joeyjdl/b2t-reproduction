import argparse
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):
    model, preprocess = clip.load('RN50', device, jit=False)  # RN50, RN101, RN50x4, ViT-B/32

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])

    if args.dataset == 'waterbirds':
        data_dir = os.path.join(args.data_dir, "cub/data/waterbird_complete95_forest2water2")
        train_dataset = Waterbirds(data_dir=data_dir, split='test', transform=transform) #edited: seems like fig 5 is done using test
        templates = waterbirds_templates.templates
        class_templates = waterbirds_templates.class_templates
        class_keywords_all = waterbirds_templates.class_keywords_all
    elif args.dataset == 'celeba':
        data_dir = os.path.join(args.data_dir, 'celebA/data')
        train_dataset = CelebA(data_dir=data_dir, split='test', transform=transform) # edited: seems like fig 5 is done using test
        templates = celeba_templates.templates
        class_templates = celeba_templates.class_templates
        class_keywords_all = celeba_templates.class_keywords_all
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter

    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_keywords_all:
            texts = [template.format(class_template.format(class_keyword)) for template in templates for class_template in class_templates for class_keyword in class_keywords]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    preds_minor, preds, targets_minor = [], [], []
    probs_list, targets_list,target_spur_list = [], [], [] # edited
    with torch.no_grad():
        for (image, (target, target_g, target_s), _) in tqdm(train_dataloader):
            image = image.to(device)
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
    
    # edited
    probs_list = torch.cat(probs_list,dim=0) # Waterbird: (col 1 == probability for land background), (col 2 == water background). CelebA: (col 1 == ""),  (col 2 == man)
    target_spur_list = torch.cat(target_spur_list) # Waterwird: (label 0 == land background) , (1 == water background) CelebA: (0 == ""), (1 ==  man)
    targets_list = torch.cat(targets_list) # Waterbird: (0 == landbird class) , (1 == water class) ; Celeba: (0 == not blond), (1 == blond class)
    
    import pandas as pd
    storage_df = pd.DataFrame(probs_list.cpu().numpy(),columns=["col_1","col_2"]) 
    storage_df["Target"] = targets_list.cpu().numpy()
    storage_df["Target_Spurious"] = target_spur_list.cpu().numpy()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    storage_df.to_csv(args.save_path)
    print("Succesfully stored probs in ",args.save_path) # take note that infer file uses train set not val set
    #########



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='waterbirds', choices=['celeba', 'waterbirds'])
    parser.add_argument('--data_dir', default='../data/cub/data')
    parser.add_argument('--save_path', default='./pseudo_bias/waterbirds.pt')

    args = parser.parse_args()
    main(args)
