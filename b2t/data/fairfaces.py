"""
Data class based on dataset by Karkkainen, Kimmo and Joo, Jungseock in the paper FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation
"""
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Fairfaces(Dataset):
    """
    Waterbirds dataset from waterbird_complete95_forest2water2 in GroupDRO paper
    """

    def __init__(self, data_dir='data/fairfaces_data', split='train', transform=None, zs_group_label=None,confounder_dict = {"White": 0, "Black": 1},minority_group=3,minority_ratio=0.05):
        self.data_dir = data_dir
        self.split = split
        self.minority_group = minority_group # 3-> black,female
        self.minority_ratio = minority_ratio # 0.05

        if split == 'train' or split == 'test':
            self.metadata_df = pd.read_csv(os.path.join(self.data_dir,"train", 'train_labels.csv'))

            self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split]

        elif split == 'val':
            self.metadata_df = pd.read_csv(os.path.join(self.data_dir,"val", 'val_labels.csv'))


        target_dict = {"Male": 0, "Female": 1}
        self.metadata_df['gender'] = self.metadata_df['gender'].map(target_dict)
        self.metadata_df['race'] = self.metadata_df['race'].map(confounder_dict)
        self.metadata_df = self.metadata_df.dropna()

        # make biased dataset (low datapoints that are black and female)
        

        # Get the y values, being 
        self.y_array = self.metadata_df['gender'].values
        self.confounder_array = self.metadata_df['race'].values
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')

        if (self.minority_ratio != -1) and (split == "train" or split == "val" ) : # no downsampling with -1
            random_sampling_state = 42 # for reproducibility

            self._downsample_group(self.minority_group, self.minority_ratio, random_sampling_state) # downsample metadata_df, then reinsitate the arrays
            self.y_array = self.metadata_df['gender'].values
            self.confounder_array = self.metadata_df['race'].values
            self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['file'].values

        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)

        self.transform = transform

        self.n_classes = 2
        self.n_groups = 4

    def __len__(self):
        return len(self.filename_array)
    
    def _downsample_group(self, target_group, target_ratio, random_state):
        """
        Downsample rows in metadata_df where group_array == target_group.
        """

        target_group_df = self.metadata_df[self.group_array == target_group]
        other_groups_df = self.metadata_df[self.group_array != target_group]
 
        if len(other_groups_df) > len(target_group_df): 
            #nr samples to keep
            num_target_samples = int(len(other_groups_df) * target_ratio / (1 - target_ratio))

            # Randomly sample the target group
            sampled_target_group_df = target_group_df.sample(n=num_target_samples, random_state=random_state)
            self.metadata_df = pd.concat([sampled_target_group_df, other_groups_df], ignore_index=True)
        else:
            #nr samples to keep
            num_target_samples = int(len(other_groups_df) * target_ratio / (1 - target_ratio))

            # Randomly sample the target group
            sampled_other_groups_df = other_groups_df.sample(n=num_target_samples, random_state=random_state)
            self.metadata_df = pd.concat([sampled_other_groups_df, target_group_df], ignore_index=True)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        x = self.transform(img)

        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]

        path = self.filename_array[idx]
        return x, (y, y_group, y_spurious), idx, path


def load_celeba(root_dir, bs_train=128, bs_val=128, num_workers=8):
    """
    Default dataloader setup for Waterbirds

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = Fairfaces(root_dir, split='train')
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    val_set = Fairfaces(root_dir, split='val')
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = Fairfaces(root_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
