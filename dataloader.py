"""Fetech dataloader"""
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import utils
import config

transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

def fetch_dataloaders(data_dir, params):
    datasets = {split: ImageFolder(os.path.join(data_dir, split), transform) 
                for split in ['train', 'val', 'test']}

    dataloaders = {split: DataLoader(datasets[split], batch_size=params.batch_size, shuffle=True, num_workers=4) 
                for split in ['train', 'val', 'test']}
    return dataloaders