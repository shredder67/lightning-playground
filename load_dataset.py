"""Load dataset and dataloader for plyaing around"""
import numpy as np

import torch
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split

def get_train_val_test_dataloaders():
    train_dataset = MNIST(root='./data', download=True, train=True, transform=T.ToTensor())
    test_dataset = MNIST(root='./data', download=True, train=False, transform=T.ToTensor())
    shuffled_indices = np.random.permutation(np.arange(len(train_dataset)))
    
    train_ds = Subset(train_dataset, shuffled_indices[:int(0.8*len(train_dataset))])
    val_ds = Subset(train_dataset, shuffled_indices[int(0.8*len(train_dataset)):])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    
    return train_loader, val_loader, test_loader

