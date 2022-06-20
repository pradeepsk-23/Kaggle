import torch
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from skimage import io
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class HAR(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = str(self.annotations.iloc[index, 1])

        if self.transform: 
            image = self.transform(image)

        return (image, y_label)

# Load Data
train_dataset = HAR(csv_file="./Dataset/Human Action Recognition/Training_set.csv",
                    root_dir="./Dataset/Human Action Recognition/train",
                    transform=transforms.ToTensor())

test_dataset = HAR(csv_file="./Dataset/Human Action Recognition/Testing_set.csv",
                   root_dir="./Dataset/Human Action Recognition/test",
                   transform=transforms.ToTensor())
    
# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)
