import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

# Dataset Class
class HAR(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class2index = {"calling":0, "clapping":1, "cycling":2, "dancing":3, "drinking":4,
        "eating":5, "fighting":6, "hugging":7, "laughing":8, "listening_to_music":9,
        "running":10, "sitting":11, "sleeping":12, "texting":13, "using_laptop":14}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.df[index, "image"]))
        label = self.class2index[self.df[index, "label"]]

        if self.transform:
            image = self.transform(image)
    
        return image, label