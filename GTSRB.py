import torch
import os
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as tt

from PIL import Image
from torchvision.datasets import GTSRB
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class GTSRB(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class2index = { 0:'Speed limit (20km/h)',
                            1:'Speed limit (30km/h)', 
                            2:'Speed limit (50km/h)', 
                            3:'Speed limit (60km/h)', 
                            4:'Speed limit (70km/h)', 
                            5:'Speed limit (80km/h)', 
                            6:'End of speed limit (80km/h)', 
                            7:'Speed limit (100km/h)', 
                            8:'Speed limit (120km/h)', 
                            9:'No passing', 
                            10:'No passing veh over 3.5 tons', 
                            11:'Right-of-way at intersection', 
                            12:'Priority road', 
                            13:'Yield', 
                            14:'Stop', 
                            15:'No vehicles',
                            16:'Veh > 3.5 tons prohibited', 
                            17:'No entry', 
                            18:'General caution', 
                            19:'Dangerous curve left', 
                            20:'Dangerous curve right', 
                            21:'Double curve', 
                            22:'Bumpy road', 
                            23:'Slippery road', 
                            24:'Road narrows on the right', 
                            25:'Road work', 
                            26:'Traffic signals', 
                            27:'Pedestrians', 
                            28:'Children crossing', 
                            29:'Bicycles crossing', 
                            30:'Beware of ice/snow',
                            31:'Wild animals crossing', 
                            32:'End speed + passing limits', 
                            33:'Turn right ahead', 
                            34:'Turn left ahead', 
                            35:'Ahead only', 
                            36:'Go straight or right', 
                            37:'Go straight or left', 
                            38:'Keep right', 
                            39:'Keep left', 
                            40:'Roundabout mandatory', 
                            41:'End of no passing', 
                            42:'End no passing veh > 3.5 tons' } 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.df.at[index, "image"]))
        label = self.class2index[self.df.at[index, "label"]]

        if self.transform:
            image = self.transform(image)

    
    
        return image, label

# Transformations
train_transform = tt.Compose([tt.Resize(28),
                        tt.RandomCrop(size=28, padding=4, padding_mode="reflect"),
                        tt.ToTensor()])
test_transform = tt.Compose([tt.Resize(28),
                        tt.ToTensor()])

# CIFAR10 dataset (images and labels)
train_dataset = GTSRB(root='./Dataset/GTSRB', train=True, transform=train_transform, download=True)

test_dataset = GTSRB(root='./Dataset/GTSRB', train=False, transform=test_transform)

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)

# Convolution block
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# Convolutional neural network
num_classes = 43
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = conv_block(3, 16) # out = [16, 28, 28]
        self.conv2 = conv_block(16, 32, pool=True) # out = [32, 14, 14]
        self.conv3 = conv_block(32, 64) # out = [64, 14, 14]
        self.conv4 = conv_block(64, 128, pool=True) # out = [128, 7, 7]

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(128*7*7, num_classes))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x     

# Model
model = ConvNet(num_classes).to(device)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 5
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i+1) % 500 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))