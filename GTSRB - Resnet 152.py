import torch
import os
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torchvision.datasets import GTSRB
from torch.utils.data import Dataset, DataLoader, random_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
root_dir = "D:\IRP\GitHub\Dataset\GTSRB\Train"

dataset = ImageFolder(root_dir, tt.Compose([tt.Resize(28),
                                            tt.RandomCrop(28), 
                                            tt.ToTensor()]))

val_pct = 0.2
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# DataLoader (input pipeline)
batch_size = 24
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

# Residual Block
class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layer
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                                                nn.BatchNorm2d(intermediate_channels * 4))
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def main():

    # Model
    # model = ResNet(block, [3, 4, 6, 3], 3, 43).to(device) # ResNet50
    # model = ResNet(block, [3, 4, 23, 3], 3, 43).to(device) # Resnet101
    model = ResNet(block, [3, 8, 36, 3], 3, 43).to(device) # Resnet152

    # Loss and optimizer
    # F.cross_entropy computes softmax internally
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Set up one-cycle learning rate scheduler
    epochs = 5
    grad_clip = 0.1

    # For updating learning rate
    def update_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=epochs, steps_per_epoch=len(train_dl))

    # Train the model
    total_step = len(train_dl)
    for epoch in range(epochs):
        lrs = []
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward and optimize
            opt.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            opt.step()

            # Record & update learning rate
            lrs.append(update_lr(opt))
            sched.step()
    
        if (i+1) % 1307 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()          # Turns off dropout and batchnorm layers for testing / validation.
    with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

if __name__ == "__main__":
    main()