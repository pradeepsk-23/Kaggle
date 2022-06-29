import os
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as tt
import torch.nn.functional as F

from model import ConvNet
from utils import get_loaders

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train Function
def train_fn(dataloader, model, opt, loss_fn):
    epochs = 5
    total_step = len(dataloader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
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

            if (i+1) % 126 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

def main():

    # Model
    num_classes=15
    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    # F.cross_entropy computes softmax internally
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dl, test_dl = get_loaders(train_csv_file = "./Dataset/Human Action Recognition/training_set.csv",
                                 train_root_dir = "./Dataset/Human Action Recognition/train",
                                 test_csv_file = "./Dataset/Human Action Recognition/testing_set.csv",
                                 test_root_dir= "./Dataset/Human Action Recognition/train")
    
    train_fn(train_dl, model, opt, loss_fn)

if __name__ == "__main__":
    main()
