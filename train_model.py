import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import sys

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

target_to_class = {k:v for v,k in ImageFolder("train/").class_to_idx.items()}

class SimpleCardClassifier(nn.Module):

    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280

        self.classifier = nn.Linear(enet_out_size, num_classes)


    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

# Setup Datasets

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_folder = "train/"
valid_folder = "valid/"
test_folder =  "test/"

train_dataset = PlayingCardDataset(train_folder, transform)
valid_dataset = PlayingCardDataset(valid_folder, transform)
test_dataset = PlayingCardDataset(test_folder, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training Loop

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(device)

model = SimpleCardClassifier()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

number_epochs = 8
train_losses, val_losses = [], []

for epoch in range(number_epochs):
    count = 1
    # Training
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"\rTrain: {round((32*100*count)/len(train_loader.dataset), 2)}%", end="", flush=True)
        count += 1
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    count = 1
    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images,labels in valid_loader:
            images,labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            sys.stdout.write('\x1b[1A') # move up one line
            print(f"\rEval: {round((32*100*count)/len(valid_loader.dataset), 2)}%", end="", flush=True)
            count += 1
            running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)

    # Print Epoch Stats
    print(f"Epoch {epoch+1}/{number_epochs} - Train loss: {train_losses} - Valid loss: {val_losses}")
