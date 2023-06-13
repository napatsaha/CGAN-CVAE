# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:48:09 2023

@author: napat

Training a Classifier for MNIST Images
"""

import torch, os
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Classifier, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels=32, 
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
            )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, out_channels=64, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1)
            )
        
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
    def get_last_layer(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, start_dim=1)
        return x
    

if __name__ == "__main__":
    name = 'mnist_clf'
    trial = 1
    name = name + "_" + str(trial).zfill(2)
    
    os.chdir("C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/Project")
    for direc in ["model",'image']:
        if not os.path.exists(direc):
            os.mkdir(direc)    
    
    batch_size = 128
    lr = 1e-3
    num_epochs = 10
    report_freq = 100
    # plot_freq = 5
    save_freq = 25
    
    image_width = 28
    image_dim = image_width**2
    # latent_dim = 64
    # hidden_size = 256
    # alpha = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = MNIST('C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=False)
    dataset_test = MNIST('C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        train=False,      
        download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    n_classes = len(dataset.classes)
    n_channels = dataset[0][0].shape[0]
    
    classifier = Classifier(n_channels, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(classifier.parameters(), lr=lr)
    
    # Reload model if exists
    if isinstance(num_epochs, tuple) and len(num_epochs) > 1:
        start_epoch = num_epochs[0] - 1 # Previous iteration
        end_epoch = num_epochs[1]
        if os.path.exists(f"model/{name}_epoch{str(start_epoch).zfill(2)}.pt"):
            classifier.load_state_dict(torch.load(f"model/{name}_epoch{str(start_epoch).zfill(2)}.pt"))
    elif isinstance(num_epochs, int):
        start_epoch = 0
        end_epoch = num_epochs
        num_epochs = start_epoch, end_epoch
    
    # Training Loop
    for epoch in range(*num_epochs):
        for i, (real, labels) in enumerate(loader):
            real = real.to(device)
            labels = labels.to(device)
            
            output = classifier(real)
            loss = criterion(output, labels)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            # Display
            if i % report_freq == 0:
                print(f"Epoch [{epoch+1}/{end_epoch}] Batch [{str(i).zfill(3)}/{len(loader)}] \
                      Loss: {loss.item():.4f}")
        
        if epoch % save_freq == 0:
            torch.save(classifier.state_dict(), f"model/{name}_epoch{str(epoch).zfill(2)}.pt")
            
    torch.save(classifier.state_dict(), f"model/{name}_epoch{str(epoch).zfill(2)}.pt")
