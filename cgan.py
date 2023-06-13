# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:26:16 2023

@author: napat

Appending Label to Laten Dimension
"""

import torch, os
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, image_dim, n_classes,
                 hidden_size=256):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.image_dim = image_dim
        self.input_dim = latent_dim + n_classes
        self.output_dim = image_dim
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Tanh()
        )
        
    def forward(self, x, labels):
        labels = F.one_hot(labels, num_classes=self.n_classes)
        x = torch.cat((x, labels), dim=1)
        return self.network(x)
    
class Discriminator(nn.Module):
    def __init__(self, image_dim, n_classes,
                 hidden_size=256):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.image_dim = image_dim
        self.n_classes = n_classes
        self.input_dim = image_dim + n_classes
        self.output_dim = 1
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        labels = F.one_hot(labels, num_classes=self.n_classes)
        x = torch.cat((x, labels), dim=1)
        return self.network(x)

def visualise(epoch, name, dataset, generator, width=5, save=True, figsize=(20,20)):
    with torch.no_grad():
        
        loader2 = DataLoader(dataset, batch_size=width**2, shuffle=True)            
        img, labels = next(iter(loader2))
        # labels = F.one_hot(labels, num_classes=n_classes).to(device)
        noise = torch.randn(width**2, latent_dim).to(device)
        # inp = torch.cat([noise, labels], dim=1)
        fake = generator(noise, labels.to(device)).view(-1, 1, image_width, image_width)
        
        # real2 = real.view(-1, 1, image_width, image_width)
        img_grid_real = make_grid(img, normalize=True, nrow=width)
        img_grid_fake = make_grid(fake, normalize=True, nrow=width)
        
        plt.figure(figsize=figsize)
        
        plt.subplot(1,2,1)
        plt.imshow(img_grid_real.permute(1,2,0).cpu())
        plt.title(f"Original Image\nEpoch {epoch}")
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(img_grid_fake.permute(1,2,0).cpu())
        plt.title(f"Generated Image\nEpoch {epoch}")
        plt.axis("off")
        
        if save and isinstance(epoch, int):
            fig_name = f"./image/{name}_epoch{str(epoch).zfill(2)}.png"
        elif save and isinstance(epoch, str):
            fig_name = f"./image/{name}_{epoch.lower()}.png"
            
        plt.savefig(fig_name)
        plt.show()

if __name__ == "__main__":
    name = 'mnist_cgan'
    trial = 1
    name = name + "_" + str(trial).zfill(2)
    
    os.chdir("C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/Project")
    for direc in ["model",'image']:
        if not os.path.exists(direc):
            os.mkdir(direc)    
    
    batch_size = 64
    lr = 1e-3
    num_epochs = 100
    report_freq = 100
    plot_freq = 5
    save_freq = 5
    
    image_width = 28
    image_dim = image_width**2
    latent_dim = 64
    hidden_size = 256
    alpha = 0.1
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = MNIST('C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_classes = len(dataset.classes)
    
    generator = Generator(latent_dim, image_dim, n_classes, hidden_size).to(device)
    discriminator = Discriminator(image_dim, n_classes, hidden_size).to(device)
    
    if isinstance(num_epochs, tuple) and len(num_epochs) > 1:
        start_epoch = num_epochs[0] - 1 # Previous iteration
        end_epoch = num_epochs[1]
        if os.path.exists(f"model/{name}_generator_epoch{str(start_epoch).zfill(2)}.pt"):
            generator.load_state_dict(torch.load(f"model/{name}_generator_epoch{str(start_epoch).zfill(2)}.pt"))
            discriminator.load_state_dict(torch.load(f"model/{name}_discriminator_epoch{str(start_epoch).zfill(2)}.pt"))
    elif isinstance(num_epochs, int):
        start_epoch = 0
        end_epoch = num_epochs
        num_epochs = start_epoch, end_epoch
    
    optim_gen = optim.Adam(generator.parameters(), lr=lr)
    optim_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    criterion = nn.BCELoss()
    
    visualise("Initialisation", name, dataset, generator)
    
    # Training Loop
    for epoch in range(*num_epochs):
        for i, (real, labels) in enumerate(loader):
            real = real.view(-1, image_dim).to(device)
            labels = labels.to(device)
            
            # Generate fake image
            noise = torch.randn(real.shape[0], latent_dim).to(device)
            fake = generator(noise, labels)
            
            # Discriminate images
            disc_real = discriminator(real, labels)
            disc_fake = discriminator(fake.detach(), labels)
            
            # Updata Discriminator
            loss_disc = (
                    criterion(disc_real, torch.full_like(disc_real, 1 - alpha)) +
                    criterion(disc_fake, torch.zeros_like(disc_fake))
                ) / 2
            discriminator.zero_grad()
            loss_disc.backward()
            optim_disc.step()
            
            # Generator Update
            gen = discriminator(fake, labels)
            loss_gen = criterion(gen, torch.ones_like(gen))
            generator.zero_grad()
            loss_gen.backward()
            optim_gen.step()
            
            # Display
            if i % report_freq == 0:
                print(f"Epoch [{epoch+1}/{end_epoch}] Batch [{str(i).zfill(3)}/{len(loader)}] \
                      Loss Discriminator: {loss_disc.item():.4f} Loss Generator: {loss_gen.item():.4f}")
        
        if epoch % save_freq == 0:
            torch.save(generator.state_dict(), f"model/{name}_generator_epoch{str(epoch).zfill(2)}.pt")
            torch.save(discriminator.state_dict(), f"model/{name}_discriminator_epoch{str(epoch).zfill(2)}.pt")
            
        
        if (epoch+1) % plot_freq == 0:
            visualise(epoch+1, name, dataset, generator)
    
                
    torch.save(generator.state_dict(), f"model/{name}_generator_epoch{str(epoch).zfill(2)}.pt")
    torch.save(discriminator.state_dict(), f"model/{name}_discriminator_epoch{str(epoch).zfill(2)}.pt")