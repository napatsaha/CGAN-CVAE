# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:41:22 2023

@author: Napat Sahapat (20619406)

Exploring 2D Latent Variable Space with VAE and CVAE
"""
from cgan import Generator
from vae import VAE
import torch, os
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Directory Management
    if os.path.exists("Project"):
        # Project already exists -> Change into Project
        os.chdir("./Project")
    elif not os.path.exists("Project") and not os.path.exists("../Project"):
        # Project hasn't existed and is not currently inside -> Create and Change into Project
        os.mkdir("Project")
        os.chdir("./Project")
    os.chdir("./Project")

    image_width = 28
    image_dim = image_width**2
    latent_dim = 2
    hidden_size = 256
    batch_size = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = MNIST('../data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_classes = len(dataset.classes)
    
    vae = VAE(image_dim, hidden_size, latent_dim, n_classes, conditional=False).to(device)

    vae.load_state_dict(torch.load("model/mnist_vae_01_epoch99.pt"))

    latent_mu = torch.empty(0,5).to(device)

    for X, y in loader:
    # X, y = next(iter(loader))
        X = X.view(-1, image_dim).to(device)
        y = y.to(device)
        
        inp = vae.concat_label(X, y) if vae.conditional else X
        
        mu, logvar = vae.encode(inp)
        
        batch_mu = torch.cat([y.unsqueeze(-1), mu, logvar], dim=1)
        # batch_mu = batch_mu.cpu().detach()
        
        latent_mu = torch.cat([latent_mu, batch_mu], dim=0)
        
    latent_mu = latent_mu.cpu().detach()
    
    # Mu
    sc = plt.scatter(latent_mu[:,1], latent_mu[:,2], c=latent_mu[:,0],
                cmap=plt.cm.Paired, marker='.', s=10, alpha=0.5)    
    plt.legend(*sc.legend_elements())
    plt.xlabel("$\mu_1$")
    plt.ylabel("$\mu_2$", rotation=0)
    plt.title("VAE Latent means by digit class")
    plt.show()

    # Logvar
    sc = plt.scatter(latent_mu[:,3], latent_mu[:,4], c=latent_mu[:,0],
                cmap=plt.cm.Paired, marker='.', s=10, alpha=0.5)    
    plt.legend(*sc.legend_elements())
    plt.xlabel("$\log{\sigma}_1$")
    plt.ylabel("$\log{\sigma}_2$", rotation=0)
    plt.title("VAE Latent log var by digit class")
    plt.show()