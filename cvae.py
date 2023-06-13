# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:27:28 2023

@author: napat
"""

import torch, os
from torch import nn, optim
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.utils import make_grid, save_image

class VAE(nn.Module):
    def __init__(self, image_dim, hidden_dim, latent_dim, n_classes):
        super(VAE, self).__init__()
        self.n_classes = n_classes
        self.image_dim = image_dim
        self.input_dim = image_dim + n_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def concat_label(self, x, labels):
        labels = F.one_hot(labels, num_classes=self.n_classes)
        x = torch.cat((x, labels), dim=1)
        return x
        
    def forward(self, x, labels):
        mu, logvar = self.encode(self.concat_label(x, labels))
        z = self.reparameterise(mu, logvar)
        output = self.decode(self.concat_label(z, labels))
        return output, mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def visualise(epoch, name, dataset, vae, width=5, save=True):
    with torch.no_grad():
        
        loader2 = DataLoader(dataset, batch_size=width**2, shuffle=True)            
        img, labels = next(iter(loader2))
        img = img.view(-1, image_dim).to(device)
        labels = labels.to(device)
        recon, mu, logvar = vae(img, labels)
        recon = recon.view(-1, 1, image_width, image_width)
        img = img.view(-1, 1, image_width, image_width)
        
        # real2 = real.view(-1, 1, image_width, image_width)
        img_grid_real = make_grid(img, normalize=True, nrow=width)
        img_grid_fake = make_grid(recon, normalize=True, nrow=width)
        
        plt.figure(figsize=(10,10))
        
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
    
    name = 'mnist_cvae'
    trial = 2
    name = name + "_" + str(trial).zfill(2)
    
    os.chdir("C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/Project")
    for direc in ["model",'image']:
        if not os.path.exists(direc):
            os.mkdir(direc)    
    
    # Parameters
    # latent_dim = 2
    # hidden_sqrt = 12   # 12x12 = 144 hidden units
    # hidden_size = hidden_sqrt * hidden_sqrt
    
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
    # alpha = 0.1
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = MNIST('C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_classes = len(dataset.classes)
    
    vae = VAE(image_dim, hidden_size, latent_dim, n_classes).to(device)
    
    # generator = Generator(latent_dim, image_dim, n_classes, hidden_size).to(device)
    # discriminator = Discriminator(image_dim, n_classes, hidden_size).to(device)
    
    if isinstance(num_epochs, tuple) and len(num_epochs) > 1:
        start_epoch = num_epochs[0] - 1 # Previous iteration
        end_epoch = num_epochs[1]
        if os.path.exists(f"model/{name}_epoch{str(start_epoch).zfill(2)}.pt"):
            vae.load_state_dict(torch.load(f"model/{name}_epoch{str(start_epoch).zfill(2)}.pt"))
    elif isinstance(num_epochs, int):
        start_epoch = 0
        end_epoch = num_epochs
        num_epochs = start_epoch, end_epoch
    
    optimiser = optim.Adam(vae.parameters(), lr=lr)
    
    # criterion = nn.BCELoss()
    
    visualise("Initialisation", name, dataset, vae)
    
    # Training Loop
    for epoch in range(*num_epochs):
        for i, (images, labels) in enumerate(loader):
            images = images.view(-1, image_dim).to(device)
            labels = labels.to(device)
            
            outputs, mu, logvar = vae(images, labels)
            loss = loss_function(outputs, images, mu, logvar)
            
            vae.zero_grad()
            loss.backward()
            optimiser.step()
            
            # Display
            if i % report_freq == 0:
                print(f"Epoch [{epoch+1}/{end_epoch}] Batch [{str(i).zfill(3)}/{len(loader)}] \
                      Loss: {loss.item():.4f}")
        
        if epoch % save_freq == 0:
            torch.save(vae.state_dict(), f"model/{name}_epoch{str(epoch).zfill(2)}.pt")        
        
        if (epoch+1) % plot_freq == 0:
            visualise(epoch+1, name, dataset, vae)
    
    
    torch.save(vae.state_dict(), f"model/{name}_epoch{str(epoch).zfill(2)}.pt")                    