# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:52:14 2023

@author: napat
"""
import torch, os
import numpy as np
from scipy.linalg import sqrtm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    
    p_yx = F.softmax(p_yx, dim=1)
    # calculate p(y)
    p_y = p_yx.mean(axis=0, keepdims=True)
    # kl divergence for each image
    kl_d = p_yx * (torch.log(p_yx + eps) - torch.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(dim=1)
    # average over images
    avg_kl_d = (sum_kl_d).mean()
    # undo the logs
    inc_score = torch.exp(avg_kl_d)
    return inc_score

# calculate frechet inception distance
def calculate_fid_np(act1, act2):
    act1 = np.array(act1.detach())
    act2 = np.array(act2.detach())
    # for a in (act1, act2):
    #     if isinstance(a, torch.Tensor):
    #         a = np.array(a.detach())
    #         print(type(a))
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# calculate frechet inception distance
def calculate_fid_torch(act1, act2):
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), torch.cov(act1.T)
    mu2, sigma2 = act2.mean(axis=0), torch.cov(act2.T)
    # calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = torch.from_numpy(sqrtm(sigma1.matmul(sigma2).detach()))
    # check and correct imaginary numbers from sqrt
    if torch.is_complex(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    os.chdir("C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/Project")
    batch_size = 500
    latent_dim_gan = 64
    latent_dim_vae = 64
    image_width = 28
    image_dim = image_width**2
    # n_classes = 10
    hidden_size = 256
    
    
    
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
    
    from classify import Classifier
    from cgan import Generator
    from cvae import VAE
    
    clf = Classifier(n_channels, n_classes)
    clf.load_state_dict(torch.load("model/mnist_clf_01_epoch09.pt"))
    
    gan = Generator(latent_dim_gan, image_dim, n_classes, hidden_size)
    gan.load_state_dict(torch.load("model/mnist_cgan_01_generator_epoch99.pt"))
    
    vae = vae = VAE(image_dim, hidden_size, latent_dim_vae, n_classes)
    vae.load_state_dict(torch.load("model/mnist_cvae_01_epoch00.pt"))
    
    # x,y = next(iter(loader_test))
    
    # p_yx = clf(x)
    # p_yx = F.softmax(p_yx, dim=1)
    
    scores = torch.zeros((len(loader_test),3))
    fid_scores = torch.zeros((len(loader_test),2))
    for i, (x, y) in enumerate(loader_test):
        with torch.no_grad():
            # Real Images
            p_yx = clf(x)
            act_real = clf.get_last_layer(x)
            # p_yx = F.softmax(p_yx, dim=1)
            inc_score = calculate_inception_score(p_yx)
            scores[i,0] = inc_score
            
            # GAN Generateed Imaged
            noise = torch.randn(batch_size, latent_dim_gan)
            gan_gen = gan(noise, y).view(-1,n_channels,image_width,image_width).detach()
            p_yx = clf(gan_gen)
            act_gan = clf.get_last_layer(gan_gen)
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_np(act_real, act_gan)
            scores[i,1] = inc_score
            fid_scores[i,0] = fid_score
            
            # VAE Generateed Imaged
            vae_gen, _, _ = vae(x.view(-1, image_dim), y)
            vae_gen = vae_gen.view(-1,n_channels,image_width,image_width).detach()
            p_yx = clf(vae_gen)
            act_vae = clf.get_last_layer(vae_gen)
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_np(act_real, act_vae)
            scores[i,2] = inc_score
            fid_scores[i,1] = fid_score
        
    labels = ["Real","CGAN",'CVAE']
    plt.boxplot(scores.numpy())
    plt.xticks(ticks=np.arange(1,4),labels=labels)
    plt.title("Inception Scores")
    plt.show();
    
    plt.boxplot(fid_scores.numpy())
    plt.xticks(ticks=np.arange(1,3),labels=labels[1:])
    plt.title("FID Score")
    plt.show();
    # scores = torch.tensor(scores)
    # scores.min(), scores.max()
