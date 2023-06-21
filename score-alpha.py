# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:52:14 2023

@author: Napat Sahapat (20619406)

Calculate and Plot Inception Score and FID Scores
to Compare Different Methods

Comparing 6 Models: Adding CGAN with 3 alpha values: 0, 0.1, 0.2
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

# source: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
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

# source: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
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
    # Parameters
    batch_size = 500
    latent_dim_gan = 64
    latent_dim_vae = 64
    image_width = 28
    image_dim = image_width**2
    # n_classes = 10
    hidden_size = 256
    
    trial = 2    
    
    # Make sure we're in the right directory
    if os.path.exists("Project"):
        # Project already exists -> Change into Project
        os.chdir("./Project")
    elif not os.path.exists("Project") and not os.path.exists("../Project"):
        # Project hasn't existed and is not currently inside -> Create and Change into Project
        os.mkdir("Project")
        os.chdir("./Project")
    from classify import Classifier
    from cgan import Generator
    from cvae import VAE as CVAE
    from vae import VAE
    
    # Load dataset
    # dataset = MNIST('C:/Western Sydney/2023-Autumn/MATH 7017 - Probabilistic Graphical Models/data', 
    #                        transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #     ]),
    #     download=False)
    dataset_test = MNIST('../data', 
                           transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        train=False,      
        download=True)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    # Extract info from dataset
    n_classes = len(dataset_test.classes)
    n_channels = dataset_test[0][0].shape[0]
    
    # Load pre-trained models
    # Classifier
    clf = Classifier(n_channels, n_classes)
    clf.load_state_dict(torch.load("model/mnist_clf_01_epoch09.pt"))
    
    # CGAN
    gan = Generator(latent_dim_gan, image_dim, n_classes, hidden_size)
    gan.load_state_dict(torch.load("model/mnist_cgan_02_generator_epoch99.pt"))
    gan2 = Generator(latent_dim_gan, image_dim, n_classes, hidden_size)
    gan2.load_state_dict(torch.load("model/mnist_cgan_01_generator_epoch99.pt"))
    gan3 = Generator(latent_dim_gan, image_dim, n_classes, hidden_size)
    gan3.load_state_dict(torch.load("model/mnist_cgan_03_generator_epoch99.pt"))    
    
    # CVAE
    cvae1 = CVAE(image_dim, hidden_size, latent_dim_vae, n_classes)
    cvae1.load_state_dict(torch.load("model/mnist_cvae_02_epoch99.pt"))

    vae = VAE(image_dim, hidden_size, latent_dim_vae, n_classes, conditional=False)
    vae.load_state_dict(torch.load("model/mnist_vae_02_epoch99.pt"))
    
    # Begin Evaluating
    scores = torch.zeros((len(loader_test),6))
    fid_scores = torch.zeros((len(loader_test),5))
    for i, (x, y) in enumerate(loader_test):
        with torch.no_grad():
            # Real Images
            p_yx = clf(x)
            act_real = clf.get_last_layer(x)
            inc_score = calculate_inception_score(p_yx)
            scores[i,0] = inc_score
            
            # GAN Generateed Imaged
            noise = torch.randn(batch_size, latent_dim_gan)
            gan_gen = gan(noise, y).view(-1,n_channels,image_width,image_width).detach()
            # Pass through classifier
            p_yx = clf(gan_gen)
            act_gan = clf.get_last_layer(gan_gen)
            # Get scores
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_torch(act_real, act_gan)
            # Record
            scores[i,1] = inc_score
            fid_scores[i,0] = fid_score
            
            # GAN Generateed Imaged
            noise = torch.randn(batch_size, latent_dim_gan)
            gan_gen = gan2(noise, y).view(-1,n_channels,image_width,image_width).detach()
            # Pass through classifier
            p_yx = clf(gan_gen)
            act_gan = clf.get_last_layer(gan_gen)
            # Get scores
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_torch(act_real, act_gan)
            # Record
            scores[i,2] = inc_score
            fid_scores[i,1] = fid_score
            
            # GAN Generateed Imaged
            noise = torch.randn(batch_size, latent_dim_gan)
            gan_gen = gan3(noise, y).view(-1,n_channels,image_width,image_width).detach()
            # Pass through classifier
            p_yx = clf(gan_gen)
            act_gan = clf.get_last_layer(gan_gen)
            # Get scores
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_torch(act_real, act_gan)
            # Record
            scores[i,3] = inc_score
            fid_scores[i,2] = fid_score
            
            # cvae1 Generateed Imaged
            cvae1_gen, _, _ = cvae1(x.view(-1, image_dim), y)
            cvae1_gen = cvae1_gen.view(-1,n_channels,image_width,image_width).detach()
            # Pass through classifier            
            p_yx = clf(cvae1_gen)
            act_cvae1 = clf.get_last_layer(cvae1_gen)
            # Get scores
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_torch(act_real, act_cvae1)
            # Record
            scores[i,4] = inc_score
            fid_scores[i,3] = fid_score
            
            # VAE Generateed Imaged
            vae_gen, _, _ = vae(x.view(-1, image_dim), y)
            vae_gen = vae_gen.view(-1,n_channels,image_width,image_width).detach()
            # Pass through classifier            
            p_yx = clf(vae_gen)
            act_vae = clf.get_last_layer(vae_gen)
            # Get scores
            inc_score = calculate_inception_score(p_yx)
            fid_score = calculate_fid_torch(act_real, act_vae)
            # Record
            scores[i,5] = inc_score
            fid_scores[i,4] = fid_score
    
    # Plot Inception Score
    labels = ["Real","CGAN-0",'CGAN-0.1','CVAE-0.2','CVAE','VAE']
    plt.boxplot(scores.numpy())
    plt.xticks(ticks=np.arange(1,scores.shape[1]+1),labels=labels)
    plt.title(f"Inception Scores\nBatch Size: {batch_size}, Num Samples: {len(loader_test)}")
    plt.savefig(f"image/boxplot_inception_{str(trial).zfill(2)}.png")
    plt.show();
    
    # Plot FID Score
    plt.boxplot(fid_scores.numpy())
    plt.xticks(ticks=np.arange(1,fid_scores.shape[1]+1),labels=labels[1:])
    plt.title(f"FID Score\nBatch Size: {batch_size}, Num Samples: {len(loader_test)}")
    plt.savefig(f"image/boxplot_fid_{str(trial).zfill(2)}.png")
    plt.show();
