"""
Created on Tue Jun 13 15:52:14 2023

@author: Napat Sahapat (20619406)

Training Hyper-parameters:
    For Reference Only
"""

# CGAN
    trial = 1
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

# CVAE
    trial = 2
    batch_size = 64
    lr = 1e-3
    num_epochs = 100
    report_freq = 100
    plot_freq = 5
    save_freq = 5
    
    image_width = 28
    image_dim = image_width**2
    latent_dim = 2
    hidden_size = 256


    trial = 3
    batch_size = 64
    lr = 1e-3
    num_epochs = 100
    report_freq = 100
    plot_freq = 5
    save_freq = 5
    
    image_width = 28
    image_dim = image_width**2
    latent_dim = 2
    hidden_size = 256

# VAE
    trial = 1
    batch_size = 64
    lr = 1e-3
    num_epochs = 100
    report_freq = 100
    plot_freq = 5
    save_freq = 5
    
    image_width = 28
    image_dim = image_width**2
    latent_dim = 2
    hidden_size = 256