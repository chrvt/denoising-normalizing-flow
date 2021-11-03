# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:28:27 2021

@author: Horvat
"""


import torch; torch.manual_seed(1233)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from abc import abstractmethod
import torch.nn.init as init
import os 
from torch.nn.utils import clip_grad_norm_

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(nn.Module):
    def __init__(self, z_dim=100):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64*64*3 + z_dim, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 10),
            nn.LeakyReLU(0.2, True),
            nn.Linear(10, 1),

        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, z):
        x = x.view(-1, 64*64*3)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()



class CNNVAE1(nn.Module):

    def __init__(self, z_dim=100):
        super(CNNVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 2*z_dim, 2, 1),

        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):

        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(4, z_dim, 1, 1), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x)
            mu = stats[:, :self.z_dim].clone()
            logvar = stats[:, self.z_dim:].clone()
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()
    def sample(self,N):
        z = torch.randn(N, z_dim, 1, 1)
        z = z.to(device).float()
        """
        samples = []
        shape = torch.zeros([1,3,64,64])
        for n in range(N):
            gen_z =  #Variable(, requires_grad=False)
            gen_z = gen_z.to(device)
            x = self.decode(gen_z)
            samples.append(x.view(shape.size()).detach().numpy())
        samples = np.concatenate(samples, axis=0)
        """
        return self.decode(z)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def recon_loss(x_recon, x):
    n = x.size(0)
    loss = F.binary_cross_entropy(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z

def loss(VAE,D,x_true):
    x_recon, mu, logvar, z = VAE(x_true)
    D_xz = D(x_true, z)
    z_perm = permute_dims(z)
    D_x_z = D(x_true, z_perm)
    
    Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))
    
    vae_recon_loss = recon_loss(x_recon, x_true)
    vae_kld = kl_divergence(mu, logvar)
    vae_loss = vae_recon_loss + vae_kld + gamma * Info_xz
    return vae_loss

def train(VAE, D, optim_VAE, optim_D, vae_scheduler ,D_scheduler, data,vali, epochs=100):     
    val_best = 2**23
    VAE_parameters = list(VAE.parameters())
    D_parameters = list(D.parameters())
    for epoch in range(epochs):
        print('start training witch epoch ',epoch)
        for x in data:
            x_true = (x).to(device).float() #            
            optim_VAE.zero_grad()
            optim_D.zero_grad()
            
            vae_loss = loss(VAE,D,x_true)

            
            
            vae_loss.backward(retain_graph=True) #retain_graph=True
            
            clip_grad_norm_(D_parameters, 5.0)
            clip_grad_norm_(VAE_parameters, 5.0)
            
            optim_VAE.step()
            optim_D.step()            
        with torch.no_grad():
            VAE.eval()
            D.eval()
            val_loss = np.zeros(2)
            for x in vali:
                vae_loss = loss(VAE,D,x_true)
                val_loss[1] += float(vae_loss.item()) 
            #print('val_loss[1] / 40',val_loss[1] / 40)
            if val_loss[1] / 40 < val_best:  #average val lass per batch
                val_loss[0] = epoch
                val_best = val_loss[1] / 40                    
                print('new best model at epoch ',epoch+1)
                checkpoint = {'VAE_state_dict': VAE.state_dict(),
                               'D_state_dict': D.state_dict(),
                               }
                torch.save(checkpoint , os.path.join(r'D:\VAE_thin_spiral\results', 'checkpoints.pt'))
            VAE.train()
            D.train()
        if vae_scheduler is not None:
            vae_scheduler.step()
            D_scheduler.step() 
        #print('last loss ',vae_loss.item())
    return VAE, D


torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache() 
torch.cuda.max_memory_allocated()

##PATH
save_path = r'...\results'
data_path = r'...\datasets\2D_style_gan\train.npy')

##DATA: load data and preprocesss
style = torch.from_numpy(255*np.load(data_path)
style = style + torch.rand_like(style)
style_train = torch.clamp(style[0:9000,:]/256,0,1) )
style_val = torch.clamp(style[9000:,:]/256,0,1)  

data_loader = DataLoader(dataset=style_train, batch_size=100, shuffle=True, num_workers=0) #, drop_last=True)
val_loader = DataLoader(dataset=style_val, batch_size=100, shuffle=True, num_workers=0) #, drop_last=True)

#MODEL
image_size = 64
max_iter = int(10)
z_dim = 2 
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 10

VAE = CNNVAE1(z_dim=z_dim).to(device) 
D = Discriminator(z_dim=z_dim).to(device)

""" # load model
filepath = r'D:\VAE_thin_spiral\results\checkpoints.pt'
checkpoint = torch.load(filepath)
VAE.load_state_dict(checkpoint['VAE_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])

#with scheduler

optim_VAE = torch.optim.AdamW(VAE.parameters(), lr=3e-4, weight_decay=1.0e-5) #W
VAE_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_VAE, T_max = 100)
optim_D = torch.optim.AdamW(D.parameters(), lr=lr_D, weight_decay=1.0e-5)
D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D, T_max = 100)
"""
VAE_scheduler,D_scheduler = None, None
optim_VAE =torch.optim.Adam(VAE.parameters(), lr=lr_D, betas=(beta1_D, beta2_D)) #, weight_decay=1.0e-5
optim_D =torch.optim.Adam(D.parameters(), lr=0.0001, betas=(beta1_D, beta2_D))  #, weight_decay=1.0e-5

print('VAE parameters: ', sum(p.numel() for p in VAE.parameters()))
print('Discriminator parameters: ', sum(p.numel() for p in D.parameters()))

VAE, D = train(VAE, D, optim_VAE,optim_D, VAE_scheduler,D_scheduler, data_loader,val_loader)

checkpoint = {'VAE_state_dict': VAE.state_dict(),
               'D_state_dict': D.state_dict(),
               }

torch.save(checkpoint , os.path.join(save_path, 'checkpoints_last.pt'))

print('training finished, have a nice day')

