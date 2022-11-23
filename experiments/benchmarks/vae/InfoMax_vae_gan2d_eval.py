# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:28:27 2021

@author: Horvat
"""


import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from abc import abstractmethod
import torch.nn.init as init
import os 
from torch.nn.utils import clip_grad_norm_
import tempfile
from matplotlib import pyplot as plt

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except:
    print("Could not import fid_score, make sure that pytorch-fid is in the Python path")
    calculate_fid_given_paths = None

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

def train(VAE, D, optim_VAE, optim_D,data,vali, epochs=100):     
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

            
            
            vae_loss.backward() #retain_graph=True
            
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
            print('val_loss[1] / 40',val_loss[1] / 40)
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
        print('last loss ',vae_loss.item())
    return VAE, D


torch.cuda.empty_cache() 
torch.cuda.max_memory_allocated()

# PATH 
filepath = r'...\results\checkpoints_last.pt'           # path to model
true_dir = r'...\experiments\data\samples\gan2d\test'   # path to test samples for FID evaluation
save_path = r'...\results'                              # path to folder in which to save the results

#MODEL
image_size = 64
max_iter = int(10)
z_dim = 2 #100
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 10

VAE = CNNVAE1(z_dim=z_dim).to(device) 
D = Discriminator(z_dim=z_dim).to(device)

checkpoint = torch.load(filepath)
VAE.load_state_dict(checkpoint['VAE_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])

print('VAE parameters: ', sum(p.numel() for p in VAE.parameters()))
print('Discriminator parameters: ', sum(p.numel() for p in D.parameters()))

VAE.eval()
D.eval()
x_gen = VAE.sample(N=1000)
x_gen = 0.5 + 255.0 * x_gen
dataset = x_gen.detach().cpu().numpy()
np.save(os.path.join(save_path,'samples_VAE.npy'),x_gen.cpu().detach().numpy())

def array_to_image_folder(data, folder):
    for i, x in enumerate(data):
        x = np.clip(np.transpose(x, [1, 2, 0]) / 256.0, 0.0, 1.0)
        
        plt.imsave(f"{folder}/{i}.jpg", x)

def sum_except_batch(x, num_batch_dims=1):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

with torch.no_grad():        
    with tempfile.TemporaryDirectory() as gen_dir:
        print("Storing generated images in temporary folder")
        array_to_image_folder(dataset, gen_dir)
        
        print("Beginning FID calculation with batchsize 50")
        fid = calculate_fid_given_paths([gen_dir, true_dir], 50, "cuda", 2048)
        np.save(os.path.join(save_path,'FID.npy'),fid)
        print("FID score ",fid)

print('Calculate mean reconstruction error')
test = 0.5 + 255.0 * np.load(r'D:\manifold-flow-public\experiments\data\samples\gan2d\test.npy')
test = torch.from_numpy(test).to(device).float()
x_rec, mu, logvar, z = VAE(test)
x_rec = 0.5 + 255.0 * x_rec
MRE = sum_except_batch((test - x_rec) ** 2) ** 0.5
np.save(os.path.join(save_path,'MRE_VAE.npy'),MRE.detach().cpu().numpy())
print('Mean reconstruction error on test set is ',MRE.mean())


print('generate image grid')
x = torch.linspace(-2, 2, 7)
xx, yy = torch.meshgrid((x, x))
grid= torch.stack((xx.flatten(), yy.flatten()), dim=1).to(device).float()
images = []
#t = time.time()    
for k in range(7):
    for j in range(7):
        i = 7*k
        z = torch.ones([1,2,1,1]).to(device).float()
        z[0,:,0,0] = grid[i+j,:].reshape([1,2])
        img_ij = VAE.decode(z)
        images += [img_ij.detach().cpu().numpy()]
#elapsed = time.time() - t
images = np.concatenate(images,axis=0)
np.save(os.path.join(save_path,'grid_VAE.npy'),images)
