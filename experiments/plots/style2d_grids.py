""" Create density grid of Figure 3
    Requires: updating the data and output path (see below)
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import os    
from mpl_toolkits.axes_grid1 import ImageGrid

data_path = r'...\data\style2d'                  #<--adapt here
output_dir = r'...\images'                       #<--adapt here

#hyperparameters
latent_dim = 2

x = torch.linspace(-2, 2, 7)
xx, yy = torch.meshgrid((x, x))
grid= torch.stack((xx.flatten(), yy.flatten()), dim=1).double()

####################################
###########DNF######################
gan_images = np.load(os.path.join(data_path,'dnf_2_gan2d_paper_grid.npy')      
gan_images = gan_images.reshape([7,7,3,64,64])
gan_images = np.transpose(gan_images,axes = [1,0,2,3,4])
gan_images = gan_images.reshape([49,3,64,64])


boundary = 1.5
resolution = 7   
each = np.linspace(-boundary, boundary, resolution)
each_grid = np.meshgrid(*[each for _ in range(2)], indexing="ij")
each_grid = [x.flatten() for x in each_grid]       
gan_zs = np.vstack(each_grid).T

gan_images = np.clip(gan_images / 256.0, 0.0, 1.0)


gan_images.shape
size = 0.45

fig, ax = plt.subplots()
fig = plt.figure(figsize=(10., 10.))
#fig, ax = ps.figure(height=0.33*ps.TEXTWIDTH) #
for z, image in zip(gan_zs, gan_images):
    #print('z[0]',z[0])
    image_ = np.transpose(image, [1,2,0])
    plt.imshow(image_, extent=(z[0]-size/2, z[0]+size/2, z[1]-size/2, z[1]+size/2))

plt.xlabel(r"DNF latent variable $\tilde{u}_0$", labelpad=4, fontsize=25) 
plt.ylabel(r"DNF latent variable $\tilde{u}_1$", labelpad=1, fontsize=25)     
#plt.xlabel("StyleGAN latent variable $z_0$", labelpad=4) 
#plt.ylabel("StyleGAN latent variable $z_1$", labelpad=1)  
plt.xlim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.ylim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.xticks([-1., 0., 1.])
plt.yticks([-1., 0., 1.])
ax.tick_params(axis='y', which='major', pad=1)


plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'style2d_dnf_grid.pdf'), bbox_inches = 'tight')  

####################################
###########Style Gan################
gan_images = np.load(os.path.join(data_path,'grid.npy')  

boundary = 1.5
resolution = 7   
each = np.linspace(-boundary, boundary, resolution)
each_grid = np.meshgrid(*[each for _ in range(2)], indexing="ij")
each_grid = [x.flatten() for x in each_grid]       
gan_zs = np.vstack(each_grid).T

gan_images = gan_images.reshape((9, 9, 3, 64, 64))
gan_images = gan_images[1:-1, 1:-1, :, :, :]
gan_images = gan_images.reshape((49, 3, 64, 64))
gan_images = 0.5 + 255.0 * gan_images
gan_images = np.clip(gan_images / 256.0, 0.0, 1.0)

gan_images.shape
size = 0.45
fig, ax = plt.subplots()
fig = plt.figure(figsize=(10., 10.))
#fig, ax = ps.figure(height=0.33*ps.TEXTWIDTH) #
for z, image in zip(gan_zs, gan_images):
    #print('z[0]',z[0])
    image_ = np.transpose(image, [1,2,0])
    plt.imshow(image_, extent=(z[0]-size/2, z[0]+size/2, z[1]-size/2, z[1]+size/2))
      
plt.xlabel("StyleGAN latent variable $z_0$", labelpad=4, fontsize=25) 
plt.ylabel("StyleGAN latent variable $z_1$", labelpad=1, fontsize=25)  
plt.xlim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.ylim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.xticks([-1., 0., 1.])
plt.yticks([-1., 0., 1.])
ax.tick_params(axis='y', which='major', pad=1)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'style2d_grid.pdf'), bbox_inches = 'tight')  #dnf_gand2d_grid , dpi=72

####################################
###########VAE######################
gan_images = 0.5 + 255.0 * np.load(os.path.join(data_path,'grid_VAE.npy')  

gan_images = gan_images.reshape([7,7,3,64,64])
gan_images = np.transpose(gan_images,axes = [1,0,2,3,4])
gan_images = gan_images.reshape([49,3,64,64])

boundary = 1.5
resolution = 7   
each = np.linspace(-boundary, boundary, resolution)
each_grid = np.meshgrid(*[each for _ in range(2)], indexing="ij")
each_grid = [x.flatten() for x in each_grid]       
gan_zs = np.vstack(each_grid).T

gan_images = np.clip(gan_images / 256.0, 0.0, 1.0)
gan_images.shape
size = 0.45
fig, ax = plt.subplots()
fig = plt.figure(figsize=(10., 10.))
for z, image in zip(gan_zs, gan_images):
    image_ = np.transpose(image, [1,2,0])
    plt.imshow(image_, extent=(z[0]-size/2, z[0]+size/2, z[1]-size/2, z[1]+size/2))

plt.xlabel(r"InfoMax-VAE variable $\tilde{u}_0$", labelpad=4, fontsize=25) 
plt.ylabel(r"InfoMax-VAE variable $\tilde{u}_1$", labelpad=1, fontsize=25) 
plt.xlim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.ylim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.xticks([-1., 0., 1.])
plt.yticks([-1., 0., 1.])
ax.tick_params(axis='y', which='major', pad=1)


plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'style2d_vae_grid.pdf'), bbox_inches = 'tight')

####################################
###########PAE######################
gan_images = np.load(os.path.join(data_path,'grid_gan2d_pae.npy')  + 0.5                      

boundary = 1.5
resolution = 7   
each = np.linspace(-boundary, boundary, resolution)
each_grid = np.meshgrid(*[each for _ in range(2)], indexing="ij")
each_grid = [x.flatten() for x in each_grid]       
gan_zs = np.vstack(each_grid).T

gan_images.shape
size = 0.45

fig, ax = plt.subplots()
fig = plt.figure(figsize=(10., 10.))
for z, image in zip(gan_zs, gan_images):
    image_ = image 
    plt.imshow(image_, extent=(z[0]-size/2, z[0]+size/2, z[1]-size/2, z[1]+size/2))

plt.xlabel(r"PAE latent variable $\tilde{u}_0$", labelpad=4, fontsize=25) 
plt.ylabel(r"PAE latent variable $\tilde{u}_1$", labelpad=1, fontsize=25) 
plt.xlim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.ylim(-1.5 - 1.3*size/2, 1.5 + 1.3*size/2)
plt.xticks([-1., 0., 1.])
plt.yticks([-1., 0., 1.])
ax.tick_params(axis='y', which='major', pad=1)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'style2d_pae_grid.pdf'), bbox_inches = 'tight')