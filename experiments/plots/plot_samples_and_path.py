""" Create samples and image path of Figure 4
    Requires: updating the data and output path (see below)
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import os 
import plotsettings as ps


output_dir = r'...\images'                       #<--adapt here

###for StyleGAN 64###
data_path = r'...\data\style64d'                 #<--adapt here

name = 'gan64' # adapt paths
x_test = np.load(os.path.join(data_path,'original_samples.npy'))
x_gen_mf = np.load(os.path.join(data_path,'mf_64_gan64d_paper_samples.npy'))
x_gen_dnf = np.load(os.path.join(data_path,'mf_64_gan64d_paper_samples.npy'))
x_gen_vae = np.load(os.path.join(data_path,'samples_VAE.npy'))
x_gen_pae = np.transpose(256.0 * np.load(os.path.join(data_path,'samples_gan64_pae.npy')),[0,3,1,2])     #[0:5,:]

path_dnf = np.load(os.path.join(data_path,'dnf_64_gan64d_paper_image_path.npy')) 
path_vae = 256 * os.path.join(data_path,'img_path_VAE.npy')
path_mf = os.path.join(data_path,'mf_64_gan64d_paper_image_path.npy') 
path_pae = np.transpose(256.0 * np.load(os.path.join(data_path,'def2_pae_img_path0.npy')),[0,3,1,2])

images = [x_test, x_gen_mf, x_gen_dnf, x_gen_vae, x_gen_pae]
paths = [path_mf,path_dnf,path_vae,path_pae]
labels = ["Original", r"$\mathcal{M}$-flow", r"DNF",r"InfoMax-VAE",r"PAE"]
nrows = 5    
ncols = 10   

###uncomment for CelebA###
#data_path = r'...\data\celeba'                 #<--adapt here
#name = 'celebA'
#x_test = np.load(os.path.join(data_path,'test.npy')) 
#x_gen_mf =  np.load(os.path.join(data_path,'mf_512_celeba_paper_samples.npy'))
#x_gen_dnf = np.load(os.path.join(data_path,'dnf_512_celeba_paper_samples.npy'))

#path_mf = np.load(os.path.join(data_path, 'mf_512_celeba_paper_image_path.npy'))
#path_dnf = np.load(os.path.join(data_path,'dnf_512_celeba_paper_image_path.npy'))

# images = [x_test, x_gen_mf, x_gen_dnf]
# paths = [path_mf,path_dnf]
# labels = ["Original", r"$\mathcal{M}$-flow", r"DNF"]
# nrows = 3   
# ncols = 10     


fig, ax = plt.subplots()
fig = plt.figure(figsize=(ncols, nrows))
#add seperating line
plt.plot([70, 70], [90, 95], color="black")
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.box(False) 
               
for i in range(nrows):
    if i == 0:
        for j in range(ncols):
            if j < 5:
                x = images[i][nrows*i+j]
                x = np.clip(x , 0.0, 1.0)
                img = np.transpose(x, [1, 2, 0])
                ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if j==0:
                    plt.ylabel(labels[i])
                if j==2:
                    plt.title('Samples')
            elif j == 5:
            
                x = images[i][3]
                x = np.clip(x , 0.0, 1.0)
                img = np.transpose(x, [1, 2, 0])
                ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            
            elif j == 9:
                
                x = images[i][2]
                x = np.clip(x , 0.0, 1.0)
                img = np.transpose(x, [1, 2, 0])
                ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            
            elif j==7:
                ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
                ax.axis('off')
                plt.title('Latent Interpolation')
    else:
        for j in range(0,5):
            x = images[i][nrows*i+j]
            x = np.clip(x / 256.0, 0.0, 1.0)
            img = np.transpose(x, [1, 2, 0])
            ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                plt.ylabel(labels[i])
        for j in range(5,10):
            x = paths[i-1][j-5] #nrows*i+
            x = np.clip(x / 256.0, 0.0, 1.0)
            img = np.transpose(x, [1, 2, 0])
            ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
            ax.imshow(img, vmin=0,vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                plt.ylabel(labels[i])        

plt.tight_layout()
fig.savefig(os.path.join(output_dir, name+'_samples.pdf'), bbox_inches = 'tight')