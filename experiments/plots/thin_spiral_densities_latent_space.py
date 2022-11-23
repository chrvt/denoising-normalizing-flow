""" Create lower half of Figure 2: Density estimation in latent-space
    Requires: updating the data and output path (see below)
    Make sure the model-names are updated in case you have changed them!
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import os    
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from scipy.stats import expon#norm#chi2
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
#### Wasserstein distance ####
from layers import SinkhornDistance
#### https://github.com/dfdazac/wassdistance ####


data_path = r'...\data\thin_spiral_latent'  #<---adapt here
output_dir = r'...\images'                        #<---adapt here

# create colormap
# ---------------
# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the first color of the colormap
# - 4/5 : existing colormap

# set upper part: 4 * 256/4 entries
upper = mpl.cm.jet(np.arange(256))
# set lower part: 1 * 256/4 entries
# - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
lower = np.ones((int(256/4),4)) * 0.8
# - modify the first three columns (RGB):
#   range linearly between white (1,1,1) and the first color of the upper colormap
for i in range(3):
  lower[:,i] = np.linspace(0.8, upper[0,i], lower.shape[0])
lower[0:8,:]=1
# combine parts of colormap
cmap = np.vstack(( lower, upper ))
# convert to matplotlib colormap
cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

############plotting
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title   

#hyperparameters
line ='--'
log_sale = False
y_title = 0.85


fig = plt.figure(figsize=(4., 6.))
axes = []
####################################
###########thin spiral##############
#original
n_pts = 100

ax = fig.add_subplot(3,2,1) 
ax1 = ax
axes += [ax1]
plt.title('original $\pi(u)$', y=y_title)

#calculate original density and Jacobian det
latent_test = np.load(os.path.join(data_path,'x_test_latent.npy'))
order = np.argsort(latent_test)
latent_test = latent_test[order] #sort: lowest to highest
probs_test = expon.pdf(latent_test,scale=0.3) 

c1 = 540 * 2* np.pi / 360
r = np.sqrt(latent_test) * c1
jac_det = ((1+r**2)/r**2) * c1**4 / 36   
factor = np.sqrt(2*np.pi*0.01)

ax.plot(latent_test,probs_test,label='original')
if log_sale:
    ax.set_yscale('log') 
    
unit   = 1
y_tick = np.array([0,3]) 
ax.set_yticks(y_tick)
    
ax.set_xticks([0,2])
ax.set_xlabel(r'$u$')
ax.xaxis.set_label_coords(0.5, -0.06 )
ax.set_ylabel(r'$\pi(u)$')
ax.yaxis.set_label_coords(-0.01, 0.45 )
plt.xlim(0,2.5)

####################################
###########standard flow############
ax = fig.add_subplot(3,2,2) 
ax2 = ax
axes += [ax2]

tag_model = 'standard NF'
log_prob =  np.load(os.path.join(data_path,'flow_1_thin_spiral_paper_latent_probs.npy'))
prob = np.exp(log_prob)   * factor  * np.sqrt(jac_det)

ax.plot(latent_test,probs_test,label='original')
ax.plot(latent_test,prob,label=tag_model,linestyle = line)
if log_sale:
    ax.set_yscale('log') 
ax.set_xticks([])
ax.set_yticks([])
plt.title(tag_model, y=y_title)

####################################
###########M-flow###################
tag_model = r'$\mathcal{M}-$flow'
ax = fig.add_subplot(3,2,3) 
ax3 = ax
axes += [ax3]
log_prob = np.load(os.path.join(data_path,'mf_1_thin_spiral_paper_latent_probs.npy'))
prob = np.exp(log_prob)   * factor  * np.sqrt(jac_det)
ax.plot(latent_test,probs_test,label='original')
ax.plot(latent_test,prob,label=tag_model,linestyle = line)
if log_sale:
    ax.set_yscale('log')    
ax.set_xticks([])
ax.set_yticks([])
plt.title(tag_model, y=y_title)

####################################
###########DNF###################
tag_model = 'DNF'
ax = fig.add_subplot(3,2,4) 
ax3= ax
axes += [ax3]
plt.title(tag_model) 
log_prob = np.load(os.path.join(data_path,'dnf_1_thin_spiral_paper_latent_probs.npy'))   
log_prob_dnf = log_prob
prob = np.exp(log_prob)  * factor  * np.sqrt(jac_det)

ax.plot(latent_test,probs_test,label='original')
ax.plot(latent_test,prob,label=tag_model,linestyle = line)
if log_sale:
    ax.set_yscale('log') 
ax.set_xticks([])
ax.set_yticks([])
plt.title(tag_model, y=y_title)

####################################
###########PAE######################
ax = fig.add_subplot(3,2,5) 
ax4= ax
axes += [ax4]
plt.title('PAE', y=y_title) 
log_prob = np.load(os.path.join(data_path,'pae_1_thin_spiral_paper_latent_probs.npy'))
prob = np.exp(log_prob) * factor  * np.sqrt(jac_det)
prob_PAE = prob
ax.plot(latent_test,probs_test,label='original')
ax.plot(latent_test,prob,label=tag_model,linestyle = line)
# ax.set_xlim(-4.8,6)   
# ax.set_ylim(-4.5,5)  
if log_sale:
    ax.set_yscale('log')    
ax.set_xticks([])
ax.set_yticks([])

####################################
###########VAE######################
ax = fig.add_subplot(3,2,6) 
ax5= ax
axes += [ax5]
plt.title('VAE', y=y_title)
log_prob = -np.load(os.path.join(data_path,'VAE_latent_probs.npy'))
prob = np.exp(log_prob)   * factor  * np.sqrt(jac_det)
ax.plot(latent_test,probs_test,label='original')
ax.plot(latent_test,prob,label=tag_model,linestyle = line)
if log_sale:
    ax.set_yscale('log')    
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout(h_pad=0.2)                                                      #<--adapt here
fig.savefig(os.path.join(output_dir, 'thin_spiral_latent_densities.pdf'), bbox_inches = 'tight')