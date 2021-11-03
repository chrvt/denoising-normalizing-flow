""" Create upper half of Figure 2: Density estimation in data-space
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

def log_density(dataset, x):
    if dataset == 'thin_spiral':
        z = np.linalg.norm(x,axis=1) * 3
        d1x = - np.cos(z) * z 
        d1y =   np.sin(z) * z 
        x_hat = np.stack([ d1x,  d1y], axis=1) / 3
        print('distances',np.abs( np.linalg.norm(x_hat,axis=1)-np.linalg.norm(x,axis=1) ))
        ind = np.abs( np.linalg.norm(x_hat,axis=1)-np.linalg.norm(x,axis=1) ) <= 10**(-6)
        logp = - np.inf * np.ones([z.shape[0],1])
        logp[ind] = 0
        
    return logp

    def sample(self, n, parameters=None):
        z = self._draw_z(n)
        x = self._transform_z_to_x(z)
        return x


data_path = r'D:\PROJECTS\DNF\NIPS_\Denoising_NF\Python\data\thin_spiral'
output_dir = r'D:\PROJECTS\DNF\NIPS_\Denoising_NF\images'       
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

#hyperparameters
############plotting
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title   

y_title=0.96

fig = plt.figure(figsize=(4., 6.))
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('None')
plt.gca().spines['right'].set_color('None')
plt.gca().spines['top'].set_color('None')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
axes = []
####################################
###########thin spiral##############
n_pts = 100
ax = fig.add_subplot(3,2,1) 
ax1 = ax
axes += [ax1]
plt.title('original $p(x)$',y=y_title)

x = torch.linspace(-4.8,6, n_pts)
y = torch.linspace(-4.5,5, n_pts)
xx, yy = torch.meshgrid((x, y))
grid= torch.stack((xx.flatten(), yy.flatten()), dim=1).numpy()
xx_flat = xx.flatten()
yy_flat = yy.flatten()
grid = np.stack((xx.flatten(), yy.flatten()), axis=1)

log_prob = log_density('thin_spiral',grid)
prob = np.exp(log_prob)
ax.pcolormesh(xx.numpy(), yy.numpy(), prob.reshape(n_pts,n_pts),color='white') #prob.reshape(n_pts,n_pts), cmap=plt.cm.jet,shading='auto')

numbs = np.random.exponential(scale=0.3,size=10000) #(np.random.normal(0,0.1,10000))**2
N = 1000
numbs = np.linspace(0,3.5,N)
#numbs = np.arange(-2,2,1000)
probs = expon.pdf(numbs,scale=0.3) #,df=1,loc=0,scale=0.1)
z = np.sqrt(numbs) * 540 * (2 * np.pi) / 360  #np.linspace(0,1,1000)
z = np.sort(z)
d1x = - np.cos(z) * z 
d1y =   np.sin(z) * z 
x = np.stack([ d1x,  d1y], axis=1) / 3

c1 = 540 * 2* np.pi / 360
r = np.sqrt(numbs) * c1
jac_det = ((1+r**2)/r**2) * c1**4 / 36   
factor = np.sqrt(2*np.pi*0.01)

#get scaling right
p_max = np.max(probs / np.sqrt(jac_det))  #we are going from latent to data space, therefore divide by jac_det
normalize = matplotlib.colors.Normalize(vmin=0, vmax=p_max,clip=True)  #10 is max value if p*(x)
probs = normalize(probs / np.sqrt(jac_det))
for i in range(N-1):
    z2 = x[i,0]**2+x[i,1]**2
    p_ = probs[i] 
    ax.plot(x[i:i+2,0], x[i:i+2,1], color=cmap(p_),linewidth=0.7)
ax.set_xlim(-4.8,6)  
ax.set_ylim(-4.5,5)    
ax.set_xticks([])
ax.set_yticks([])

####################################
###########standard flow############
ax = fig.add_subplot(3,2,2) 
ax2 = ax
axes += [ax2]
plt.title('standard NF',y=y_title)
log_prob =  np.load(os.path.join(data_path,'flow_1_thin_spiral_paper_log_grid_likelihood.npy'))
prob = np.exp(log_prob) 
ax.pcolormesh(xx.numpy(), yy.numpy(), prob.reshape(n_pts,n_pts),cmap=cmap)
ax.set_xticks([])
ax.set_yticks([])

####################################
###########M-flow###################
ax = fig.add_subplot(3,2,3) 
ax3 = ax
axes += [ax3]
plt.title(r'$\mathcal{M}-$flow',y=y_title)
log_prob = np.load(os.path.join(data_path,'mf_1_thin_spiral_paper_log_grid_likelihood.npy'))
prob = np.exp(log_prob)
ax.pcolormesh(xx.numpy(), yy.numpy(), prob.reshape(n_pts,n_pts),cmap=cmap,shading='auto') 
ax.set_xticks([])
ax.set_yticks([])

####################################
###########DNF######################
ax = fig.add_subplot(3,2,4) 
ax3= ax
axes += [ax3]
plt.title('DNF',y=y_title) 
log_prob = np.load(os.path.join(data_path,'dnf_1_thin_spiral_paper_log_grid_likelihood_theta3.npy'))
log_prob_dnf = log_prob
prob = np.exp(log_prob)
#print('max prob DNF', np.max(prob))
ax.pcolormesh(xx.numpy(), yy.numpy(), prob.reshape(n_pts,n_pts),cmap=cmap,shading='auto')
ax.set_xlim(-4.8,6)   
ax.set_ylim(-4.5,5)     
ax.set_xticks([])
ax.set_yticks([])

####################################
###########PAE######################
ax = fig.add_subplot(3,2,5) 
ax4= ax
axes += [ax4]
plt.title('PAE',y=y_title) 
log_prob = np.load(os.path.join(data_path,'pae_1_thin_spiral_paper_log_grid_likelihood.npy'))
prob = np.exp(log_prob)
prob_PAE = prob
first = ax.pcolormesh(xx.numpy(), yy.numpy(), prob.reshape(n_pts,n_pts),cmap=cmap,shading='auto')
ax.set_xlim(-4.8,6)   
ax.set_ylim(-4.5,5)     
ax.set_xticks([])
ax.set_yticks([])
#print('pae max',np.max(prob))

####################################
###########VAE######################
ax = fig.add_subplot(3,2,6) 
ax5= ax
axes += [ax5]
plt.title('VAE',y=y_title)
log_prob = np.load(os.path.join(data_path,'ELBO_grid.npy'))
prob = np.exp(log_prob)
img = ax.pcolormesh(xx.numpy(), yy.numpy(), -log_prob.reshape(n_pts,n_pts),cmap=cmap,shading='auto') #,norm=normalize)  # clim=(0,3 + 1/3)) # cmap=plt.cm.jet,shading='auto')
#whitening(prob,criterion,xx_flat,yy_flat,ax)
ax.set_xlim(-4.8,6)   
ax.set_ylim(-4.5,5)     
ax.set_xticks([])
ax.set_yticks([])
# print('vae max',np.max(prob))



# add a colorbar to the bottom of the image
cbar = fig.colorbar(first, ax=axes, orientation="horizontal", pad=0.02, ticks=[0.001, 0.6]) #0.95  ,cax=cbaxes
cbar.ax.set_xticklabels(['low', 'high'])

# fig.tight_layout() #h_pad=0.2
fig.savefig(os.path.join(output_dir, 'thin_spiral_data_densities.pdf'), bbox_inches = 'tight')