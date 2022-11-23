import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_path = r'...\results'
xtrain_path = r'...\datasets\2D_style_gan\train.npy')
xtest_path = r'...\datasets\2D_style_gan\test.npy')
xtest_latent_path = r'...\datasets\2D_style_gan\test.npy')

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 2)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(2, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.kl_batch = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu =  self.linear3(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()        
        self.kl_batch = torch.sum(sigma**2 + mu**2 - torch.log(sigma) - 1/2,dim=1)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data,sched, opt, epochs=200):  
    for epoch in range(epochs):
        print('start training witch epoch ',epoch)
        for x in data:
            x = x.to(device).float()
            opt.zero_grad()
            x_hat = autoencoder(x)
            rec = ((x - x_hat)**2).sum()
            loss = rec + autoencoder.encoder.kl
            loss.backward()
            opt.step()
        sched.step()
        print('loss' ,rec)
    return autoencoder


class UnlabelledImageDataset(Dataset):
    def __init__(self, array, transform=None):
        self.transform = transform
        self.data = self.transform(array)

    def __getitem__(self, index):
        img = self.data[index, ...]#.clone() 
        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]


with torch.no_grad():
    def ELBO_grid(n,vae):
        x = torch.linspace(-4.8,6, n)
        y = torch.linspace(-4.5,5, n)
        xx, yy = torch.meshgrid((x, y))
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        grid_loader = DataLoader(
                    grid,
                    batch_size=25,
                    num_workers=0,
                )
        # Evaluate
        ELBO = []
        #n_batches = (args.evaluate**2 - 1) // args.evalbatchsize**2 + 1
        #for j in range(n_batches):
        for x in grid_loader:
            x = x.to(device)
            x_hat = vae(x)
            elbo = torch.sum((x - x_hat)**2,dim=1) + vae.encoder.kl_batch
            ELBO.append(elbo.cpu().detach().numpy())       
        ELBO = np.concatenate(ELBO,axis=0)
        return ELBO

with torch.no_grad():
    def ELBO_latent(vae):
        latent_test = np.load(xtest_latent_path)
        order = np.argsort(latent_test)
        latent_test = latent_test[order] #sort: lowest to highest
        z = np.sqrt(latent_test) * 540 * (2 * np.pi) / 360 
        d1x = - np.cos(z) * z     #d/dz = -cos(z) +sin(z)z    --> ||grad||^2 = cos^2 - cos sin z + sin^2 z^2 +
        d1y =   np.sin(z) * z     #d/dz =  sin(z) +cos(z)z   --->             sin^2 + cos sin z + cos^2 z^2
        x = np.stack([ d1x,  d1y], axis=1) / 3  #        
        x = torch.tensor(x).to(torch.float) 
        
        x_loader = DataLoader(
                    x,
                    batch_size=25,
                    num_workers=0,
                )
        # Evaluate
        ELBO = []
        #n_batches = (args.evaluate**2 - 1) // args.evalbatchsize**2 + 1
        #for j in range(n_batches):
        for x_ in x_loader:
            x_ = x_.to(device)
            x_hat = vae(x_)
            elbo = torch.sum((x_ - x_hat)**2,dim=1) + vae.encoder.kl_batch
            ELBO.append(elbo.cpu().detach().numpy())       
        ELBO = np.concatenate(ELBO,axis=0)
        return ELBO


spiral = torch.from_numpy(np.load(xtrain_path)) 
# spiral = spiral + 0.1 * torch.rand(spiral.shape).float()
#np.load(r'D:\InfoMaxVAE-master\data\gan2d\train.npy') #
#spiral = spiral[1:100,:]#torch.from_numpy() 

#transform = transforms.Compose([
#    transforms.ToTensor(),
#])

#dataset = UnlabelledImageDataset(spiral, transform=transform)

"""
array_to_image_folder(style, transform, train_folder)


transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = ImageFolder('./images/folder/', transform)
"""
data_loader = DataLoader(dataset=spiral, batch_size=200, shuffle=True, num_workers=0) #, drop_last=True)

latent_dims = 1
vae = VariationalAutoencoder(latent_dims).to(device) # GPU

print('VAE parameters: ', sum(p.numel() for p in vae.parameters()))

vae.train() 
opt = torch.optim.AdamW(vae.parameters())
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 100)
vae = train(vae, data_loader,sched,opt)

vae.eval()

ELBO = ELBO_grid(100,vae)
latent_probs = ELBO_latent(vae)

np.save(os.path.join(xtest_latent_path,'VAE_latent_probs.npy'),latent_probs)
np.save(os.path.join(xtest_latent_path,'ELBO_grid.npy'),ELBO)
