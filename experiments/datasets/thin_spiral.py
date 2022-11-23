#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator


logger = logging.getLogger(__name__)


class ThinSpiralSimulator(BaseSimulator):
    def __init__(self, latent_dim=1, data_dim=2, epsilon=0.):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon

        assert data_dim > latent_dim

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    def log_density(self, x, parameters=None, precise=False):
        raise NotImplementedError

    def sample(self, n, parameters=None):
        z,_ = self._draw_z(n)
        x = self._transform_z_to_x(z)
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        numbs = np.random.exponential(scale=0.3,size=n)#(np.random.normal(0,1,n))**2
        z = np.sqrt(numbs) * 540 * (2 * np.pi) / 360
        return z, 0

    def _transform_z_to_x(self, z):
        d1x = - np.cos(z) * z 
        d1y =   np.sin(z) * z 
        x = np.stack([ d1x,  d1y], axis=1) / 3
        return x

    def _transform_x_to_z(self, x):
        raise NotImplementedError
        
    def _log_density(self, z):
        return np.zeros([z.shape[0],1])
    
    def generate_grid(self,n):
        import torch
        x = torch.linspace(-4.8,6, n)
        y = torch.linspace(-4.5,5, n)
        xx, yy = torch.meshgrid((x, y))
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)#.numpy()
        return grid
        
    def load_latent(self, train, dataset_dir, latent=True, paramscan=False, run=0):
        tag = "train" if train else "test"
        latents = np.load("{}/x_{}_latent.npy".format(dataset_dir, tag))
        return latents

    def load_idx(self, dataset_dir):
        val_idx = np.load("{}/validation_index.npy".format(dataset_dir))
        return val_idx

