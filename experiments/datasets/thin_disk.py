#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator


logger = logging.getLogger(__name__)


class ThinDiskSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, phases=0.5 * np.pi, widths=0.25 * np.pi, epsilon=0.):
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
        z = self._transform_x_to_z(x)
        x_hat = _transform_z_to_x(z)
        if np.linalg.norm(x_hat-x) <= 2**(-8):
            logp = self._log_density(z)
        else: logp = - np.inf
        return logp

    def sample(self, n, parameters=None):
        z, r = self._draw_z(n)
        x = self._transform_z_to_x(z,r)
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def distance_from_manifold(self, x):
        z_phi, z_eps = self._transform_x_to_z(x)
        return np.sum(z_eps ** 2, axis=1) ** 0.5

    def _draw_z(self, n):
        numbs = np.random.exponential(scale=0.7,size=n)#(np.random.normal(0,1,1000))**2
        z = np.sqrt(numbs) * 540 * (2 * np.pi) / 360
        r = np.random.uniform(-0.1,0.1,1000)
        return z, r

    def _transform_z_to_x(self, z,r):
        dr = r
        d1x = - np.cos(z) * (z+dr) 
        d1y =   np.sin(z) * (z+dr) 
        d1z = np.zeros(z.shape)
        x = np.stack([ d1x,  d1y, d1z], axis=1) / 3
        return x

    def _transform_x_to_z(self, x):
        z = np.linalg.norm(x,axis=1)
        return z

    def _log_density(self, z):
        return np.zeros([z.shape[0],1])
    
    def generate_grid(self,n):
        import torch
        x = torch.linspace(-4.8,6, n)
        y = torch.linspace(-4.5,5, n)
        xx, yy = torch.meshgrid((x, y))
        grid = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(yy.flatten().shape)), dim=1).numpy()
        return grid
        


