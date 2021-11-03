#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator, IntractableLikelihoodError
from scipy.special import i0

logger = logging.getLogger(__name__)


class VonMisesSimulator(BaseSimulator):
    def __init__(self, latent_dim=1, data_dim=2, epsilon=0.):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        self.mu = 0.0
        self.kappa = 8.0
        
        assert data_dim > latent_dim

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    def log_density(self, x): 
        raise NotImplementedError

    def sample(self, n_samples, parameters=None):          
        z = self._draw_z(n)
        x = self._transform_z_to_x(z)
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, self._data_dim))
        return x + noise

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        z = np.random.vonmises(self.mu,self.kappa,n)
        return z, 0

    def _transform_z_to_x(self, z): 
        raise NotImplementedError

    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _log_density(self, z):
        log_probs= (self.kappa*np.cos(z-self.mu))- np.log(2*np.pi*i0(self.kappa)) 
        return log_probs
    
    def generate_grid(self,n):
        import torch
        x = torch.linspace(-4.8,6, n)
        y = torch.linspace(-4.5,5, n)
        xx, yy = torch.meshgrid((x, y))
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)#.numpy()
        return grid
        


