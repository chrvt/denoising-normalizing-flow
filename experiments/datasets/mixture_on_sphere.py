#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0

logger = logging.getLogger(__name__)


class MixtureSphereSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, kappa=6.0, epsilon=0.):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon

        self.kappa = 6.0
        self.mu11 = 1*np.pi/4       
        self.mu12 = np.pi/2             #northpole
        self.mu21 = 3*np.pi/4 
        self.mu22 = 4*np.pi/3           #southpole
        self.mu31 = 3*np.pi/4 
        self.mu32 = np.pi/2 
        self.mu41 = np.pi/4
        self.mu42 = 4*np.pi/3 

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
        theta, phi = self._draw_z(n)
        x = self._transform_z_to_x(theta,phi)
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        n_samples = np.random.multinomial(n,[1/4]*4,size=1)
        n_samples_1 = n_samples[0,0]
        n_samples_2 = n_samples[0,1]
        n_samples_3 = n_samples[0,2]
        n_samples_4 = n_samples[0,3]
        
        theta1 = np.random.vonmises((self.mu11-np.pi/2)*2,self.kappa,n_samples_1 )/2 + np.pi/2
        phi1 = np.random.vonmises(self.mu12-np.pi,self.kappa,n_samples_1 ) + np.pi
        theta2 = np.random.vonmises((self.mu21-np.pi/2)*2,self.kappa,n_samples_2 )/2 + np.pi/2
        phi2 = np.random.vonmises(self.mu22-np.pi,self.kappa,n_samples_2 ) + np.pi
        theta3 = np.random.vonmises((self.mu31-np.pi/2)*2,self.kappa,n_samples_3 )/2 + np.pi/2
        phi3 = np.random.vonmises(self.mu32-np.pi,self.kappa,n_samples_3 ) + np.pi
        theta4 = np.random.vonmises((self.mu41-np.pi/2)*2,self.kappa,n_samples_4 )/2 + np.pi/2
        phi4 = np.random.vonmises(self.mu42-np.pi,self.kappa,n_samples_4 ) + np.pi
        
        theta = np.concatenate([theta1,theta2,theta3,theta4],axis=0)
        phi = np.concatenate([phi1,phi2,phi3,phi4],axis=0)     
        
        return theta, phi
    
    
    def _transform_z_to_x(self, theta,phi, mode='train'):
        c, a = 0, 1
        d1x = (c + a*np.sin(theta)) * np.cos(phi)
        d1y = (c + a*np.sin(theta)) * np.sin(phi)
        d1z = (a * np.cos(theta))
        x = np.stack([ d1x, d1y, d1z], axis=1) 
        params = np.ones(x.shape[0])
        if mode=='train':
            x = NumpyDataset(x, params)
        return x

    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _density(self, data):
        theta = data[0]
        phi = data[1]
        probs = 1/4* (2*np.exp(self.kappa*np.cos(2* (theta-self.mu31))) * np.exp(self.kappa*np.cos(phi-self.mu32)) *(1/(2*np.pi*i0(self.kappa))**2)
             +2*np.exp(self.kappa*np.cos(2* (theta-self.mu11))) * np.exp(self.kappa*np.cos(phi-self.mu12)) *(1/(2*np.pi*i0(self.kappa))**2)   
             +2*np.exp(self.kappa*np.cos(2* (theta-self.mu21))) * np.exp(self.kappa*np.cos(phi-self.mu22)) *(1/(2*np.pi*i0(self.kappa))**2)  
             +2*np.exp(self.kappa*np.cos(2* (theta-self.mu41))) * np.exp(self.kappa*np.cos(phi-self.mu42)) *(1/(2*np.pi*i0(self.kappa))**2)
             )
        return probs
    
    def generate_grid(self,n,mode='sphere'):
        theta = np.linspace(0, np.pi, n+2)
        phi = np.linspace(0, 2*np.pi, n+1)
        xx, yy = np.meshgrid(theta[1:n+1], phi[1:n+1])
        if mode == 'sphere':            
            grid = np.stack((xx.flatten(), yy.flatten()), axis=1)#.numpy()
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        
        return data #torch.tensor(data)
        


