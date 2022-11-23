# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:45:27 2021

@author: horvat
"""
import sys

sys.path.append(r"C:\Users\Horvat\Anaconda3\pkgs\pytorch-fid-master\src\pytorch_fid")
from fid_score import calculate_fid_given_paths
import tempfile
#gan2
x_true = np.load(r'D:/manifold-flow-public/experiments/data/samples/gan2d/test.npy')
x_gen = np.load(r'D:/manifold-flow-public/experiments/data/results/PAE_model_samples.npy')

def array_to_image_folder(data, folder):
    for i, x in enumerate(data):
        print('x shape',x.shape)
        print('x...',x[0:10,:])
        x = np.clip(np.transpose(x, [1, 2, 0]) / 256.0, 0.0, 1.0)
        if i == 0:
            logger.debug("x: %s", x)
        plt.imsave(f"{folder}/{i}.jpg", x)
        
with tempfile.TemporaryDirectory() as gen_dir:
    array_to_image_folder(x_gen, gen_dir)
    true_dir = r'D:\manifold-flow-public\experiments\data\samples\gan2d\test'
    
    fid = calculate_fid_given_paths([gen_dir, true_dir], 50, "cuda", 2048)

np.save(r'D:\manifold-flow-public\experiments\data\results\PAE_fid.npy',fid)
