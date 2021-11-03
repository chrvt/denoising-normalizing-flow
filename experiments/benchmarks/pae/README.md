# Probabilistic Auto Encoder (PAE)

The PAE turns a usual autoencoder (AE) into a generative model by learning the AE's induced latent density with a Normalizing Flow. Thus, first a AE is trained on the reconstruction error, and then a NF is used to learn the latent density.

### Training and Evaluation
For training on the gan2d and gan64d image manifolds, we used the original [Probabilistic Auto-Encoder](https://github.com/VMBoehm/PAE) implementation by V. Böhm and U. Seljak. For that, we added gan2d and gan64d to the set of possible datasets (see [experiments/benchmarks/pae/modified](experiments/benchmarks/pae/modified)). 
Originally, training the NF and evaluating the model is done in jupyter notebooks. For convenience, we translate the training notebook into python code, see [experiments/benchmarks/pae](experiments/benchmarks/pae). Our adapted notebooks (e.g. for generating the image grid) can be also found in [experiments/benchmarks/pae](experiments/benchmarks/pae). Therefore, to reproduce our results, the following steps need to be conducted:
  1. Install [Probabilistic Auto-Encoder](https://github.com/VMBoehm/PAE)
  2. Replace "create_datasets.py", "load_data.py" and "main.py" with files in [experiments/benchmarks/pae/modified]
  3. Train models 
  4. Train NFs using "train_NSF_gan2.py" and "train_NSF_gan64.py", respectively
  5. Open notebooks "FIDScore_and_Reconstruction_Error-gan64.ipynb" and "FIDScore_and_Reconstruction_Error-gan2.ipynb", respectively (on cluster: execute jupyter.sh)
  6. Calculate FID score locally using saved model samples  
