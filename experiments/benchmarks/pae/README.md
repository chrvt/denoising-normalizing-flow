# Probabilistic Auto Encoder (PAE)

The PAE turns a usual autoencoder (AE) into a generative model by learning the AE's induced latent density with a Normalizing Flow. Thus, first a AE is trained on the reconstruction error, and then a NF is used to learn the latent density.

### Training and Evaluation
For training on the gan2d and gan64d image manifolds, we used the original [Probabilistic Auto-Encoder](https://github.com/VMBoehm/PAE) implementation by V. Böhm and U. Seljak. For that, we added gan2d and gan64d to the set of possible datasets (see [experiments/benchmarks/pae/modified](experiments/benchmarks/pae/modified)). 
Originally, training the NF and evaluating the model is done in jupyter notebooks. For convenience, we translate the training notebook into python code, see [experiments/benchmarks/pae](experiments/benchmarks/pae). Our adapted notebooks (e.g. for generating the image grid) can be also found in [experiments/benchmarks/pae](experiments/benchmarks/pae).
