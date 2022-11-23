# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:37:31 2021

@author: horvat
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import sys
import pickle
from functools import partial
 
plt.rcParams.update({'font.size': 16,                                                                                                                                                    
                     'axes.labelsize': 16, 'legend.fontsize': 12, 
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 16,
                     'axes.linewidth': 1.5}) 

import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors

from pae.model_tf2 import get_prior, get_posterior, get_likelihood
import pae.create_datasets as crd
param_file = 'params_gan2_-1_2_infoGAN_AE_default1_full_sigma' 
PROJECT_PATH = r"/storage/homefs/ch19g182/Python/PAE-master2" 
PARAMS_PATH = os.path.join(PROJECT_PATH,'params')
params   = pickle.load(open(os.path.join(PARAMS_PATH,param_file+r'.pkl'),'rb'))

params['plot_dir'] = r"plots" 
if not os.path.isdir(params['plot_dir']):
    os.makedirs(params['plot_dir'])
if not os.path.isdir(params['data_dir']):
    os.makedirs(params['data_dir'])
print('module_dir',params['module_dir'])

generator_path   = os.path.join(params['module_dir'],'decoder')
encoder_path     = os.path.join(params['module_dir'],'encoder')

train_input_fn, eval_input_fn = crd.build_input_fns(params,label=params['class_label'],flatten=False,num_repeat=15)

tf.reset_default_graph()



traindata     = train_input_fn()
validdata     = eval_input_fn()
encoder       = hub.Module(encoder_path, trainable=False)
generator     = hub.Module(generator_path, trainable=False)


posterior     = get_posterior(encoder)
likelihood    = get_likelihood(generator,params)

print('path',encoder_path)


# inspect the model
zt    = posterior(traindata).mean()
recont = likelihood(zt).mean()

zv    = posterior(validdata).mean()
reconv = likelihood(zv).mean()

sess = tf.Session()

if params['data_set']=='gan2':
    train_size = 9000
    val_size   = 1000
elif params['data_set']=='gan64':
    train_size = 18000
    val_size   = 2000
else:
    train_size = 50000
    val_size   = 10000


print(train_size//params['batch_size']+1)
enc_path = os.path.join('encoded/',params['data_set'],str(params['latent_size']))
print(os.path.join(params['data_dir'],enc_path))
if not os.path.isdir(os.path.join(params['data_dir'],enc_path)):
    os.makedirs(os.path.join(params['data_dir'],enc_path))
    
try:
    z_sample = np.load(os.path.join(params['data_dir'],enc_path,'train.npy'))
except:
    z_sample=[]
    for ii in range(train_size//params['batch_size']+1):
        if ii%100==0:
            print(ii)
        z_sample+=[sess.run(zt)]
        
    z_sample = np.asarray(z_sample).reshape((-1,params['latent_size']))
    z_sample = z_sample[0:train_size]
    
    np.save(os.path.join(params['data_dir'],enc_path,'train.npy'),z_sample)
    
try:
    z_sample_valid = np.load(os.path.join(params['data_dir'],enc_path,'valid.npy'))
except:
    z_sample_valid=[]

    for ii in range(val_size//params['batch_size']+1):
        if ii%100==0:
            print(ii)
        z_sample_valid+=[sess.run(zv)]
    z_sample_valid=np.asarray(z_sample_valid).reshape((-1,params['latent_size']))
    z_sample_valid= z_sample_valid[0:val_size]
    np.save(os.path.join(params['data_dir'],enc_path,'valid.npy'),z_sample_valid)
    
nvp_depth    = 20
dims         = np.asarray(20*[params['latent_size']])
nvp_size     = np.arange(nvp_depth)
indices      = np.arange(params['latent_size'])

def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
        event_size = tf.convert_to_tensor(
            event_size, dtype_hint=tf.int32, name='event_size')
        batch_shape = tf.convert_to_tensor(
            batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
        random_matrix = tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=dtype,
            seed=seed)
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper,
            trainable=True,
            name='lower_upper')
        # Initialize a non-trainable variable for the permutation indices so
        # that its value isn't re-sampled from run-to-run.
        permutation = tf.Variable(
            initial_value=permutation,
            trainable=False,
            name='permutation')
        return lower_upper, permutation

class SplineParams(tf.Module):

    def __init__(self, nunits, nbins=32):
        self._nbins = nbins
        self._nunits= nunits
        self._built = False
        self._bin_widths = None
        self._bin_heights = None
        self._knot_slopes = None

    def _bin_positions(self, x):
        x = tf.reshape(x, [-1,self._nunits,self._nbins])
        net = tf.layers.Dense(self._nunits)(x)
        net = tf.nn.leaky_relu(net)
        net = tf.layers.Dense(self._nunits)(net)
        net = tf.nn.leaky_relu(net)
        net = tf.layers.Dense(self._nbins)(net)
        net = tf.reshape(net, [-1,self._nunits, self._nbins])
        return tf.math.softmax(net, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

    def _slopes(self, x):
        x = tf.reshape(x, [-1,  self._nunits, ( self._nbins - 1)])
#         net = tf.layers.Dense(self._nunits)(x)
#         net = tf.nn.leaky_relu(net)
#         net = tf.layers.Dense(self._nbins-1)(net)
#         net = tf.reshape(net, [-1,  self._nunits, ( self._nbins - 1)])
        return tf.math.softplus(x) + 1e-2

    def __call__(self, x, nunits):

        bin_positions = self._bin_positions
        if not self._built:
            self._bin_widths  = tf.layers.Dense(self._nunits * self._nbins, activation=bin_positions, name='w')
            self._bin_heights = tf.layers.Dense(self._nunits * self._nbins, activation=bin_positions, name='h')
            self._knot_slopes = tf.layers.Dense(self._nunits * (self._nbins - 1), activation=self._slopes, name='s')
        self._built = True
        result = tfb.RationalQuadraticSpline(bin_widths=self._bin_widths(x),bin_heights=self._bin_heights(x),knot_slopes=self._knot_slopes(x))
        
        return result
    
tf.reset_default_graph()

lr            = tf.placeholder_with_default(0.001,shape=[])
data          = tf.placeholder(shape=[None,params['latent_size']],dtype=tf.float32)

optimizer     = tf.train.AdamOptimizer(learning_rate=lr)

def init_once(x, name):
    return tf.get_variable(name, initializer=x, trainable=False)


#nvp_size      = [[params['latent_size'],params['latent_size']] for ii in range(nvp_depth)]
#[64,64,64,64,32,32,8,8])#

def get_nvp():
    base_dis          = get_prior(params['latent_size'])
    chain             = []
    reduction         = (params['latent_size']-dims)/params['latent_size']#np.repeat(np.arange(nvp_depth//2)/(nvp_depth//2),2)# np.repeat([0],nvp_depth)#
    current_size      = []
    perms_swap        = []
    perms_rand        = []
    perms_train       = []
    splines           = []
    size              = []
    #current_size      = params['latent_size']
    for i,s in enumerate(nvp_size):
        current_size  = np.int(params['latent_size']*(1-reduction[i]))
        swapping      = np.concatenate((np.arange(current_size//2,current_size),np.arange(0, current_size//2)))
        perms_swap.append(tfb.Permute(permutation=init_once(swapping,name="perm_swap%d"%i)))
        permutation   = np.random.permutation(np.arange(0, current_size))
        perms_rand.append(tfb.Permute(permutation=init_once(permutation,name="perm_rand%d"%i)))
        perms_train.append(tfb.MatvecLU(*trainable_lu_factorization(current_size),validate_args=True, name="perms_train%d"%i))
        splines.append(SplineParams(current_size//2))
        size.append(current_size)
        print(current_size, size)
    
    chain.append(tfb.Scale(10))
    for i,s in enumerate(nvp_size):
        if i%2==0:
            chain.append(tfb.Blockwise(bijectors=[perms_train[i],tfb.Identity()],block_sizes=[size[i],params['latent_size']-size[i]]))
        else:
            chain.append(tfb.Blockwise(bijectors=[perms_swap[i],tfb.Identity()],block_sizes=[size[i],params['latent_size']-size[i]]))
        if i<4: 
            bijector1 = tfb.RealNVP(num_masked=size[i]-size[i]//2,bijector_fn=splines[i])
        if i<12 and i>4:
            bijector1 = tfb.RealNVP(num_masked=size[i]-size[i]//2,shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[size[i],size[i]]))                 
        else:
            bijector1 = tfb.RealNVP(num_masked=size[i]-size[i]//2,shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[size[i],size[i]],shift_only=True))  
        bijector2 = tfb.Identity()
        chain.append(tfb.Blockwise(bijectors=[bijector1, bijector2], block_sizes=[size[i],params['latent_size']-size[i]]))
        
    #chain.append(tfb.Scale(0.1))
    
    # p(x) is Gaussian, chain: y=g(x)
    nvp=tfd.TransformedDistribution(distribution=tfd.Normal(0,1),bijector=tfb.Chain(chain),name='mynvp', event_shape=(params['latent_size'],))
                 
    return nvp

def nvp_module_spec():
    z_sample     = tf.placeholder(tf.float32, shape=[None,params['latent_size']])
    sample_size  = tf.placeholder(tf.int32, shape=[])
    u_sample     = tf.placeholder(tf.float32, shape=[None,params['latent_size']])
    nvp_         = get_nvp()
    log_prob     = nvp_.log_prob(z_sample)
    nvp_sample   = nvp_.sample(sample_size)
    nvp_fwd_pass = nvp_.bijector.forward(u_sample)
    nvp_bwd_pass = nvp_.bijector.inverse(z_sample)
    hub.add_signature(inputs={'z_sample':z_sample,'sample_size':sample_size, 'u_sample':u_sample}\
                      ,outputs={'log_prob':log_prob, 'sample':nvp_sample, 'fwd_pass': nvp_fwd_pass, 'bwd_pass': nvp_bwd_pass})

nvp_spec  = hub.create_module_spec(nvp_module_spec)
#nvp_funcs = hub.Module(nvp_path, trainable=True)# if already created once
nvp_funcs = hub.Module(nvp_spec, name='nvp_funcs',trainable=True)
loss      = -tf.reduce_mean(nvp_funcs({'z_sample':data,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))},as_dict=True)['log_prob'])

opt_op_nvp= optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

nvp_tloss = []
nvp_vloss = []
ii     = 0

learning_rate = 1e-4
batchsize = 64
epoch = train_size//batchsize

n_epochs = 160
while ii<n_epochs:
    print('epoch ', ii)
    jj = 0
#     if ii>1:
#         learning_rate = 1e-3
#         batchsize = 64
#     if ii>50:
#         batchsize = 128
#     if ii>60:
#         learning_rate = 1e-4
#     if ii>90:
#         batchsize=64
    if ii>90:
        batchsize=128
#     if ii>110:
#         learning_rate = 1e-3
#     if ii>120:
#         learning_rate = 1e-4
    epoch = train_size//batchsize
    while jj<epoch:
        noise = np.random.normal(size=(batchsize,params['latent_size']))*0.#2e-2
        _, ll = sess.run([opt_op_nvp,loss],  feed_dict={lr: learning_rate, data:z_sample[jj*batchsize:(jj+1)*batchsize]+noise})
        if jj%200==0:
            np.random.shuffle(z_sample_valid)
            l_v = sess.run(loss,  feed_dict={lr: learning_rate,lr: learning_rate,data:z_sample_valid})
            nvp_vloss+=[l_v]
            inds = np.random.randint(low=0, high=train_size, size=1024)
            l_t = sess.run(loss,  feed_dict={lr: learning_rate,lr: learning_rate,data:z_sample[inds]})
            nvp_tloss+=[l_t]
            print(l_t, l_v)
#             sample = sess.run(fwd_pass)
#             print(np.amax(sample))
#         if jj==0:
#             nn = np.random.randint(low=0,high=63)
#             plt.scatter(z_sample_valid[:500,nn],z_sample_valid[:500,nn+1],s=1, label='encoded data valid')
#             plt.scatter(z_sample[:500,nn],z_sample[:500,nn+1],s=2, label='encoded data') 
#             plt.scatter(latent_samples[:500,nn],latent_samples[:500,nn+1],s=2, label='RealNVP samples') 
#             plt.show()
        jj+=1
    if ii>80:
        nvp_path         = os.path.join(params['module_dir'],'hybrid4_nepoch%d'%ii)
        nvp_funcs.export(nvp_path,sess)
        
    np.random.shuffle(z_sample)
    ii+=1

plt.figure()
plt.plot(nvp_tloss,label='training loss')
plt.plot(nvp_vloss,label='validation loss' )
plt.xlabel('# iteration')
plt.ylabel('RealNVP loss')
plt.ylim(-61,30)
plt.legend()
plt.savefig(os.path.join(params['plot_dir'],'RealNVP_loss.pdf'),bbox_inches='tight')

sess.run(loss,  feed_dict={lr: learning_rate,lr: learning_rate,data:z_sample_valid})

tf.reset_default_graph()

lr            = tf.placeholder_with_default(0.001,shape=[])
sigma         = tf.placeholder_with_default(0.1,shape=[])
sample_size   = tf.placeholder_with_default(params['batch_size'],shape=[])
data          = train_input_fn()
validdata     = eval_input_fn()
encoder       = hub.Module(encoder_path, trainable=False)
generator     = hub.Module(generator_path, trainable=False)
nvp_funcs     = hub.Module(nvp_path, trainable=False)
optimizer     = tf.train.AdamOptimizer(learning_rate=lr)

prior         = get_prior(params['latent_size'])
posterior     = get_posterior(encoder)
likelihood    = get_likelihood(generator,params)

# inspect the model
z_      = posterior(validdata).mean()
recon   = likelihood(z_).mean()
prior_sample = prior.sample(sample_size)
x_old   = likelihood(prior_sample).mean()


nvp_sample = nvp_funcs({'z_sample':np.zeros((1,params['latent_size'])),'sample_size':1, 'u_sample':prior_sample}, as_dict=True)['fwd_pass']
x_new      = likelihood(nvp_sample).mean()
 
# # ---------------------------end train nvp ----------------- #

# #comment train nvp above section above and uncomment this for loading the trained module

# #

"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

rec, d = sess.run([recon,validdata])
x_sample = sess.run(x_old)

shape=params['data_shape']

#tf.compat.v1.enable_eager_execution()
x_samples = []
for k in range(63):
    prior_sample = prior.sample(16)
    nvp_sample = nvp_funcs({'z_sample':np.zeros((1,params['latent_size'])),'sample_size':1, 'u_sample':prior_sample}, as_dict=True)['fwd_pass']
    x_new      = likelihood(nvp_sample).mean()
    x_samples += [x_new.reshape(shape)]
x_samples = tf.concat(x_samples, axis=0, name='concate')
x_samples = x_samples.eval(session=tf.compat.v1.Session())  #convert into numpy
x_samples = x_samples[:1000,:]#.reshape([1000,3,64,64])
np.save(os.path.join(params['plot_dir'],'model_samples64.npy'),x_samples)
"""