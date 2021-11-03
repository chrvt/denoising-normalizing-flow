def infoGAN_encoder(params,is_training):

    is_training = tf.constant(is_training, dtype=tf.bool)

    def encoder(x):
        with tf.variable_scope('model/encoder',['x'], reuse=tf.AUTO_REUSE):  

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='conv1', use_sn=True))
            net = conv2d(net, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            net = batch_norm(net, is_training=is_training, scope='b_norm1')
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            net = lrelu(net)
    
            net = tf.reshape(net, [params['batch_size'], -1])
            net = linear(net, 1024, scope="ln1", use_sn=True)
            net = batch_norm(net, is_training=is_training, scope='b_norm2')
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            net = lrelu(net)

            net = linear(net, 2 * params['latent_size'], scope="ln_output", use_sn=True)
        
        return net

    return encoder


def infoGAN_decoder(params,is_training):

    is_training = tf.constant(is_training, dtype=tf.bool)

    def decoder(z):
        with tf.variable_scope('model/decoder',['z'], reuse=tf.AUTO_REUSE):
        
            net = tf.nn.relu(batch_norm(linear(z, 1024, 'ln2'), is_training=is_training, scope='b_norm3'))
            net = tf.nn.relu(batch_norm(linear(net, 128 * (params['width'] // 4) * (params['height'] // 4), scope='ln3'), is_training=is_training, scope='b_norm4'))
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            
            net = tf.reshape(net, [params['batch_size'], params['width'] // 4, params['height'] // 4, 128])
            
            net = tf.nn.relu(batch_norm(deconv2d(net, [params['batch_size'], params['width'] // 2, params['height'] // 2, 64], 4, 4, 2, 2, name='conv3'), is_training=is_training, scope='b_norm5'))
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training) 
            net = tf.nn.sigmoid(deconv2d(net, [params['batch_size'], params['width'], params['height'], params['n_channels']], 4, 4, 2, 2, name='conv4'))
            net = net-0.5
        return net

    return decoder# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:57:10 2021

@author: horvat
"""


