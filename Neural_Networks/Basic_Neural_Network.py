#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:58:45 2018

@author: joy
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST/',one_hot = True)

learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

num_hidden_1 = 256
num_hidden_2 = 256
num_input = 784
num_cls = 10 

x = tf.placeholder('float',[None,num_input])
y = tf.placeholder('float',[None,num_cls])

weights = {
        'h1': tf.Variable(tf.random_normal([num_input,num_hidden_1])),
        'h2': tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_hidden_2,num_cls]))
        }

Bias = {
                'b1':tf.Variable(tf.random_normal([num_hidden_1])),
                'b2':tf.Variable(tf.random_normal([num_hidden_2])),
                'out':tf.Variable(tf.random_normal([num_cls])),
        }




































