#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:48:16 2018

@author: joy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:06:38 2018

@author: joy
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("../MNIST/",one_hot = True)

learning_rate = 0.0001
training_steps = 10000
batch_size = 128
display_size = 200

num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

X =  tf.placeholder('float',[None,timesteps,num_input])
Y = tf.placeholder('float',[None,num_classes])


weight = {
                'out':tf.Variable(tf.random_normal([2*num_hidden,num_classes]))
        }

bias = {
            'out' : tf.Variable(tf.random_normal([num_classes]))
        }













