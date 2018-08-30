#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:51:22 2018

@author: joy
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST/',one_hot = True)

learning_rate =0.01
batch_size = 100
epochs = 100000
display = 1

x_train,y_train = mnist.train.next_batch(5000)
x_test,y_test = mnist.test.next_batch(200)

tr = tf.placeholder("float",[None,784])
ts = tf.placeholder("float",[784])











