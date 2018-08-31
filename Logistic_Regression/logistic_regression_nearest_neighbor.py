#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:51:22 2018

@author: joy
"""

import tensorflow as tf
import numpy as np
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


distance = tf.reduce_sum(tf.abs(tf.add(tr,tf.negative(ts))),reduction_indices=1)

pred = tf.arg_min(distance,0)

accuracy =0
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(x_test)):
        nn_index = sess.run(pred,feed_dict={tr : x_train,ts : x_test[i,:]})
        print("Test: ",i,", Prediction Class: ",np.argmax(y_train[nn_index]),", True Class: ",np.argmax(y_test[i]))
        if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
            accuracy+=1/len(x_test)
        
    print("Accuarcy: ",accuracy)





