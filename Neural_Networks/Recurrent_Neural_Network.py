#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:06:38 2018

@author: joy
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST/",one_hot = True)

learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_size = 200

num_input = 128
timesteps = 128
num_hidden = 128
num_classes = 10

x = tf.placeholder("float",[None,timesteps,num_input])
y = tf.placeholder("float",[None,num_classes])

weights = {
        "out" : tf.Variable(tf.random_normal([num_hidden,num_classes]))
        }

biases = {
        "out" : tf.Variable(tf.random_normal([num_classes]))
        }


def RNN(x,weights,biases):
    x =tf.unstack(x,timesteps,1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1)
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.mat_mul(outputs[-1],weights["out"])+biases["out"]


logits = RNN(x)










































