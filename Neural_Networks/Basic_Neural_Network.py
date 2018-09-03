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
num_steps = 5000
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


bias = {
            'b1':tf.Variable(tf.random_normal([num_hidden_1])),
            'b2':tf.Variable(tf.random_normal([num_hidden_2])),
            'out':tf.Variable(tf.random_normal([num_cls])),
        }




def neural_net(x):
    
    layer1 = tf.add(tf.matmul(x,weights['h1']),bias['b1'])
    
    layer2 = tf.add(tf.matmul(layer1,weights['h2']),bias['b2'])
    
    output_layer = tf.add(tf.matmul(layer2,weights['out']),bias['out'])
    
    return output_layer

logits = neural_net(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict = {x : batch_x, y:batch_y})
        if num_steps%step ==0 :
            loss,acc= sess.run([loss_op,accuracy], feed_dict = {x : batch_x, y:batch_y})
    
            print("Step " + str(step) + ", Minibatch Loss= " +"{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

























