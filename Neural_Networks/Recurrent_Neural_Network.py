#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:06:38 2018

@author: joy
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
mnist = input_data.read_data_sets("../MNIST/",one_hot = True)

learning_rate = 0.0001
training_steps = 1000
batch_size = 128
display_size = 200

num_input = 28
timesteps = 28
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
    return tf.matmul(outputs[-1],weights["out"])+biases["out"]


logits = RNN(x,weights,biases)
pred = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size,timesteps,num_input))
        sess.run(train_op,feed_dict = {x : batch_x,y:batch_y})
        if step%display_size==0 or step==1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict = {x:batch_x,y:batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: test_data, y: test_label}))     
            
    







































