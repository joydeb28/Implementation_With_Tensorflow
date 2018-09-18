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
tf.reset_default_graph()
mnist = input_data.read_data_sets("../MNIST/",one_hot = True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_size = 200

num_input = 28
timesteps = 28
num_hidden = 256
num_classes = 10

X =  tf.placeholder('float',[None,timesteps,num_input])
Y = tf.placeholder('float',[None,num_classes])


weight = {
                'out':tf.Variable(tf.random_normal([2*num_hidden,num_classes]))
        }

bias = {
            'out' : tf.Variable(tf.random_normal([num_classes]))
        }


def bi_RNN(x,weight,bias):
    x = tf.unstack(x,timesteps,1)
    
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)
    try:
        outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    except:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    
    return tf.matmul(outputs[-1],weight['out'])+bias['out']


logits = bi_RNN(X,weight,bias)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x,batch_y =  mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, (-1, 28, 28))
        sess.run(train_op,feed_dict = {X:batch_x,Y:batch_y})
        if step ==1 or step%display_size==0:
            loss,acc =sess.run([loss_op,accuracy],feed_dict= {X:batch_x,Y:batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))









