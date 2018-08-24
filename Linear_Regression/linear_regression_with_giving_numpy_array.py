import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random #random number generator

learning_rate = 0.01
epochs = 10000
display = 50

train_x = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

nsample = train_x.shape[0]

x = tf.placeholder('float')
y = tf.placeholder('float')

w = tf.Variable(rng.random())
b = tf.Variable(rng.random())

pred = tf.add(tf.multiply(x,w),b)

cost = tf.reduce_sum(tf.pow(pred-y,2))/(2*nsample)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        for (x_val,y_val) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict = {x : x_val,y: y_val})
        
        if (epoch+1)%display == 0:
            c = sess.run(cost,feed_dict = {x:train_x, y:train_y})
            print("Epoch: ",epoch,"Cost: ",c,"Weight: ",sess.run(w),"Bias: ",sess.run(b))
    
    
    training_cost = sess.run(cost,feed_dict = {x:train_x,y:train_y})
    print("Cost: ",training_cost,"Weight: ",sess.run(w),"Bias: ",sess.run(b))
    
    plt.plot(train_x,train_y,'ro',label = "original")
    plt.plot(train_x,sess.run(w)*train_x+sess.run(b),label = "trained")
    plt.legend()
    plt.show()
            
