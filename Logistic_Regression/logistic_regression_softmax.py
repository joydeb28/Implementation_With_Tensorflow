import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST/',one_hot = True)

learning_rate =0.01
batch_size = 100
epochs = 100000
display = 1

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,w)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _,c = sess.run([optimizer,cost],feed_dict = {x:batch_x,y:batch_y})
        avg_cost+= c/total_batch
        
        if (epoch+1)%display ==0:
            print("Epoch: ",epoch,"Cost: ",avg_cost)
            
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("Accuracy: ",accuracy.eval({x:mnist.test.images[:3000],y:mnist.test.labels[:3000]}))
            
        
    