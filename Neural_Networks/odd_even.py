import numpy as np
import tensorflow as tf

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def encode(i):
    if   i % 2 == 0: 
        return np.array([0, 1])
   
    else:             
        return np.array([1, 0])
    

NUM_DIGITS = 10
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([encode(i) for i in range(101, 2 ** NUM_DIGITS)])

NUM_HIDDEN = 1000

X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 2])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 2])

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = py_x,labels = Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

saver = tf.train.Saver()

predict_op = tf.argmax(py_x, 1)

def predict(i, prediction):
    return ["odd", "even"][prediction]


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(1000):
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]
    
    BATCH_SIZE = 20
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: trX, Y: trY})))
        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(predict)(numbers, teY)
        #save_path = saver.save(sess, "odd_even_models/model")
        #print("Model saved in path: %s" % save_path)
        print(output)
