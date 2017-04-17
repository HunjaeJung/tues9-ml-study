#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

mean = []
std = []
def normalize(x):
    row, col= x.shape
    normalized_x = np.ones([row, col], dtype=np.float32)

    for i in range(col):
        _x = x[:, i:i+1]
        x_mean = _x.mean()
        x_std = _x.std()
        normalized_x[:, i:i+1] = (_x - x_mean) / x_std
        mean.append(x_mean)
        std.append(x_std)

    return normalized_x

xy = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.float32)
x_data = xy[:, :2]
y_data = xy[:, 2:3]
x_data = normalize(x_data)

# y_data = y_data / 100000

###### Tensorflow Gradient !!!!!

def runGradient(x_data, y_data):
    print(x_data.shape, x_data, len(x_data))
    print(y_data.shape, y_data, len(y_data))

    X = tf.placeholder(tf.float32, shape=[None, 2])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    bias = tf.Variable(tf.zeros(shape=[1]))
    W = tf.Variable(tf.random_normal(shape=[2, 1]))
    W = tf.Variable(tf.zeros([2, 1]))

    hypothesis = tf.matmul(X, W) + bias
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    gradient = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = gradient.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.initialize_all_variables())

    for i in range(20001):
        cost_val, h, w_val, bias_val, _ = sess.run([cost, hypothesis, W, bias, train],
                              feed_dict={
                                  X: x_data,
                                  y: y_data
                              })

        if i % 200 is 0:
            print(cost_val, w_val, bias_val)

    print mean, std
    x_test = np.array([[1650, 3]], dtype=np.float32)
    x_test[:, 0] = (x_test[:, 0] - mean[0]) / std[0]
    x_test[:, 1] = (x_test[:, 1] - mean[1]) / std[1]

    print("score: ", sess.run([hypothesis], feed_dict={X: x_test}))

# runGradient(x_data, y_data)

### Normalized
### mean 2000.0, 3.17
### std 786.-, 0.75
### Result :
### 109444.-(1st theta), -6576.-(2nd theta), 340404.- (bias)
### Test [1650, 3] Result :
### 293074.-

####### Custom Gradient!!

def runCustomGradient(x_data, y_data):
    def customGradient(W, bias, X, y):
        row, _ = x_data.shape

        # (tf.matmul(X, W) - y) -> [47, 1] || * [47, 2] 1, 47 * 47, 2 =>
        w_gradient = W - learning_rate / row * tf.matmul(X, (tf.matmul(X, W) + bias - y), transpose_a=True)
        bias_gradient = bias - learning_rate / row * tf.reduce_mean(tf.matmul(X, W) + bias - y)
        w_update = W.assign(w_gradient)
        bias_update = bias.assign(bias_gradient)
        return w_gradient, w_update, bias_gradient, bias_update

    print(x_data.shape, x_data, len(x_data))
    print(y_data.shape, y_data, len(y_data))

    X = tf.placeholder(tf.float32, shape=[None, 2])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    bias = tf.Variable(tf.zeros(shape=[1]))
    W = tf.Variable(tf.random_normal(shape=[2, 1]))
    W = tf.Variable(tf.zeros([2, 1]))

    hypothesis = tf.matmul(X, W) + bias
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    learning_rate = 0.01

    w_gradient, w_update, bias_gradient, bias_update = customGradient(W, bias, X, y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.initialize_all_variables())
    print(sess.run([bias_gradient, bias_update, w_gradient, w_update], feed_dict={
        X: x_data,
        y: y_data
    }))

    for i in range(40001):
        cost_val, h, w_val, bias_val, _, _, _, _ = sess.run([cost, hypothesis, W, bias, bias_gradient, bias_update, w_gradient, w_update],
                              feed_dict={
                                  X: x_data,
                                  y: y_data
                              })

        if i % 200 is 0:
            print(cost_val, w_val, bias_val)

    print mean, std
    x_test = np.array([[1650, 3]], dtype=np.float32)
    x_test[:, 0] = (x_test[:, 0] - mean[0]) / std[0]
    x_test[:, 1] = (x_test[:, 1] - mean[1]) / std[1]

    print("score: ", sess.run([hypothesis], feed_dict={X: x_test}))

runCustomGradient(x_data, y_data)

### First W, bias :
### b:72.-, W:[1057.-, 547.-]
### Result :
### 109447.-(1st theta), -6578.-(2nd theta), 340339.- (bias)
### Test [1650, 3] Result :
### 293008.-

### Octave First W, bias :
### b:3404.- W:[1046.-, 541.-]

### Octave Result:
### 100087.- (1st theta), 3673.- (2nd theta), bias : 334302.063993
### Octave Test [1650, 3]:
### 289314.-

