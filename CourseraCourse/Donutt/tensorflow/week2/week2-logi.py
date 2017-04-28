#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')

# X = tf.placeholder(tf.float32, shape=[None, 2])
# W = tf.Variable(tf.random_normal([2, 1]), name='weight')

y = tf.placeholder(tf.float32, shape=[None, 1])

b = tf.Variable(tf.random_normal([1]), name='bias')

# sig = 1 / (1 + tf.exp(-(tf.matmul(X,W)+b))) -> 틀렸을듯
sig = tf.div(1., 1. + tf.exp(-(tf.matmul(X,W)+b)))
# sig = tf.sigmoid(-(tf.matmul(X,W)+b)) -> 다른방법

row = x_data.__len__()
# row = 1 -> 난 costfunction 공식대로 했을뿐인데 (물론 최소화 하는과정에서 나누기는 아무런 영향을 미치지 않지만) row = 1로 주지 않는한 학습이 엄청나게 오래걸림. 데이터문젠가...
# 근데 자세히보면 cost가 굉장히 큰걸 알수있다
row = 1
cost = -tf.reduce_mean(y * tf.log(sig) + (1-y) * tf.log(1-sig))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(sig > 0.8, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

def runLogi(x_data, y_data):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10001):
            cost_val, accuracy_val, _ = sess.run([cost, accuracy, train],
                                   feed_dict={
                                       X:x_data,
                                       y:y_data
                                   })

            if i % 200 == 0:
                print(i, " cost: ", cost_val, "accuracy: ", accuracy_val)

        h, c, a = sess.run([sig, predicted, accuracy],
                           feed_dict={
                               X:x_data,
                               y:y_data
                           })
        print(h,a)
# runLogi(x_data, y_data)

### Binary Logi with data
data = np.loadtxt('data-week2.csv',delimiter=',')
x_data = data[:, :-1]
y_data = data[:, 8:9]

print x_data.shape
print y_data.shape

# runLogi(x_data, y_data)

def runSoftmax(x_data, y_data, x_row=None, y_row=None):
    print x_row, y_row
    X = tf.placeholder(tf.float32, shape=[None, x_row])
    W = tf.Variable(tf.random_normal(shape=[x_row, y_row], dtype=tf.float32))
    b = tf.Variable(tf.random_normal(shape=[y_row], dtype=tf.float32))

    y = tf.placeholder(tf.float32, shape=[None, y_row])

    h = tf.matmul(X, W) + b
    softmax = tf.nn.softmax(h)
    # softmax = tf.div(tf.exp(h), tf.reduce_mean(tf.exp(h)))
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y)
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(softmax), axis=1))
    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    prediction = tf.argmax(h, 1)
    correction = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),tf.arg_max(softmax,1)), tf.float32))

    def isMatched(sess, y_data, y_hat):
        m = len(y_data)
        matched = 0
        for i in range(m):
            if (sess.run(tf.arg_max(y_data[i],0)) == sess.run(tf.arg_max(y_hat[i],0))):
                matched += 1

        return matched

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(30001):
            cost_val, softmax_val, accuracy_val, _ = sess.run([cost, softmax, accuracy,optimizer],
                               feed_dict={
                                   X:x_data,
                                   y:y_data
                               })
            if i % 200 == 0:
                print (i, cost_val)
                # print isMatched(sess, y_data, softmax_val)
                print accuracy_val

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]]

# runSoftmax(x_data, y_data, len(x_data[0]), len(y_data[0]))

# TF방식으로 W(theta)와 bias(뒤에 붙는 상수) 를 따로 정의하여 푸는 방식의 구현
def runLogi2(x_data, y_data):
    x_row, x_col = x_data.shape
    y_row, y_col = y_data.shape
    X = tf.placeholder(shape=[None, x_col], dtype=tf.float32)
    y = tf.placeholder(shape=[None, y_col], dtype=tf.float32)

    W = tf.Variable(tf.zeros(shape=[x_col, 1]), dtype=tf.float32)
    b = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32)
    h = tf.matmul(X,W) + b

    sig = tf.div(1., 1. + tf.exp(-h))

    cost = tf.div(-tf.reduce_mean(y * tf.log(sig) + (1 - y) * tf.log(1 - sig)), 1)
    train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    # https: // stats.stackexchange.com / questions / 184448 / difference - between - gradientdescentoptimizer - and -adamoptimizer - tensorflow
    # 엄청난 정확도의 상승. 학습속도 빠르기 매우 상승
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


    predicted = tf.cast(sig > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print sess.run(cost, feed_dict={X:x_data, y:y_data})


        for i in range(10001):
            cost_val, accuracy_val, _ = sess.run([cost, accuracy, train],
                                   feed_dict={
                                       X:x_data,
                                       y:y_data
                                   })

            if i % 200 == 0:
                print(i, " cost: ", cost_val, "accuracy: ", accuracy_val)

        h, c, a, w_val, b_val = sess.run([sig, predicted, accuracy, W, b],
                           feed_dict={
                               X:x_data,
                               y:y_data
                           })
        print(w_val, b_val, a)

data = np.loadtxt('ex2data1.txt', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1:]

print y_data.shape

# runLogi2(x_data, y_data)

# Ng 교수님 방식으로 W(theta)와 bias(뒤에 붙는 상수) 를 같이 정의하여 푸는 방식의 구현
def runLogi3(x_data, y_data):
    x_row, x_col = x_data.shape
    y_row, y_col = y_data.shape
    # reshape X
    x_data = np.c_[np.ones(x_row), x_data]
    x_row, x_col = x_data.shape

    X = tf.placeholder(shape=[None, x_col], dtype=tf.float32)
    y = tf.placeholder(shape=[None, y_col], dtype=tf.float32)

    W = tf.Variable(tf.zeros(shape=[x_col, 1]), dtype=tf.float32)
    h = tf.matmul(X,W)

    sig = tf.div(1., 1. + tf.exp(-h))

    cost = tf.div(-tf.reduce_mean(y * tf.log(sig) + (1 - y) * tf.log(1 - sig)), 1)
    train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    # https: // stats.stackexchange.com / questions / 184448 / difference - between - gradientdescentoptimizer - and -adamoptimizer - tensorflow
    # 엄청난 정확도의 상승. 학습속도 빠르기 매우 상승
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


    predicted = tf.cast(sig > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print sess.run(cost, feed_dict={X:x_data, y:y_data})


        for i in range(10001):
            cost_val, accuracy_val, _ = sess.run([cost, accuracy, train],
                                   feed_dict={
                                       X:x_data,
                                       y:y_data
                                   })

            if i % 200 == 0:
                print(i, " cost: ", cost_val, "accuracy: ", accuracy_val)

        h, c, a, w_val = sess.run([sig, predicted, accuracy, W],
                           feed_dict={
                               X:x_data,
                               y:y_data
                           })
        print(w_val, a)

        print sess.run([sig], feed_dict={X:[[1, 45, 85]]})

runLogi3(x_data, y_data)