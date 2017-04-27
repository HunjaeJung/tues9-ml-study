#-*- coding: utf-8 -*-
# Tensor Ranks, Shapes, and Types

# Ranks : 몇 차원 Array 인가
# 0 Scalar | s = 483
# 1 Cector | v = [1, 2, 3]
# 2 Matrix | m = [[1,2,3], [1,2,3], [1,2,3]]
# 3 3-Tensor | t = [[[2],[4]], [[8],[10]]]
# n n-Tensor |

# Shapes : 각각의 element에 몇개씩 들어가잇나
# 0 [] | 0-D tensor A scalar
# 1 [D0] | 1-D [5]
# 2 [D0, D1] | [3, 4]
# 3 [D0, D1, D2] | [1, 4, 3]
# n
# ex) t= [[1,2,3], [4,5,6], [7,8,9]] => [3 3] shapes

# Data type
# float : tf.float32
# double : tf.float64
# int8 : tf.int8
# int32 : tf.int32
# etc

import tensorflow as tf
# import matplotlib.pyplot as plt

class LinearReg1:
    def __init__(self, x, y):
        self.W = tf.Variable(tf.random_normal([1]), name='weight')
        self.b = tf.Variable(tf.random_normal([1]), name='bias')

        self.x = x
        self.y = y
        self.h = self.x * self.W + self.b

        self.cost = tf.reduce_mean(tf.square(self.h - self.y))
        self.gradDescent()
        self.sess = tf.Session()

    def gradDescent(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train = optimizer.minimize(self.cost)

    def run(self):

        # initializes global variables in the graph
        self.sess.run(tf.global_variables_initializer())

        for step in range(2001):
            cost_val, W_val, b_val, _ = self.sess.run([self.cost, self.W, self.b, self.train],
                                                 feed_dict={
                                                     self.x: [1,2,3,4,5],
                                                     self.y: [2.1, 3.1, 4.1, 5.1, 6.1]
                                                 })

            if step % 20 == 0:
                print(step, cost_val, W_val, b_val)
                # print(step, sess.run(self.cost), sess.run(self.W), sess.run(self.b))

    def result(self, test_set):
        print(self.sess.run(self.h, feed_dict={self.x: test_set}))

### Linear Regression with TF
x_train = [1, 2, 3]
y_train = [1, 2, 3]
#lin = LinearReg(x_train, y_train)
#lin.run()

# shape=[None] => 1차원 무작위 크기
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
# lin = LinearReg1(X, y)
# lin.run()
# lin.result([5])

class LinearReg2:
    def __init__(self, learning_rate=0.01):
        X = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)
        self.w = tf.placeholder(tf.float32)
        self.h = X * self.w

        self.cost = tf.reduce_mean(tf.square(self.h - y))
        self.gradDescent()
        self.sess = tf.Session()
        # initializes global variables in the graph
        self.sess.run(tf.global_variables_initializer())

    def gradDescentCustom(self, X, y, learning_rate):
        gradient = tf.reduce_mean((self.w - y) * X)
        descent = self.w - learning_rate * gradient
        update = self.w.assign(descent)

    def gradDescent(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train = optimizer.minimize(self.cost)

    def plotCost(self, x, y):
        W_val = []
        cost_val = []
        for i in range(-30, 50):
            feed_W = i * 0.1
            curr_cost, curr_W = self.sess.run([self.cost, self.w], feed_dict={self.w: feed_W, X:x, y:y})
            W_val.append(curr_W)
            cost_val.append(curr_cost)

        plt.plot(W_val, cost_val)
        plt.show()

    def _run(self, x, y):
        return self.sess.run([self.cost, self.b, self.train],
                                                 feed_dict={
                                                     X: [1,2,3,4,5],
                                                     y: [2.1, 3.1, 4.1, 5.1, 6.1]
                                                 })
    def run(self, x, y):
        for step in range(2001):
            cost_val, W_val, b_val, _ = self._run(x,y)

            if step % 20 == 0:
                print(step, cost_val, W_val, b_val)
                # print(step, sess.run(self.cost), sess.run(self.W), sess.run(self.b))

    def result(self, test_set):
        print(self.sess.run(self.h, feed_dict={self.x: test_set}))


# lin = LinearReg2()
# lin.plotCost([1,2,3], [1,2,3])


####################

x1_train = [73, 93, 89, 96, 73]
x2_train = [80, 88, 91, 98, 66]
x3_train = [75, 93, 90, 100, 70]

y_train = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
h = x1*w1 + x2*w2 + x3*w3 + b

cost = tf.reduce_mean(tf.square(h-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2001):
#     cost_val, h_val, w1_val, w2_val, w3_val, b_val, _ = sess.run([cost, h, w1, w2, w3, b, train],
#                                   feed_dict={
#                                       x1: x1_train,
#                                       x2: x2_train,
#                                       x3: x3_train,
#                                       y: y_train
#                                   })
#     if step % 20 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction\n", h_val)
#         print(w1_val, w2_val, w3_val, b_val)

######### Matrix

x_train = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96,98,100], [73,66,70]]
y_train = [[152],[185],[180],[196],[142]]

X = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(h-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train],
                                  feed_dict={
                                      X: x_train,
                                      y: y_train
                                  })
    if step % 20 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction\n", h_val)
        # print(w1_val, w2_val, w3_val, b_val)
