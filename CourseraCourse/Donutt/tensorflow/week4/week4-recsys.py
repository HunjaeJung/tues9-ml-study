#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

#
y_data = np.loadtxt('files/ex8_movies_Y.csv', delimiter=',')
R_data = np.loadtxt('files/ex8_movies_R.csv', delimiter=',')

#
# print y_data.shape
# print R_data.shape
#
# # print 'Average rating for movie 1 (Toy Story): '
# # print np.mean(y_data[np.where(R_data[0, :] == 1), :])
#
# def plot(X, show=True, marker='o', color='blue'):
#     plt.plot(X,
#              color=color,
#              marker=marker,
#              markersize=1,
#              fillstyle='full',
#              markeredgecolor=color,
#              markeredgewidth=0.0,
#              linestyle='None')
#
#     if (show):
#         plt.show()
#
# plot(y_data)

# theta = user 가중치, x = 데이터에서의 feature 가중치 (장르 등)
feature_data = np.loadtxt('files/ex8_movieParams_num_features.csv', delimiter=',', dtype=np.float32)
movies_data = np.loadtxt('files/ex8_movieParams_num_movies.csv', delimiter=',', dtype=np.float32)
users_data = np.loadtxt('files/ex8_movieParams_num_users.csv', delimiter=',', dtype=np.float32)
theta_data = np.loadtxt('files/ex8_movieParams_Theta.csv', delimiter=',', dtype=np.float32)
x_data = np.loadtxt('files/ex8_movieParams_X.csv', delimiter=',', dtype=np.float32)

print x_data.shape
print theta_data.shape
print feature_data.shape
print movies_data.shape
print users_data.shape


def runCofi(y_data,R_data,x_data,theta_data, num_users = 3, num_movies = 4, num_features = 2):
    x_reshape = x_data[0:num_movies, 0:num_features]
    theta_reshape = theta_data[0:num_users, 0:num_features]
    y_reshape = y_data[0:num_movies, 0:num_users]
    r_reshape = R_data[0:num_movies, 0:num_users]

    X = tf.Variable(tf.constant(x_reshape), dtype=tf.float32)
    theta = tf.Variable(tf.constant(theta_reshape), dtype=tf.float32)

    y = tf.constant(y_reshape, dtype=tf.float32)
    R = tf.constant(r_reshape, dtype=tf.float32)

    print r_reshape
    print y_reshape
    print x_reshape
    print theta_reshape

    regularized_lambda = 1.5
    regularized_params_theta = tf.reduce_sum(tf.square(theta)) * (regularized_lambda) * 0.5
    regularized_params_x = tf.reduce_sum(tf.square(X)) * (regularized_lambda) * 0.5

    W = tf.matmul(theta, X, transpose_b=True)

    costFunction = tf.reduce_sum(tf.square(tf.where(tf.transpose(R)<1, tf.zeros_like(tf.transpose(R), dtype=tf.float32), W) - tf.transpose(y))) * (0.5)
    regularized_costFunction = costFunction + regularized_params_theta + regularized_params_x

    # gradient = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    gradient = tf.train.AdamOptimizer(learning_rate=0.01)
    train = gradient.minimize(costFunction)
    regularized_train = gradient.minimize(regularized_costFunction)

    # t = tf.where(tf.transpose(R)<1, tf.zeros_like(tf.transpose(R), dtype=tf.float32), W)
    #
    # x_gradient = tf.matmul(tf.matmul(theta, X, transpose_b=True) - tf.transpose(y), theta, transpose_a=True) + X * regularized_lambda
    # x_update = X.assign(x_gradient)
    # theta_gradient = tf.matmul(tf.matmul(theta, X, transpose_b=True) - tf.transpose(y), X) + theta * regularized_lambda
    # theta_update = theta.assign(theta_gradient)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print sess.run([costFunction])

        for i in range(10001):
            # x_v, t_v, cost, _ = sess.run([X, theta, regularized_costFunction, regularized_train])
            x_v, t_v, cost, _ = sess.run([X, theta, costFunction, train])
            if i % 200 is 0:
                print (i, cost)

                print sess.run(costFunction)
                # print sess.run([regularized_params_theta, regularized_params_x])

        print x_v, t_v

runCofi(y_data, R_data, x_data, theta_data)

def loadMovie():
    # return np.loadtxt('files/movie_ids.txt', delimiter=',')
    return open('files/movie_ids.txt')

movies = np.array(loadMovie().read().split('\n'))
movies = movies[:-1]
print movies.shape

my_ratings = np.zeros(movies.shape)
my_ratings[0] = 4
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[97] = 2
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print movies[0]

# print 'My Ratings:'
# for name, rating in zip(movies[np.where(my_ratings > 0)], my_ratings[np.where(my_ratings > 0)]):
#     print name, ':', rating

r_my_ratings = np.where(my_ratings>0, np.ones([1682,]), np.zeros([1682,]))

my_ratings = np.expand_dims(my_ratings, axis=1)
r_my_ratings = np.expand_dims(r_my_ratings, axis=1)

y_data = np.hstack((y_data, my_ratings))
R_data = np.hstack((R_data, r_my_ratings))
num_movies, num_users = y_data.shape; num_features = 10
print num_movies, num_users, num_features

nomalized_y = y_data.copy()

for i in range(num_movies):
    nomalized_y[i, np.where(R_data[i, :] > 0)] = y_data[i, np.where(R_data[i, :] > 0)] - y_data[i, np.where(R_data[i, :] > 0)].mean(axis=1)

def runCofi2(y_data, R_data, num_users = 3, num_movies = 4, num_features = 2):
    X = tf.Variable(tf.random_normal(shape=[num_movies, num_features]), dtype=tf.float32)
    theta = tf.Variable(tf.random_normal(shape=[num_users, num_features]), dtype=tf.float32)

    y_reshape = y_data[0:num_movies, 0:num_users]
    r_reshape = R_data[0:num_movies, 0:num_users]
    y = tf.constant(y_reshape, dtype=tf.float32)
    R = tf.constant(r_reshape, dtype=tf.float32)

    regularized_lambda = 1.5
    regularized_params_theta = tf.reduce_sum(tf.square(theta)) * (regularized_lambda) * 0.5
    regularized_params_x = tf.reduce_sum(tf.square(X)) * (regularized_lambda) * 0.5

    W = tf.matmul(theta, X, transpose_b=True)

    costFunction = tf.reduce_sum(tf.square(tf.where(tf.transpose(R)<1, tf.zeros_like(tf.transpose(R), dtype=tf.float32), W) - tf.transpose(y))) * (0.5)
    regularized_costFunction = costFunction + regularized_params_theta + regularized_params_x

    # gradient = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    gradient = tf.train.AdamOptimizer(learning_rate=0.001)
    train = gradient.minimize(costFunction)
    regularized_train = gradient.minimize(regularized_costFunction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(30001):
            x_v, t_v, cost, _ = sess.run([X, theta, costFunction, train])
            # print x_v, t_v
            if i % 200 is 0:
                print (i, cost)

                print sess.run(costFunction)
                # print sess.run([regularized_params_theta, regularized_params_x])

        # print x_v, t_v

# runCofi2(nomalized_y, R_data, num_users=200, num_movies=500, num_features=num_features)
