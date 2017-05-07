#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

x_data = np.loadtxt('files/ex8data1_X.csv', delimiter=',')
x_val_data = np.loadtxt('files/ex8data1_Xval.csv', delimiter=',')
y_val_data = np.loadtxt('files/ex8data1_yval.csv', delimiter=',')
print x_data.shape

def plot(X, show=True, marker='o', color='blue'):
    plt.plot(X[:, 0], X[:, 1],
             color=color,
             marker=marker,
             fillstyle='full',
             markeredgecolor=color,
             markeredgewidth=0.0,
             linestyle='None')

    if (show):
        plt.show()

# plot(x_data)

# multivariate_var = np.sum(np.multiply((X - m), (X - m)))
# print np.cov(X.T)
def estimateGaussian(X):
    m = np.mean(X, axis=0)
    var = np.cov(X.T)

    return (m, var)

def run_ad(x_data, x_val_data, y_val_data, draw_crossvalidate=False, draw_x=True):
    mu, sigma = estimateGaussian(x_data)
    dist = tf.contrib.distributions.MultivariateNormalFull(mu, sigma)
    n = mu.shape[0]

    tfSigma = tf.constant(sigma)
    tfMu = tf.constant(mu)
    X = tf.placeholder(dtype=tf.float64)

    downside = tf.constant(((2 * math.pi) ** (n / 2)) * (np.linalg.det(sigma) ** (0.5)))
    upside = tf.exp( (-0.5) * tf.reduce_sum(tf.matmul((X-tfMu), (tf.matrix_inverse(tfSigma))) * (X-tfMu), axis=1))
    custom_dist = upside/downside

    p = dist.pdf(x_val_data)
    min_p = tf.reduce_min(p)
    max_p = tf.reduce_max(p)

    # tp = tf.equal(test, y)

    best_f1 = 0
    best_thre = 0

    with tf.Session() as sess:
        p_val, min_val, max_val= sess.run([p, min_p, max_p])
        print p_val
        print sess.run(custom_dist, feed_dict={X: x_val_data})


        for thre in np.linspace(min_val, max_val, num=1000):
            pred_with_thre = p_val < thre

            tp = float(sum(1 for p, v in zip(pred_with_thre, y_val_data) if p == 1 and p == v))
            fp = float(sum(1 for p, v in zip(pred_with_thre, y_val_data) if p == 1 and p != v))
            tn = float(sum(1 for p, v in zip(pred_with_thre, y_val_data) if p == 0 and p == v))
            fn = float(sum(1 for p, v in zip(pred_with_thre, y_val_data) if p == 0 and p != v))

            prec = tp / (tp + fp) if tp + fp != 0 else 0
            rec = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if prec + rec != 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_thre = thre

        print best_thre
        print best_f1
        if draw_crossvalidate:
            draw_anomalies(x_val_data, p_val, best_thre)

        if draw_x:
            p = dist.pdf(x_data)
            p_val = sess.run(p)
            draw_anomalies(x_data, p_val, best_thre)

def draw_anomalies(X, p_val, thre):
    pred_with_thre = p_val < thre
    anomalies = np.where(pred_with_thre == True)
    print anomalies

    plot(X, show=False)
    for v in anomalies:
        plot(X[v, :],marker='X', color='red', show=False)
    plt.show()

run_ad(x_data, x_val_data, y_val_data)

# x_data = np.loadtxt('files/ex8data2_X.csv', delimiter=',')
# x_val_data = np.loadtxt('files/ex8data2_Xval.csv', delimiter=',')
# y_val_data = np.loadtxt('files/ex8data2_yval.csv', delimiter=',')
# print x_data.shape
# print x_val_data.shape
#
# run_ad(x_data, x_val_data, y_val_data)

# (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))