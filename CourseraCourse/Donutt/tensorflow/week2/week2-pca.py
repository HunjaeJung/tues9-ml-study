#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

x_data = np.loadtxt('ex7-csv/ex7data1.csv', delimiter=',')
print x_data.shape

def drawFig(X):
    minx = miny = -1
    maxx = maxy = 30

    plt.plot(X[:, 0], X[:, -1],
             marker='o',
             fillstyle='full',
             markeredgecolor='red',
             markeredgewidth=0.0,
             linestyle='None')
    plt.show()

# drawFig(x_data)

def meanNormalization(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X

# print x_data
# print meanNormalization(x_data)
# drawFig(x_data)

# x_data = meanNormalization(x_data)

def getCovariance(X):
    return np.dot(X.T, X) / X.shape[0]

# sigma = getCovariance(x_data)

def getSVD(x_data, sigma_data):
    X = tf.placeholder(dtype=tf.float32)
    sigma = tf.placeholder(dtype=tf.float32)
    svd = tf.svd(sigma, full_matrices=True, compute_uv=True)

    with tf.Session() as sess:
        svd_data = sess.run([svd],
                 feed_dict={
                     sigma:sigma_data
                 })

        # 값이 eigenvector 값이 기호가 다름... 아 왜그러는지 모르겟다
        # return np.array(svd_data[0][0]), np.array(svd_data[0][1]), np.array(svd_data[0][2])
    return np.linalg.svd(sigma_data)

# s, u, v = getSVD(x_data, sigma)

# u, s ,v = getSVD(x_data, sigma)
# print sigma
# print u
# print s
# print v

def getReduceData(u, k):
    return u[:, 0:k]

# u_reduce_data = getReduceData(u, 1)

def getProjection(X, u_reduce):
    return np.dot(X, u_reduce)

# projection_x_data = getProjection(x_data, u_reduce_data)
# print projection_x_data.shape

def getRecover(X, u, k):
    return np.dot(X, u[:, 0:k].T)

# recover_x_data = getRecover(projection_x_data, u)
# print recover_x_data.shape

def drawPCA(X, X_recover):
    # print X
    # print X_recover
    plt.plot(X[:, 0], X[:, -1],
             marker='o',
             fillstyle='full',
             markeredgecolor='blue',
             markeredgewidth=0.0,
             linestyle='None')

    plt.plot(X_recover[:, 0], X_recover[:, -1],
             color='red',
             marker='o',
             fillstyle='full',
             markeredgecolor='red',
             markeredgewidth=0.0,
             linestyle='None')
    plt.show()

# drawPCA(x_data, recover_x_data)

# x_data = np.loadtxt('ex7-csv/ex7faces.csv', delimiter=',')
# x_data = x_data[0:100, :]
# np.savetxt('ex7-csv/ex7faces-100.csv', x_data, delimiter=',')

# x_data = np.loadtxt('ex7-csv/ex7faces-100.csv', delimiter=',')
# x_data = x_data[0:4, :]
# x_data = x_data.reshape(128, 32)
# # x_data = x_data.reshape(96,96)
#
# plt.imshow(x_data, cmap='gray')
# plt.show()

def drawFace(origin_data, image_length, show=True):
    K = origin_data.shape[0]
    # calculate rows count
    k = math.sqrt(K).__int__()
    # print k

    # we need to show full image as k by k.
    rows = None
    row_stack = 0
    for row_count in range(k):
        row = origin_data[k*(row_stack):k*(row_stack+1), :]
        row = row.reshape(image_length*k, image_length)

        if rows == None:
            rows = row
        else:
            rows = np.hstack((rows, row))

        row_stack += 1

    # print rows.shape
    plt.imshow(rows.T, cmap='gray')

    if(show):
        plt.show()

origin_data = np.loadtxt('ex7-csv/ex7faces-100.csv', delimiter=',')
print origin_data.shape
# drawFace(origin_data, 32)

normalized_origin_data = meanNormalization(origin_data)
sigma = getCovariance(normalized_origin_data)
u, s ,v = getSVD(normalized_origin_data, sigma)
# print sigma
print sigma.shape
print u.shape
# drawFace(u[:,0: 100].T, 32)

compress_k = 20
u_reduce_data = getReduceData(u, compress_k)
print u_reduce_data.shape

projection_x_data = getProjection(normalized_origin_data, u_reduce_data)
print projection_x_data.shape

recover_x_data = getRecover(projection_x_data, u, compress_k)
print recover_x_data.shape

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.title('Original Data (100 Picture)')
drawFace(origin_data, 32, False)
plt.subplot(132)
plt.title('Principal Component')
drawFace(u[:,0: 100].T, 32, False)
plt.subplot(133)
plt.title('Reconstruction (K = {0})'.format(compress_k))
drawFace(recover_x_data, 32, False)
plt.show()

