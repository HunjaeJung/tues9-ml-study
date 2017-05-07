#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# x_data = np.loadtxt('ex7-csv/ex7data1.csv', delimiter=',')
x_data = np.loadtxt('ex7-csv/ex7data2.csv', delimiter=',')
# print x_data.shape[0]

def drawFig(X, ax):
    minx = miny = -1
    maxx = maxy = 30

    ax.plot(X[:, 0], X[:, -1],
             marker='o',
             fillstyle='full',
             markeredgecolor='red',
             markeredgewidth=0.0,
             linestyle='None')

# drawFig(x_data)

def kMeanInitCentroid(X, k):
    t = range(0, X.shape[0])
    random.shuffle(t)

    return [X[t[i]] for i in range(k)]

centroid_data = kMeanInitCentroid(x_data, 3)
# print centroid_data
centroid_changed = []

class KMean():

    def __init__(self, x_data, centroid_data):
        global X; X = tf.placeholder(tf.float32)
        global Centroid; Centroid= tf.placeholder(tf.float32)
        global expanded_vectors; expanded_vectors = tf.expand_dims(X, 0)
        global expanded_centroids; expanded_centroids = tf.expand_dims(Centroid, 1)

        global distances; distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
        global min_distances; min_distances = tf.arg_min(distances, 0)
        global total_distances; total_distances = tf.reduce_sum(tf.reduce_min(distances,0))

        self.x_data = x_data
        self.centroid_data = centroid_data
        self.track_centroid = []

    def findClosestCentroids(self):
        with tf.Session() as sess:
            return sess.run([min_distances],
                            feed_dict={
                                X: self.x_data,
                                Centroid: self.centroid_data
                            }
                            )

    def computeMean(self, C, K, d):
        new_centroid = np.zeros([K, d])
        for i in range(C[0].__len__()):
            v = C[0][i]
            new_centroid[v, :] = new_centroid[v, :] + self.x_data[i, :]
        # print C
        for k in range(K):
            len = (C[0] == k).sum()
            new_centroid[k, :] = new_centroid[k, :] / len
            if len == 0:
                new_centroid[k, :] = centroid_data[k, :]

        return new_centroid

    def addCentroid(self):
        self.track_centroid.append(self.centroid_data)

    def getTrackedCentroid(self):
        return self.track_centroid

    def getCentroid(self):
        return self.centroid_data

    def run(self):
        dimension = self.x_data.shape[1]

        for i in range(10):
            self.addCentroid()
            self.centroid_data = self.computeMean(self.findClosestCentroids(), self.centroid_data.__len__(), dimension)

        # print centroid_data

    def compressToCentroid(self):
        # __ret = []
        min_distances_val = self.findClosestCentroids()
        # print min_distances_val.shape
        # for i in min_distances_val.__iter__():
        #     print i
        #     __ret.append(centroid_data[i])
        return np.array(self.centroid_data[min_distances_val])

    def totalDistance(self):
        with tf.Session() as sess:
            return sess.run([total_distances], feed_dict={X:self.x_data, Centroid:self.centroid_data})

def drawKmean(X, tracked_centroid):
    np_centroid_changed = np.array(tracked_centroid)
    print np_centroid_changed
    x = [4.0, 7.0, 4.5]
    y = [3.0, 1.0, 5.5]
    u = [0, 0, 0]
    v = [1, 1, 1]

    plt.plot(X[:, 0], X[:, -1],
             marker='o',
             fillstyle='full',
             markeredgecolor='red',
             markeredgewidth=0.0,
             linestyle='None')

    for i in range(np_centroid_changed.shape[0]):
        if i == np_centroid_changed.shape[0] - 1:
            plt.plot(np_centroid_changed[i][:, 0], np_centroid_changed[i][:, -1],
                     color='black',
                     marker='o',
                     fillstyle='full',
                     markeredgecolor='red',
                     markeredgewidth=0.0,
                     linestyle='None')
            continue

        plt.plot(np_centroid_changed[i][:, 0][0], np_centroid_changed[i][:, -1][0],
                 color='red',
                 marker='o',
                 fillstyle='full',
                 markeredgecolor='red',
                 markeredgewidth=0.0,
                 linestyle='None')
        plt.plot(np_centroid_changed[i][:, 0][1], np_centroid_changed[i][:, -1][1],
                 color='orange',
                 marker='o',
                 fillstyle='full',
                 markeredgecolor='orange',
                 markeredgewidth=0.0,
                 linestyle='None')
        plt.plot(np_centroid_changed[i][:, 0][2], np_centroid_changed[i][:, -1][2],
                 color='blue',
                 marker='o',
                 fillstyle='full',
                 markeredgecolor='blue',
                 markeredgewidth=0.0,
                 linestyle='None')


    plt.show()

# drawFig(x_data, ax)
# plt.show()
# test()
# kMean = KMean(x_data, centroid_data)
# kMean.run()
# centroid_changed = kMean.getTrackedCentroid()
# #
# drawKmean(x_data, centroid_changed)

#### Quiver 실패!!!!
# for i in range(np_centroid_changed.shape[0]):
#     plt.plot(np_centroid_changed[i][:, 0][0], np_centroid_changed[i][:, -1][0],
#              color='red',
#              marker='o',
#              fillstyle='full',
#              markeredgecolor='red',
#              markeredgewidth=0.0,
#              linestyle='None')
#
#     if i == np_centroid_changed.shape[0] - 1:
#         print 'pass'
#         continue
#
#     U1, U2, V1, V2 = np_centroid_changed[i][:, 0][0], np_centroid_changed[i][:, -1][0], \
#                      np_centroid_changed[i + 1][:, -1][0] - np_centroid_changed[i][:, 0][0], \
#                      np_centroid_changed[i + 1][:, -1][0] - np_centroid_changed[i][:, -1][0]
#     scale = np.sqrt(U2 ** 2 + V2 ** 2)
#     print scale
#     U2, V2 = U2 / scale, V2 / scale
#     # scale = np.linalg.norm(np.array([x[i+1] - x[i], y[i+1] - y[i]]))
#     plt.quiver(x[i], y[i], U2, V2, scale=scale)
#
#     plt.quiver(U1, V1, U2, V2, scale=scale,
#                color='red')
#
#     plt.quiver(np_centroid_changed[i][:, 0][1], np_centroid_changed[i][:, -1][1],
#                np_centroid_changed[i + 1][:, -1][1], np_centroid_changed[i + 1][:, -1][1],
#                color='blue')
#     plt.quiver(np_centroid_changed[i][:, 0][2], np_centroid_changed[i][:, -1][2],
#                np_centroid_changed[i + 1][:, -1][2], np_centroid_changed[i + 1][:, -1][2],
#                color='orange')

##### Image Compression (256x256x256 color based -> 16 color based image)

import matplotlib.image as mpimg
import numpy as np

def load_image_test(infilename):
    return mpimg.imread(infilename)

x_data = load_image_test('ex7-csv/bird_small.png')

def drawImage(image_data, show=True):
    # print rows.shape
    plt.imshow(image_data)

    if(show):
        plt.show()

print x_data.shape
# drawImage(x_data)

compressed_color = 16
loopcount = 5
image_length_width = x_data.shape[0]
image_length_height = x_data.shape[1]

# normalize
x_reshaped_data = x_data.reshape(image_length_width*image_length_height, 3)

print x_reshaped_data.shape
centroid_data = kMeanInitCentroid(x_reshaped_data, compressed_color)
centroid_data = np.array(centroid_data, dtype=np.float32)

def runKmeanWithLoop(count):
    KMEAN_ARR = []
    KMEAN_DIST_ARR = []

    for i in range(count):
        centroid_data = np.array(kMeanInitCentroid(x_reshaped_data, compressed_color), dtype=np.float32)
        kMean = KMean(x_reshaped_data, centroid_data)
        kMean.run()

        td = kMean.totalDistance()[0]
        print td
        if math.isnan(td):
            continue
        KMEAN_DIST_ARR.append(td)
        KMEAN_ARR.append(kMean.compressToCentroid())

    compressed_data = KMEAN_ARR[KMEAN_DIST_ARR.index(np.min(np.array(KMEAN_DIST_ARR)))]
    compressed_data = compressed_data.reshape(image_length_width,image_length_height,3)
    print compressed_data.shape
    # compressed_data = kMean.compressToCentroid()
    # compressed_data = compressed_data.reshape(128,128,3)

    plt.subplot(121)
    plt.title('Original Image (normal RGB)')
    drawImage(x_data, False)

    plt.subplot(122)
    plt.title('Compressed Image (16 color)')
    drawImage(compressed_data, False)
    plt.show()

    print compressed_data
    print x_data
runKmeanWithLoop(loopcount)

# X = tf.placeholder(tf.float32)
# Centroid = tf.placeholder(tf.float32)
# expanded_vectors = tf.expand_dims(X, 0)
# expanded_centroids = tf.expand_dims(Centroid, 1)
# subtracted = tf.subtract(expanded_vectors, expanded_centroids)
# distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
# min_distances = tf.arg_min(distances, 0)
# total_distnaces = tf.reduce_sum(distances)

# with tf.Session() as sess:

    # min_value = tf.reduce_min(distances, 0)
    # expanded_vectors, expanded_centroids, distances = sess.run([expanded_vectors, expanded_centroids, subtracted],
    #                                                 feed_dict={
    #                                                     X:x_reshaped_data,
    #                                                     Centroid:centroid_data
    #                                                 })
    #
    # np_ev = np.array(expanded_vectors)
    # np_centroid = np.array(expanded_centroids)
    #
    # print x_reshaped_data.shape, np_ev.shape
    # print centroid_data.shape, np_centroid.shape
    # print distances.shape
    #
    # print np_ev
    # print np_centroid
    # print distances
    # distances, min_distances, min_value = sess.run([distances, min_distances, min_value],feed_dict={
    #                                        X: x_reshaped_data,
    #                                        Centroid: centroid_data
    #                                    })
    #
    # print distances.shape, min_distances.shape
    # print min_distances
    # test = []
    # for i in min_distances.__iter__():
    #     test.append(centroid_data[i])
    # test = np.array(test)
    # print test.shape

# centroid_data = kMeanInitCentroid(x_data, 3)
# # print centroid_data
# X = tf.placeholder(tf.float32)
# Centroid = tf.placeholder(tf.float32)
# expanded_vectors = tf.expand_dims(X, 0)
# expanded_centroids = tf.expand_dims(Centroid, 1)
#
# distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
# min_distances = tf.arg_min(distances, 0)
# centroid_changed = []
