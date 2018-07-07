#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from math import sqrt
MOVING_AVERAGE_DECAY = 0.9999


def tf_inference(images, BATCH_SIZE, image_size, NUM_CLASSES):
    def put_kernels_on_grid (kernel, pad = 1):
        #https://gist.github.com/kukuruza/03731dc494603ceab0c5
        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))
        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7

    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        tensor_name = x.op.name
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name=scope.name)

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable('weights', shape=[256, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name=scope.name)

    return fc7