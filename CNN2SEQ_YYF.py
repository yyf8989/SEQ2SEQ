#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'YYF'

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as Img

class EncoderNet:
    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 8], dtype=tf.float32, stddev=0.01))
        self.conv1_b = tf.Variable(tf.zeros([8]))

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 16], dtype=tf.float32, stddev=0.01))
        self.conv2_b = tf.Variable(tf.zeros([16]))

        self.mlp_w = tf.Variable(tf.truncated_normal(shape=[15*30*16, 128], stddev=0.01))
        self.mlp_b = tf.Variable(tf.zeros([128]))

    def forward(self, x):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, [1, 1, 1, 1], 'SAME') + self.conv1_b)
        self.maxpool1 = tf.nn.max_pool(self.conv1, [1,2,2,1], [1,2,2,1], 'SAME')

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.maxpool1, self.conv2_w, [1, 1, 1, 1], 'SAME') + self.conv2_b)
        self.maxpool2 = tf.nn.max_pool(self.conv2, [1,2,2,1], [1,2,2,1], 'SAME')

        self.flat = tf.reshape(self.maxpool2, [-1, 15*30*16])

        # self.dropout = tf.nn.dropout(self.flat, keep_prob=0.6)

        self.MLP_output = tf.matmul(self.flat, self.mlp_w) +self.mlp_b

        return self.MLP_output


class DecoderNet:
    def __init__(self):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.init_state = self.cell.zero_state(100, dtype=tf.float32)

        self.mlp_w1 = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev=0.01))
        self.mlp_b1 = tf.Variable(tf.zeros([10]))

    def forward(self, x):
        y = tf.expand_dims(x, axis=1)
        y = tf.tile(y, [1, 4, 1])

        y, _ = tf.nn.dynamic_rnn(self.cell, y, initial_state=self.init_state, time_major=False)
        y = tf.reshape(y, [-1, 128])

        y = tf.nn.softmax(tf.matmul(y, self.mlp_w1) + self.mlp_b1)
        y = tf.reshape(y, [-1, 4, 10])

        return y

class CNN2SEQ_Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 60, 120, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 4, 10])

        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self):
        y = self.encoder.forward(self.x)
        self.output = self.decoder.forward(y)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.y))
        # self.loss = tf.reduce_mean(tf.pow(self.output - self.y, 2))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracys(self):
        self.bool = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.bool, dtype=tf.float32))


class Sample:
    def __init__(self):
        self.image_dataset = []

        for filename in os.listdir('E:\Pycharmprojects\SEQ2SEQ\Code'):
            x = plt.imread(os.path.join('E:\Pycharmprojects\SEQ2SEQ\Code', filename))/255. - 0.5
            ys = filename.split('.')[0]
            y = self.__one_hot(ys)
            self.image_dataset.append([x, y])

    def __one_hot(self, y):
        z = np.zeros([4, 10])
        for i in range(4):
            index = int(y[i])
            z[i][index] += 1
        return z

    def image_get_data(self, x):
        xs = []
        ys = []
        for _ in range(x):
            index = np.random.randint(0, len(self.image_dataset))
            xs.append(self.image_dataset[index][0])
            ys.append(self.image_dataset[index][1])
        return xs, ys


if __name__ == '__main__':
    sample = Sample()
    net = CNN2SEQ_Net()
    net.forward()
    net.backward()
    net.accuracys()

    init = tf.global_variables_initializer()
    plt.ion()
    a = []
    b = []
    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_CNN2SEQ_YYF\ckpt')

        for epoch in range(5000):
            train_x, train_y = sample.image_get_data(100)
            train_loss, train_acc, _ = sess.run([net.loss, net.accuracy, net.opt], feed_dict={net.x:train_x, net.y:train_y})

            if epoch % 10 == 0:
                test_x, test_y = sample.image_get_data(100)
                test_acc, test_output = sess.run([net.accuracy, net.output], feed_dict={net.x:test_x, net.y:test_y})
                test_output_label = np.argmax(test_output[0], 1)
                test_y_label = np.argmax(test_y[0], 1)
                print('test_output:{}， test_label:{}'.format(test_output_label, test_y_label))
                print('epoch:{}, train_loss:{:5.6f}, train_acc:{:5.2f}%,test_accu:{:5.2f}%'.format(
                    epoch, train_loss, train_acc*100, test_acc * 100))
                save.save(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_CNN2SEQ_YYF\ckpt')



