import tensorflow as tf
import os
import matplotlib.image as implt
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class EncoderNet:
    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.01))
        self.conv1_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=[16]))

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.01))
        self.conv2_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=[32]))

        self.w1 = tf.Variable(tf.truncated_normal(shape=[7*7*32, 100]))

    def forward(self, x):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, [1, 1, 1, 1], padding='SAME') + self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # 形状改变为14*14*16
        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1, self.conv2_w, [1, 1, 1, 1], padding='SAME') + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # 形状改变7*7*32
        self.flat = tf.reshape(self.pool2, shape=[-1, 7*7*32])
        self.output = tf.matmul(self.flat, self.w1)

        return self.output
        # 形状为batch_size * 100


class DecoderNet:
    def __init__(self):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.init_state = self.cell.zero_state(100, dtype=tf.float32)

        self.w2 = tf.Variable(tf.truncated_normal(shape=[128, 10]))

    def forward(self, y):
        y = tf.expand_dims(y, axis=1)
        # y = tf.tile(y, [1, 4, 1])
        y, _ = tf.nn.dynamic_rnn(self.cell, y, initial_state=self.init_state, time_major=False)

        y = tf.reshape(y, [-1, 128])
        y = tf.nn.softmax(tf.matmul(y, self.w2))
        y = tf.reshape(y, [-1, 10])
        # 返回形状为[batch_size,10]
        return y


class CnnseqNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()

    def forward(self):
        y = self.encoderNet.forward(self.x)
        self.output = self.decoderNet.forward(y)

    def backward(self):
        self.loss = tf.reduce_mean((self.output - self.y)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracys(self):
        self.bool = tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.bool, dtype=tf.float32))


if __name__ == '__main__':
    cnn2seq = CnnseqNet()
    cnn2seq.forward()
    cnn2seq.backward()
    cnn2seq.accuracys()

    init = tf.global_variables_initializer()
    plt.ion()
    a = []
    b = []
    save = tf.train.Saver()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Session() as sess:
        sess.run(init)
        # sess.run(save.restore(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_cnn2seq_minist\ckpt' ))

        for epoch in range(5000):
            train_x, train_y = mnist.train.next_batch(100)
            train_x1 = train_x.reshape([100, 28, 28, 1])
            train_loss, _, train_output = sess.run([cnn2seq.loss, cnn2seq.opt, cnn2seq.output],
                                                   feed_dict={cnn2seq.x:train_x1, cnn2seq.y:train_y})
            if epoch % 100 == 0:
                test_x, test_y = mnist.test.next_batch(100)
                test_x = test_x.reshape([100, 28, 28, 1])
                test_accu, test_output = sess.run([cnn2seq.accuracy,cnn2seq.output],
                                                  feed_dict={cnn2seq.x:test_x, cnn2seq.y:test_y})
                test_output_y = np.argmax(test_output[1])
                test_label = np.argmax(test_y[1])
                print('epoch:{} train_loss:{:5.4f},test_accuracy:{:5.2f}%'.format(epoch, train_loss, test_accu*100))
                print('test_output:{}, test_label:{}'.format(test_output_y, test_label))
                save.save(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_cnn2seq_minist\ckpt' )




