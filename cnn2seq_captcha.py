import tensorflow as tf
import os
import matplotlib.image as implt
import matplotlib.pyplot as plt
import numpy as np


class EncoderNet:
    def __init__(self):
        with tf.name_scope('encoder_cnn'):
            self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 16], stddev=0.02))
            self.conv1_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=[16]))

            self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.02))
            self.conv2_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=[32]))

            self.conv3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.02))
            self.conv3_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=[64]))

        with tf.name_scope('encoder_mlp'):
            self.w1 = tf.Variable(tf.truncated_normal(shape=[8*15*64, 128], stddev=tf.sqrt(1/128)))
            self.b1 = tf.Variable(tf.zeros(shape=[128]))

    def forward(self, x):
        with tf.name_scope('encoder_cnn'):
            self.conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, [1, 1, 1, 1], padding='SAME') + self.conv1_b)
            self.pool1 = tf.nn.max_pool(self.conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            # 形状改变为30*60*16
            # print(self.pool1.shape)
            self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1, self.conv2_w, [1, 1, 1, 1], padding='SAME') + self.conv2_b)
            self.pool2 = tf.nn.max_pool(self.conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            # 形状改变为15*30*32
            # print(self.pool2.shape)
            self.conv3 = tf.nn.relu(tf.nn.conv2d(self.pool2, self.conv3_w, [1, 1, 1, 1], padding='SAME') + self.conv3_b)
            self.pool3 = tf.nn.max_pool(self.conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            # 形状改变为8*15*64
            # print(self.pool3.shape)
        with tf.name_scope('encoder_mlp'):
            self.flat = tf.reshape(self.pool3, shape=[-1, 8*15*64])
            self.output = tf.matmul(self.flat, self.w1) + self.b1
            # print(self.output.shape)

            return self.output
        # 形状为batch_size * 128


class DecoderNet:
    def __init__(self):
        with tf.name_scope('decoder_cnn'):
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            self.init_state = self.cell.zero_state(100, dtype=tf.float32)

        with tf.name_scope('decoder_mlp'):
            self.w2 = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev=tf.sqrt(1/100)))
            self.b2 = tf.Variable(tf.zeros([10]))

    def forward(self, x):
        with tf.name_scope('decoder_cnn'):
            y = tf.expand_dims(x, axis=1)
            y = tf.tile(y, [1, 4, 1])
            y, _ = tf.nn.dynamic_rnn(self.cell, y, initial_state=self.init_state, time_major=False)
            # 输出形状为NSV 100*4*128
            # print(y.shape)

        with tf.name_scope('decoder_mlp'):
            y = tf.reshape(y, [-1, 128])
            # print(y.shape)
            y = tf.nn.softmax(tf.matmul(y, self.w2) + self.b2)
            # print(y.shape)
            y = tf.reshape(y, [-1, 4, 10])
            # print(y.shape)
            # 返回形状为[100,4,10]
            return y


class CnnseqNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 60, 120, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 4, 10])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()


    def forward(self):
        with tf.name_scope('CNN2SEQ_forward'):
            y = self.encoderNet.forward(self.x)
            self.output = self.decoderNet.forward(y)

    def backward(self):
        with tf.name_scope('CNN2SEQ_loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
            tf.summary.scalar("loss", self.loss)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracys(self):
        with tf.name_scope('CNN2SEQ_accuracy'):
            self.bool = tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.bool, dtype=tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

class Sample:

    def __init__(self):
        with tf.name_scope('Sample'):
            self.image_dataset = []

            for filename in os.listdir('E:\Pycharmprojects\SEQ2SEQ\Code'):
                x = implt.imread(os.path.join('E:\Pycharmprojects\SEQ2SEQ\Code',filename))/255. - 0.5
                # print(x.shape)
                ys = filename.split(".")[0]
                y = self.__one_hot(ys)
                self.image_dataset.append([x, y])

    def __one_hot(self, x):
        with tf.name_scope('Sample'):
            z = np.zeros(shape=[4, 10])
            for i in range(4):
                index = int(x[i])
                z[i][index] += 1
            return z

    def image_get_batch(self, batch_size):
        with tf.name_scope('Sample'):
            xs = []
            ys = []
            for _ in range(batch_size):
                index = np.random.randint(0, len(self.image_dataset))
                xs.append(self.image_dataset[index][0])
                ys.append(self.image_dataset[index][1])
            return xs, ys


if __name__ == '__main__':
    sample = Sample()
    cnn2seq = CnnseqNet()
    cnn2seq.forward()
    cnn2seq.backward()
    cnn2seq.accuracys()

    init = tf.global_variables_initializer()
    plt.ion()
    a = []
    b = []
    save = tf.train.Saver()

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_cnnseq\ckpt')
        writer = tf.summary.FileWriter("./logs", sess.graph)
        for epoch in range(5000):
            train_x, train_y = sample.image_get_batch(100)
            # print(np.array(train_x).shape)
            train_loss, _ = sess.run([cnn2seq.loss, cnn2seq.opt],
                                                   feed_dict={cnn2seq.x:train_x, cnn2seq.y:train_y})
            # print(train_output.shape)
            # print(cnn2seq.y.shape)

            if epoch % 20 == 0:
                test_x , test_y = sample.image_get_batch(100)
                summary, test_output_y, test_accu = sess.run([merged,cnn2seq.output, cnn2seq.accuracy],
                                                    feed_dict={cnn2seq.x:test_x, cnn2seq.y:test_y})

                # print(train_output.shape)
                # print(np.array(test_y).shape)
                test_output_y= np.argmax(test_output_y[2], 1)
                test_label= np.argmax(test_y[2], 1)
                print('test_output:{}， test_label:{}'.format(test_output_y, test_label))
                print('epoch:{}, train_loss:{:5.6f}, test_accu:{:5.2f}%'.format(epoch, train_loss, test_accu*100))
                save.save(sess, 'E:\Pycharmprojects\SEQ2SEQ\ckpt_cnnseq\ckpt')
                tf.summary.scalar("accuracy", test_accu)
                writer.add_summary(summary, epoch)


