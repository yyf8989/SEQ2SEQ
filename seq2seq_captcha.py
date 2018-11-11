import tensorflow as tf
import os
import matplotlib.image as implt
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as pdraw
import PIL.ImageFont as Font

image_path = r'E:\Pycharmprojects\SEQ2SEQ\Code'
font_path = r'E:\Pycharmprojects\SEQ2SEQ\arial.ttf'
save_path = r'E:\Pycharmprojects\SEQ2SEQ\ckpt_seq2seq\pt'
batch_size = 100

class EncoderNet:

    def __init__(self):
        # NHWC形式(100, 60, 120, 3) ==> NWHC(100, 120, 60, 3) ==> NW* HC(100*120, 60*3)（横向处理图片）
        self.w1 = tf.Variable(tf.truncated_normal(shape=[60*3, 128], stddev=tf.sqrt(1 / 128)))
        # 定义权重，形状与输入对应，定义神经元个数为128
        self.b1 = tf.Variable(tf.zeros([128]))

    def forward(self, x):
        # 变量名只能做Encoder中使用
        with tf.name_scope('EncoderNet') as scope1:
            # NHWC --> NWHC,矩阵转置,交换1/2轴
            y = tf.transpose(x, [0,2,1,3])
            # print(y.shape)
            # 重置形状后继续转化为NV型结构，NHWC--> NWHC-->NW*HC -->N*V
            y = tf.reshape(y, [batch_size*120, 60*3])
            # print(y.shape)
            # 输入relu时为[100*120,60*3]-->乘以权重w1[60*3, 128]-->[100*120, 128]
            y = tf.nn.relu(tf.matmul(y, self.w1) + self.b1)
            # print(y.shape)
            # 因为要输入rnn，所以数据类型转换为NSV结构，就是[100*120,128]-->[100,120,128]
            y = tf.reshape(y, [batch_size, 120, 128])
            # print(y.shape)
            # 建立RNN cell结构，初始为128个cell unit
            cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            # 初始化所有N（10个batch）的cell unit state为0
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            # 接收LSTMCell中输出值，一个为output，一个为final_state
            encoder_output, _ = tf.nn.dynamic_rnn(cell, y, initial_state=init_state, time_major=False, scope=scope1)
            # 输出的形状不变，仍为NSV(100, 120, 128)结构,但是针对RNN输出，只看最后一个state，所以从State角度出发
            # 进行形状变换，NSV-->SNV --> (120, 100, 128)，取[-1]变为（1，100，,18），即为（100,128）
            # print(encoder_output.shape)
            y = tf.transpose(encoder_output, [1, 0, 2])[-1]
            # print(y.shape)

            # 返回y,形状为（100,128）
            return y

# 定义解码类，由结构可知，此处接收encoder返回值y
class DecoderNet:
    def __init__(self):
        # 10分类问题,故定义权重形状为[128, 10]
        self.w2 = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev=tf.sqrt(1 / 10)))
        self.b2 = tf.Variable(tf.zeros([10]))

    def forward(self, y):
        # 变量名只在DecoderNet中使用
        with tf.name_scope('DecoderNet') as scope2:
            # 因为是解码，但是编码输出值为（100,128）结构，但又因要输入到解码层为RNN结构，故结构需要变换为NSV结构
            # y = tf.reshape(y, [batch_size, 1, 128])
            # 或者使用扩维语句，tf.expand_dims(在tensor中加入一个为1的维度)
            y = tf.expand_dims(y, axis=1)
            # 将[100,1,128]用广播的形式进行扩展成[100,4,128]，原因为图片中有4个数字，所以可以理解为要用4个步长来接收
            y = tf.tile(y, [1, 4, 1])
            # print(y.shape)
            # 定义RNN LSTMCell的unit个数为128
            cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            # 初始化N（100组）cell unit状态均为0
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            # 对LSTMCELL的记忆运算进行接收，如encoder所示
            decoder_output, _ = tf.nn.dynamic_rnn(cell, y, initial_state=init_state,
                                                                    time_major=False, scope=scope2)
            # 计算完成之后，输出值形状为[100,4,128],如上要将NSV-->NV结构
            # print(decoder_output.shape)
            y = tf.reshape(decoder_output, [batch_size*4, 128])
            # print(y.shape)
            self.y1 = tf.matmul(y, self.w2) + self.b2
            # self.y1激活函数处理后输出形状范围为（400,10），NV结构
            # 因为要进行softmax输出，故将NV结构拆分为NSV结构
            # print(self.y1.shape)
            self.y2 = tf.reshape(self.y1, [-1, 4, 10])
            # 使用softmax进行输出

            # print(self.y2.shape)
            y = tf.nn.softmax(self.y2)
            # 返回softmax输出后的值也就是解码后的值，形状为[100,4,10]
            # print(y.shape)
            return y


class Net:

    def __init__(self):
        # 输入NHWC结构
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 60, 120, 3])
        # 输出NSV结构，10分类
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4, 10])

        # 实例化编码网络
        self.encoderNet = EncoderNet()
        # 实例化解码网络
        self.decoderNet = DecoderNet()

    def forward(self):
        # 调用编码网络中前向运算函数，传入参数（归一化后的图像），得到最终的向量Y
        y = self.encoderNet.forward(self.x)
        # print(y.shape)
        # 调用解码网络中的前向运算函数，传入参数（编码后的向量y），得到结果
        self.output = self.decoderNet.forward(y)
        # print(self.output.shape)

    def backward(self):
        # 损失函数使用softmax交叉熵进行运算
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.decoderNet.y2, labels=self.y))
        # 优化器选用Adam优化器
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracys(self):
        # 先生成bool类型，使用self.output和self.y进行比较
        self.bool = tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1))
        # 修正bool类型的数据为float32，然后进行求平均值
        self.accuracy = tf.reduce_mean(tf.cast(self.bool, dtype=tf.float32))


class Sampling:

    def __init__(self):
        self.image_dataset = []
        # 遍历文件列表系统里的每一个文件
        for filename in os.listdir(image_path):
            # 从每个文件中读取每个图片的数据，然后进行归一化处理（除以255），然后添加一个bias，是数据更加集中于对称
            x = implt.imread(os.path.join(image_path, filename))/255. - 0.5
            # 用.将数据和文件名分开，然后取用前面的文件名作为数据输入
            ys = filename.split(' ')[0]
            # 对取用的文件名进行One——hot处理
            y = self.__one_hot(ys)
            # 数据处理完毕后进行拼接与封装
            self.image_dataset.append([x, y])

    def __one_hot(self, x):
        # 初始化一个[4,10]的0矩阵
        z = np.zeros(shape=(4, 10))
        for i in range(4):
            # 针对每一行的数据进行处理，因为验证码上为4个数字，所有我们要在4行中每一行生成一个index为该数的值
            # 数据取整，然后确认该数字在该行的index（索引）
            index = int(x[i])
            # 然后对该行的对应位置赋值为1
            z[i][index] += 1
        # 返回这个由验证码上数字组成的onehot数据集
        return z

    def image_get_batch(self, batch_size):
        # 创建batch xs的空集合
        xs = []
        # 创建batch ys的空集合
        ys = []
        # 生成对应batch_size的数据集
        for _ in range(batch_size):
            # 在图片总集合中随机取一张图片，共取batch_size个数据
            index = np.random.randint(0, len(self.image_dataset))
            # 找到对应索引的图片后，然后取用第一个数据作为输入数据，如135行所述
            xs.append(self.image_dataset[index][0])
            # 同理，将找到索引的数据第二个值作为labels
            ys.append(self.image_dataset[index][1])
        # 返回的为分别包含输入数据和标签的两个列表
        # print(np.array(xs).shape, np.array(ys).shape)
        return xs, ys


if __name__ == '__main__':
    # 依次实例化对应的方法
    sample = Sampling()
    seq2seq = Net()
    seq2seq.forward()
    seq2seq.backward()
    seq2seq.accuracys()


    # train_x, y = sample.image_get_batch(1)
    # print(train_x)
    # print(y)

    init = tf.global_variables_initializer()
    plt.ion()
    a = []
    b = []
    save = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        save.restore(sess, save_path=save_path)

        for epoch in range(1000):
            train_x, train_y = sample.image_get_batch(batch_size)
            # print(np.array(train_x).shape)
            train_loss, _ = sess.run([seq2seq.loss, seq2seq.opt], feed_dict={seq2seq.x:train_x,seq2seq.y:train_y})
            if epoch % 10 == 0:
                test_x, test_y = sample.image_get_batch(batch_size)
                # print(np.array(test_x).shape)
                # print(np.array(test_y).shape)
                test_output, test_accu = sess.run([seq2seq.output, seq2seq.accuracy], feed_dict={seq2seq.x:test_x, seq2seq.y:test_y})
                # 取用一组数据进行确认
                # print(test_output)
                # print(test_y)
                test_output = np.argmax(test_output[1], axis=1)
                label = np.argmax(test_y[1], axis=1)
                # 打印相关信息
                print('test_output:', test_output, end=" ")
                print('label:', label)
                print('epoch:{}, loss:{:5.5f}, accu:{:5.2f}%'.format(epoch, train_loss, test_accu*100))

                # # 还原图片
                # img = (test_x[0] + 0.5) * 255
                # # 从数组回复成图片
                # image = pimg.fromarray(np.uint8(img))
                # imgdraw = pdraw.ImageDraw(image)
                # font = Font.truetype(font_path, size=20)
                # imgdraw.text(xy=(0,0), text=str(test_output), fill='red', font=font)
                # # image.show()
                #
                a.append(epoch)
                b.append(train_loss)
                save.save(sess, save_path)

        plt.clf()
        plt.plot(a, b)
        # plt.pause(0.01)
        plt.show()

