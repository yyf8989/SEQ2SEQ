import numpy as np

def _one_hot():
    z = np.zeros(shape=(4,10))
    for i in range(4):
        index = int("1 7 3 4".split(" ")[i])
        z[i][index]+=1
    return z

if __name__ == '__main__':
    result = _one_hot()
    print(result)

a = [[1,2],[3,4]]
print(a)


import tensorflow as tf

a = tf.constant([[1,2], [3,4]])
b = tf.reduce_sum(a, axis=1)

with tf.Session() as sess:
    # sess.run(a)
    sess.run(b)
    print(sess.run(b))