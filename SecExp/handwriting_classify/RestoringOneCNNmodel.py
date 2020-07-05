import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)
def load_CNN_Model():
    # 变量不需要初始化
    X = tf.placeholder(tf.float32, [None, 784], name = 'X')
    Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')
    X_ = tf.reshape(X, [-1, 28, 28, 1])

    w1 = tf.get_variable('w1',[4,4,1,8])
    w2 = tf.get_variable('w2',[2,2,8,16])

    z1 = tf.nn.conv2d(X_,w1,strides = [1,1,1,1],padding = "SAME")
    a1 = tf.nn.relu(z1)
    p1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

    z2 = tf.nn.conv2d(p1,w2,strides = [1,1,1,1],padding = "SAME")
    a2 = tf.nn.relu(z2)
    p2 = tf.nn.max_pool(a2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
    p2 = tf.contrib.layers.flatten(p2)
    z3 = tf.contrib.layers.fully_connected(p2,num_outputs = 10,activation_fn = None)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "Tensorflow_CNN_model/Tensorflow_CNN_model.ckpt")
        print("Model restored.")
        predict = tf.argmax(z3, 1)  
        predict_num = sess.run(predict, feed_dict={X: mnist.train.images})
        print(predict_num)

    # tf.argmax(input, axis=None, name=None, dimension=None)
    # 此函数是对矩阵按行或列计算最大值

    # 参数
    # input：输入Tensor
    # axis：0表示按列，1表示按行
    # name：名称
    # dimension：和axis功能一样，默认axis取值优先。新加的字段
    # 返回：Tensor  一般是行或列的最大值下标向量
    # sess.run(tf.argmax) 返回一个列表：元素由行或列的最大值下标构成   

if __name__ == "__main__":
    load_CNN_Model()

