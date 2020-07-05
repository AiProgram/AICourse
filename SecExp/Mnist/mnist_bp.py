import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)


X = tf.placeholder(tf.float32, [None, 784], name = 'X')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')

tf.set_random_seed(1)
w1 = tf.get_variable('w1', [30,784], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
b1 = tf.get_variable('b1', [30,1], initializer = tf.zeros_initializer())
w2 = tf.get_variable('w2', [20,30], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
b2 = tf.get_variable('b2', [20,1], initializer = tf.zeros_initializer())
w3 = tf.get_variable('w3', [10,20], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
b3 = tf.get_variable('b3', [10,1], initializer = tf.zeros_initializer())

z1 = tf.add(tf.matmul(w1,tf.transpose(X)), b1)
a1 = tf.nn.relu(z1)
z2 = tf.add(tf.matmul(w2,a1), b2)
a2 = tf.nn.relu(z2)
z3 = tf.add(tf.matmul(w3,a2), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(z3), labels = Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

saver = tf.train.Saver()  # defaults to saving all variables

sess.run(init)

epoches = 10000
batch_size = 800

step = []
loss1 = []

start = time.perf_counter()
for i in range(epoches):
    x, y = mnist.train.next_batch(batch_size)
    _, loss = sess.run([optimizer, cost],feed_dict ={X : x, Y : y})
    # if i %100 == 0:
    #     print("step %d, cost %f" % (i, loss))
    #     step.append(i)
        # loss1.append(float(loss))
end = time.perf_counter()
# plt.plot(step,loss1)
# plt.show()
""" 保存模型
saver.save(sess, 'C:/Users/Administrator/Desktop/CNN/models/model.ckpt')
""" 
""" 恢复模型
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "C:/Users/Administrator/Desktop/CNN/models/model.ckpt")#这里使用了之前保存的模型参数
    print ("Model restored.")
"""
# 保存模型参数，注意把这里改为自己的路径
# saver.save(sess, './Tensorflow_BP_model/Tensorflow_BP_model.ckpt')
correct_prediction = tf.equal(tf.argmax(tf.transpose(tf.sigmoid(z3)), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
score = 100 * sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels})
print("Mnist——The Accuracy of BP:{:.2f}%".format(score))
print("Running time：{:.3f}minute".format((end - start) / 60))
sess.close()