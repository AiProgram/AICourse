import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name = 'X')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')
X_ = tf.reshape(X, [-1, 28, 28, 1])

tf.set_random_seed(1)

w1 = tf.get_variable('w1',[4,4,1,8],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
w2 = tf.get_variable('w2',[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed = 0))

z1 = tf.nn.conv2d(X_,w1,strides = [1,1,1,1],padding = "SAME")
a1 = tf.nn.relu(z1)
p1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

z2 = tf.nn.conv2d(p1,w2,strides = [1,1,1,1],padding = "SAME")
a2 = tf.nn.relu(z2)
p2 = tf.nn.max_pool(a2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
p2 = tf.contrib.layers.flatten(p2)
z3 = tf.contrib.layers.fully_connected(p2,num_outputs = 10,activation_fn = None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z3,labels = Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

saver = tf.train.Saver()  # defaults to saving all variables

sess.run(init)

epoch = 5000
batch_size = 800

step = []
loss1 = []

start = time.perf_counter()
for i in range(epoch):
    x, y = mnist.train.next_batch(batch_size)
    _, loss = sess.run([optimizer, cost],feed_dict ={X : x, Y : y})
#     if i %100 == 0:
#         print("step %d, cost %f" % (i, loss))
#         step.append(i)
#         loss1.append(loss)
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
# saver.save(sess, './Tensorflow_CNN_model/Tensorflow_CNN_model.ckpt')
correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(accuracy, feed_dict={X: mnist.train.images, Y:mnist.train.labels})
# print("training accuracy:", 100*sess.run(accuracy, feed_dict={X: mnist.train.images, Y:mnist.train.labels}),'%')
score = 100*sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels})
print("Mnist——The Accuracy of CNN:{:.2f}%".format(score))
print("Running time：{:.3f}minute".format((end - start)/60))
sess.close()