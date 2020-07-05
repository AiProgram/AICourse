import tensorflow as tf
import time
import data.ourdataset as datasrc
import os

data_train,data_test,label_train,label_test = datasrc.get_our_data()
thisFolder=os.path.dirname(os.path.abspath(__file__))


# X = tf.placeholder(tf.float32, [None, 784], name = 'X')
# Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')

# tf.set_random_seed(1)
# w1 = tf.get_variable('w1', [30,784], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
# b1 = tf.get_variable('b1', [30,1], initializer = tf.zeros_initializer())
# w2 = tf.get_variable('w2', [20,30], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
# b2 = tf.get_variable('b2', [20,1], initializer = tf.zeros_initializer())
# w3 = tf.get_variable('w3', [10,20], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
# b3 = tf.get_variable('b3', [10,1], initializer = tf.zeros_initializer())

# z1 = tf.add(tf.matmul(w1,tf.transpose(X)), b1)
# a1 = tf.nn.relu(z1)
# z2 = tf.add(tf.matmul(w2,a1), b2)
# a2 = tf.nn.relu(z2)
# z3 = tf.add(tf.matmul(w3,a2), b3)

X_bp = tf.placeholder(tf.float32, [None, 784], name = 'X_bp')
Y_bp = tf.placeholder(tf.float32, [None, 10], name = 'Y_bp')

tf.set_random_seed(1)
w_1 = tf.get_variable('w_1', [30,784])
b_1 = tf.get_variable('b_1', [30,1])
w_2 = tf.get_variable('w_2', [20,30])
b_2 = tf.get_variable('b_2', [20,1])
w_3 = tf.get_variable('w_3', [10,20])
b_3 = tf.get_variable('b_3', [10,1])

z_1 = tf.add(tf.matmul(w_1,tf.transpose(X_bp)), b_1)
a_1 = tf.nn.relu(z_1)
z_2 = tf.add(tf.matmul(w_2,a_1), b_2)
a_2 = tf.nn.relu(z_2)
z_3 = tf.add(tf.matmul(w_3,a_2), b_3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(z_3), labels = Y_bp))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

saver_bp = tf.train.Saver()  # defaults to saving all variables

sess.run(init)

epoch = 1000
batch_size = 20

step = []
loss1 = []

start = time.perf_counter()
for i in range(epoch):
    #x,y = mnist.train.next_batch(batch_size)
    x,y=data_train,label_train
    for j in range(int(len(data_train)/batch_size)):
        if j==len(data_train)/batch_size-1:
            x=data_train[batch_size*j:]
            y=label_train[batch_size*j:]
        else:
            x=data_train[batch_size*j:batch_size*(j+1)]
            y=label_train[batch_size*j:batch_size*(j+1)]
        _, loss = sess.run([optimizer, cost],feed_dict ={X_bp : x, Y_bp : y})
    #_, loss = sess.run([optimizer, cost],feed_dict ={X : data_train, Y : label_train})
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
saver_bp.save(sess, thisFolder+'/model/BP_model.ckpt')
correct_prediction = tf.equal(tf.argmax(tf.transpose(tf.sigmoid(z_3)), 1), tf.argmax(Y_bp, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
score = sess.run(accuracy, feed_dict={X_bp: data_test, Y_bp:label_test}) * 100
print("Ourdata——The Accuracy of BP:{:.2f}%".format(score))
sess.close()