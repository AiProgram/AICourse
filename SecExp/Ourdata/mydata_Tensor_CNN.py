import tensorflow as tf
import time
import data.ourdataset as datasrc
import os

data_train,data_test,label_train,label_test = datasrc.get_our_data()
thisFolder=os.path.dirname(os.path.abspath(__file__))

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

optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

saver = tf.train.Saver()  # defaults to saving all variables

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
        _, loss = sess.run([optimizer, cost],feed_dict ={X : x, Y : y})
    #_, loss = sess.run([optimizer, cost],feed_dict ={X : data_train, Y : label_train})
end = time.perf_counter()

# 保存模型参数，注意把这里改为自己的路径
saver.save(sess, thisFolder+'/model/CNN_model.ckpt')
correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
score = sess.run(accuracy, feed_dict={X: data_test, Y:label_test}) * 100
print("Ourdata——The Accuracy of CNN:{:.2f}%".format(score))
sess.close()