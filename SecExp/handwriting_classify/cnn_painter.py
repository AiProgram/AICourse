import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing
import tensorflow as tf
# import dataset

thisFolder=os.path.dirname(os.path.abspath(__file__))
drawing = False # true if mouse is pressed
def Normalization(x):
    return [2*(i-np.min(x[0]))/(np.max(x[0])-np.min(x[0]))-1 for i in x[0]]
def drawNum(sess,predict,X):
  ix,iy = -1,-1
  def nothing(x):
    pass
  # mouse callback function
  def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,image
    b = param[0]
    g = param[1]
    r = param[2]
    shape = param[3]
    image=param[4]
    if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
      if drawing == True:
        if shape == 0:
          cv2.rectangle(img,(ix,iy),(x,y),(g,b,r),-1)
        else:
          cv2.circle(img,(x,y),5,(g,b,r),-1)
    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      if shape == 0:
        cv2.rectangle(img,(ix,iy),(x,y),(g,b,r),-1)
      else:
        cv2.circle(img,(x,y),5,(g,b,r),-1)
    elif event==cv2.EVENT_RBUTTONDOWN:
        scaledImage=cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        #scaledImage = cv2.filter2D(scaledImage, -1, kernel=kernel)
        img_gray = cv2.cvtColor(scaledImage,cv2.COLOR_RGB2GRAY)
        #img_gray=255-img_gray#image.shape表示图像的尺寸和通道信息(高,宽,通道)
        med = Normalization(np.reshape(img_gray, (1, 784)))
        img_scaled = np.reshape(med, (1, 784))
        # print(type(img_scaled))
        predict_num=sess.run(predict, feed_dict={X: img_scaled})
        print(predict_num[0])
        #cv2.imwrite("handwriting.png",img_gray)

  # Create a black image, a window
  img = np.zeros((256,256,3), np.uint8)
  cv2.namedWindow('image')
  cv2.resizeWindow('image',640,640)

  switch1 = '0 : OFF \n1 : ON'
  switch2 = '0: Rectangle \n1: Line '
  cv2.createTrackbar(switch1, 'image',0,1,nothing)
  #cv2.createTrackbar(switch2, 'image',0,1,nothing)
  while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    # get current positions of four trackbars
    if k == 27:
        break
    r = 255
    g = 255
    b = 255
    shape = 1
    s = cv2.getTrackbarPos(switch1,'image')
    if s == 0:
      img[:] = 0
    else:
      if k == 27:
        break
      cv2.setMouseCallback('image',draw_circle,(b,g,r,shape,img))
  cv2.destroyAllWindows()

def get_model():
  # data_train,data_test,label_train,label_test = dataset.get_our_data()
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

  optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)

  init = tf.global_variables_initializer()

  # sess = tf.Session()

  # saver = tf.train.Saver()  # defaults to saving all variables

  # sess.run(init)

  # epoch = 3000
  # batch_size = 20

  # step = []
  # loss1 = []

  # start = time.perf_counter()
  # for i in range(epoch):
  #     #x,y = mnist.train.next_batch(batch_size)
  #     x,y=data_train,label_train
  #     _, loss = sess.run([optimizer, cost],feed_dict ={X : x, Y : y})
  #     if i %100 == 0:
  #         print("step %d, cost %f" % (i, loss))
  #         step.append(i)
  #         loss1.append(loss)
  # end = time.perf_counter()
  # plt.plot(step,loss1)
  # plt.show()
  """ 保存模型
  saver.save(sess, 'C:/Users/Administrator/Desktop/CNN/models/model.ckpt')
  """ 
  #恢复模型
  saver = tf.train.Saver()
  sess=tf.Session()
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  saver.restore(sess, thisFolder+'/CNN_model.ckpt')#这里使用了之前保存的模型参数
  # print ("Model restored.")
  
  # 保存模型参数，注意把这里改为自己的路径
  predict=tf.argmax(z3,1)
  #saver.save(sess, thisFolder+'/CNN_model.ckpt')
  #correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(Y, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # sess.run(accuracy, feed_dict={X: mnist.train.images, Y:mnist.train.labels})
  # print("training accuracy:", 100*sess.run(accuracy, feed_dict={X: mnist.train.images, Y:mnist.train.labels}),'%')
  #print("Tensorflow CNN model Accuracy:", 100*sess.run(accuracy, feed_dict={X: data_test, Y:label_test}),'%')
  #print(sess.run(predict, feed_dict={X: data_test}))
  #print("运行时间：{:.3f}分钟".format((end - start)/60))
  drawNum(sess,predict,X)
  sess.close()

if __name__=="__main__":
  get_model()
  #drawNum()