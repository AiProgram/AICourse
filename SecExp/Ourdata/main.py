import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.externals import joblib
thisFolder = os.path.dirname(os.path.abspath(__file__))
drawing = False  # true if mouse is pressed
ix, iy = -1, -1
g1 = tf.Graph()  # 加载到Session 1的graph
sess1 = tf.Session(graph=g1)  # Session1
g2 = tf.Graph()  # 加载到Session 2的graph
sess2 = tf.Session(graph=g2)  # Session2

###############################################################CNN
with sess2.as_default(): 
    with g2.as_default():
        tf.global_variables_initializer().run()
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
        z3 = tf.contrib.layers.fully_connected(p2, num_outputs=10, activation_fn=None)

##################################################################BP
with sess1.as_default(): 
    with g1.as_default():
        tf.global_variables_initializer().run()
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
        z_3 = tf.add(tf.matmul(w_3, a_2), b_3)
##################################################################SVM        
svm_clf = joblib.load(thisFolder + "/model/svm_model.model")

##################################################################NaiveBayes
NaiveBayes_gb = joblib.load(thisFolder + '/model/NaiveBayes_model.model')

def Normalization(x):
    return [2 * (i - np.min(x[0])) / (np.max(x[0]) - np.min(x[0])) - 1 for i in x[0]]
def nothing(x):
    pass
  # mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,img_scaled
    r = 255
    g = 255
    b = 255
    shape = 1
    img = param
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
        scaledImage=cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        #scaledImage = cv2.filter2D(scaledImage, -1, kernel=kernel)
        img_gray = cv2.cvtColor(scaledImage,cv2.COLOR_RGB2GRAY)
        #img_gray=255-img_gray#image.shape表示图像的尺寸和通道信息(高,宽,通道)
        img_scaled = np.reshape(Normalization(np.reshape(img_gray, (1, 784))), (1, 784))

        cnn_result = pred_CNN_model(img_scaled)
        print("******* CNN prediction: ********", cnn_result)

        bp_result = pred_BP_model(img_scaled)
        print("******* BP prediction: ********", bp_result)

        svm_result = svm_clf.predict(np.reshape(img_scaled,(1,784)))[0]
        print("******* SVM prediction: ********", bp_result)

        NaiveBayes_result = NaiveBayes_gb.predict(np.reshape(img_scaled,(1,784)))[0]
        print("******* NaiveBayes prediction: ********", NaiveBayes_result)


def load_CNN_model():
    global sess2, g2
    with sess2.as_default(): 
        with g2.as_default():
            model_saver = tf.train.Saver()
            # model_saver = tf.train.Saver(tf.global_variables())        
            model_saver.restore(sess2, thisFolder+ '/model/CNN_model.ckpt')

def pred_CNN_model(test_images):
    with sess2.as_default():
        with sess2.graph.as_default():  # 2
            predict = tf.argmax(z3, 1)  
            predict_num = sess2.run(predict, feed_dict={X: test_images})
    #sess1.close()
    return predict_num[0]

def load_BP_model():
    global sees1, g1
    with sess1.as_default(): 
        with g1.as_default():
            model_saver = tf.train.Saver()            
            model_saver.restore(sess1, thisFolder + '/model/BP_model.ckpt')

def pred_BP_model(test_images):
    with sess1.as_default():
        with sess1.graph.as_default():  # 2
            predict_bp = tf.argmax(tf.transpose(tf.sigmoid(z_3)), 1)  
            predict_num = sess1.run(predict_bp, feed_dict={X_bp: test_images}) # 预测
    #sess1.close()
    return predict_num[0]  # 返回预测结果
    

def drawNum():  
    # Create a black image, a window
    load_CNN_model()
    load_BP_model()
    img = np.zeros((256,256,3), np.uint8)
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 640, 640)
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
        # r = 255
        # g = 255
        # b = 255
        # shape = 1
        s = cv2.getTrackbarPos(switch1,'image')
        if s == 0:
            img[:] = 0
        else:
            if k == 27:
                break
            cv2.setMouseCallback('image',draw_circle,(img))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    drawNum()