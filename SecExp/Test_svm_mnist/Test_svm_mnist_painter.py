import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing

thisFolder=os.path.dirname(os.path.abspath(__file__))
drawing = False # true if mouse is pressed
def Normalization(x):
    return [2*(i-np.min(x[0]))/(np.max(x[0])-np.min(x[0]))-1 for i in x[0]]
def drawNum():
  clf=joblib.load(thisFolder+"/mnist_svm_model.model")
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
        img_scaled=Normalization(np.reshape(img_gray,(1,784)))
        predict=clf.predict(np.reshape(img_scaled,(1,784)))
        print("SVM预测值：",int(predict[0]))
        cv2.imwrite("handwriting.png",img_gray)

  # Create a black image, a window
  img = np.zeros((256,256,3), np.uint8)
  cv2.namedWindow('image')
  cv2.resizeWindow('image',640,640)
  # create trackbars for color change
  cv2.createTrackbar('R','image',0,255,nothing)
  cv2.createTrackbar('G','image',0,255,nothing)
  cv2.createTrackbar('B','image',0,255,nothing)
  # create switch for ON/OFF functionality
  switch1 = '0 : OFF \n1 : ON'
  switch2 = '0: Rectangle \n1: Line '
  cv2.createTrackbar(switch1, 'image',0,1,nothing)
  cv2.createTrackbar(switch2, 'image',0,1,nothing)
  while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    # get current positions of four trackbars
    if k == 27:
        break
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    shape = cv2.getTrackbarPos(switch2,'image')
    s = cv2.getTrackbarPos(switch1,'image')
    if s == 0:
      img[:] = 0
    else:
      if k == 27:
        break
      cv2.setMouseCallback('image',draw_circle,(b,g,r,shape,img))
  cv2.destroyAllWindows()

if __name__=="__main__":
  drawNum()