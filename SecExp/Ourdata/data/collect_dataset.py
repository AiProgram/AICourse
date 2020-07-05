import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing

thisFolder=os.path.dirname(os.path.abspath(__file__))
drawing = False # true if mouse is pressed
def Normalization(x):
  #归一化到图片像素到[-1,1]，同收集数据
    return [2*(i-np.min(x[0]))/(np.max(x[0])-np.min(x[0]))-1 for i in x[0]]
def drawNum():
  #clf=joblib.load(thisFolder+"/svm_model.model")
  ix,iy = -1,-1
  def nothing(x):
    pass
  # mouse callback function
  def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,image
    num = param[0]#当前收集数字正确标签
    image = param[1]#收集图像
    data=param[2]#准备存到npy文件的数据
    label=param[3]
    #笔画置为白色，背景黑色
    r=255
    g=255
    b=255
    #线条形状是线状
    shape=1
    if event == cv2.EVENT_LBUTTONDOWN:
      #鼠标左键按下，正在写字
      drawing = True
      ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
      #鼠标移动，在写字状态下记录笔画
      if drawing == True:
        if shape == 0:
          cv2.rectangle(img,(ix,iy),(x,y),(g,b,r),-1)
        else:
          cv2.circle(img,(x,y),5,(g,b,r),-1)
    elif event == cv2.EVENT_LBUTTONUP:
      #鼠标左键放开，退出写字状态
      drawing = False
      #记录最后一个笔画
      if shape == 0:
        cv2.rectangle(img,(ix,iy),(x,y),(g,b,r),-1)
      else:
        cv2.circle(img,(x,y),5,(g,b,r),-1)
    elif event==cv2.EVENT_RBUTTONDOWN:
      #鼠标右键按下，写入收集的数据
        index=int(label[0][0])
        #图片缩小到28*28大小
        scaledImage=cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        #scaledImage = cv2.filter2D(scaledImage, -1, kernel=kernel)
        #图片灰度化，不再有RGB通道，只有一个灰度通道
        img_gray = cv2.cvtColor(scaledImage,cv2.COLOR_RGB2GRAY)
        #img_gray=255-img_gray#image.shape表示图像的尺寸和通道信息(高,宽,通道)
        #重排矩阵大小，避免参数出错
        image_array=np.reshape(img_gray,(1,784))
        #若像素最大值等于最小值，说明图片全是黑色，没有数字，不保存。这也是避免错误按下右键
        if np.max(image_array)==np.min(image_array):
            return
        #记录图像
        data[index+1]=image_array
        #记录标签
        label[index+1][num]=1
        label[0][0]+=1
        #清空画板，准备下一个数字收集
        image[:]=0
        #输出已经收集的数字信息
        print("picture: "+str(label[0][0])+", tag: "+str(num))

  # Create a black image, a window
  #创建一个256*256彩色画板图像
  img = np.zeros((256,256,3), np.uint8)
  cv2.namedWindow('image')
  #设置整个窗口显示大小
  cv2.resizeWindow('image',640,540)
  #创建标签滑块
  cv2.createTrackbar('number','image',0,9,nothing)
  # create switch for ON/OFF functionality
  switch1 = '0 : OFF \n1 : ON'
  switch2 = '0: Rectangle \n1: Line '
  cv2.createTrackbar(switch1, 'image',0,1,nothing)
  data,label=read_data()
  while(1):
    #显示画板
    cv2.imshow('image',img)
    #等待键盘按下键，在esc键按下后退出
    k = cv2.waitKey(1) & 0xFF
    # get current positions of four trackbars
    if k == 27:#esc键
      #退出时再统一写入数据集文件，避免重复多次写入
        store_data(data,label)
        break
    #r = cv2.getTrackbarPos('R','image')
    #g = cv2.getTrackbarPos('G','image')
    #b = cv2.getTrackbarPos('B','image')
    #shape = cv2.getTrackbarPos(switch2,'image')
    #s = cv2.getTrackbarPos(switch1,'image')
    num=cv2.getTrackbarPos('number','image')
    r=255
    g=255
    b=255
    shape=1
    s=1
    if s == 0:
      #s在这里无效
      img[:] = 0
    else:
      if k == 27:
        store_data(data,label)
        break
      #为画板设置鼠标事件
      cv2.setMouseCallback('image',draw_circle,(num,img,data,label))
  #退出，关闭窗口
  cv2.destroyAllWindows()

def read_data():
    """读取已有数据
    """
    data=np.zeros((3000,784))
    label=np.zeros((3000,10))
    label[0][0]=0#标记空数据集大小
    #如果有数据文件存在则读取，否则跳过
    if os.path.exists(thisFolder+"/number_data.npy"):
       data=np.load(thisFolder+"/number_data.npy")
    if os.path.exists(thisFolder+"/label_data.npy"):
        label=np.load(thisFolder+"/label_data.npy")
    return data,label
def store_data(data,label):
  #保存数据集
    np.save(thisFolder+"/number_data.npy",data)
    np.save(thisFolder+"/label_data.npy",label) # 存的数据是one-hot编码

if __name__=="__main__":
    drawNum()