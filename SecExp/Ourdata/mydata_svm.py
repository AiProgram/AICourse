import numpy as np
import data.ourdataset as datasrc
# Third-party libraries
from sklearn import svm
from sklearn.externals import joblib
#from config import Config
import os

# 返回当前python执行脚本的执行路径
thisFolder = os.path.dirname(os.path.abspath(__file__))
def svm_baseline(store_error_pic=False,c=1):
    x_train,x_test,y_train,y_test = datasrc.get_our_data() # 读取ourdata数据集
    clf = svm.SVC(C=c, kernel='poly')  #计算不同核函数和错误惩罚参数对准确率的影响。
    # 将y_train,y_test由one-hot编码改成标签数字0-9
    y_train = [np.argmax(row) for row in y_train]
    y_test = [np.argmax(row) for row in y_test]
    # 训练模型
    clf.fit(x_train, y_train)
    # 保存模型
    joblib.dump(clf, thisFolder+'/model/svm_model.model')
    # 测试
    predictions = [int(a) for a in clf.predict(x_test)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, y_test))
    # 输出分类正确的数目和准确率
    # print("%s of %s values correct." % (num_correct, len(y_test)))
    # 输出准确率
    score = (num_correct/len(y_test)) * 100
    print("Ourdata——The Accuracy of SVM:{:.2f}%".format(score))

    
if __name__ == "__main__":
    svm_baseline(c=20)
        
