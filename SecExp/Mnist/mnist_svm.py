import numpy as np
# Third-party libraries
from sklearn import svm
from sklearn.externals import joblib
import time
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=False)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

def svm_baseline(store_error_pic=False, c=1):
    clf = svm.SVC(C=c, kernel='poly')  #计算不同核函数和错误惩罚参数对准确率的影响。
    # 训练模型
    start = time.perf_counter()
    clf.fit(x_train, y_train)
    end = time.perf_counter()
    # 测试
    predictions = [int(a) for a in clf.predict(x_test)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, y_test))
    # 输出分类正确的数目和准确率
    # print("%s of %s values correct." % (num_correct, len(y_test)))
    # 输出准确率
    score = (num_correct/len(y_test)) * 100
    print("Mnist——The Accuracy of SVM:{:.2f}%".format(score))
    print("Running time：{:.3f}minute".format((end - start)/60))
    
if __name__ == "__main__":
    svm_baseline(c=20)
        
