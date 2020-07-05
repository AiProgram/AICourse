import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import data.ourdataset as datasrc
import time
import pandas as pd
import os

# 返回当前python执行脚本的执行路径
thisFolder = os.path.dirname(os.path.abspath(__file__))

x_train,x_test,y_train,y_test = datasrc.get_our_data() # 读取ourdata数据集

# 将y_train,y_test由one-hot编码改成标签数字0-9
y_train = [np.argmax(row) for row in y_train]
y_test = [np.argmax(row) for row in y_test]

# 1 建立朴素贝叶斯分类器模型
gb = GaussianNB()

start = time.perf_counter()

gb.fit(x_train, y_train)
end = time.perf_counter()
# 保存模型
joblib.dump(gb, thisFolder+'/model/NaiveBayes_model.model')
# clf.fit(X,y)参数的形状：
# X：训练矢量{array-like，sparse matrix}，shape（n_samples，n_features）。
# y：相对于X数组，形状（n_samples，）的目标向量。
# 所以minist不能用one_hot

score= gb.score(x_test, y_test)
print("Ourdata——The Accuracy of NaiveBayes:{:.2f}%".format(score))