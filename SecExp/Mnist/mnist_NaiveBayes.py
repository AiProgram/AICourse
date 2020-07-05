import numpy as np
import time
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=False)

train_num = 1000
test_num = 100
class_num = 10
desimon = 784

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# train_num = len(x_train)
# test_num = len(x_test)

prediction = []
start = time.perf_counter()
for i in range(test_num):
    test = x_test[i] # 第i条的属性 1*784
    class_rate = []
    # 求每一个类别的概率，这里MNIST数据集共有10个类别
    for j in range(class_num):
        # 找到样本中类别是j的下标
        class_is_j_index = np.where(y_train[:train_num] == j)[0]
        # 类别是j的比率
        j_rate = len(class_is_j_index)/len(y_train)
        # 如果用拉普拉斯修正
        # j_rate = (len(class_is_j_index)+1)+/(len(y_train)+class_num)
        # 取出类别是j的样本
        class_is_j_x = np.array([x_train[x] for x in class_is_j_index])
        # 遍历每个维度
        for k in range(desimon): # 计算每个属性相同的概率
            # 找到j类样本集中该维度下的值与测试样本中该维度的值的差小于0.8的样本，并求占j类样本的比率，与j_rate依次相乘
            # 这里我规定的界限是0.8，因为MNIST中样本数字在0到1之间，并且是两端分布，要么是0，要么接近1。
            count = 0.0
            for item in class_is_j_x:
                if np.fabs(np.fabs(item[k] - test[k]) < 0.8):
                    count = count + 1
            j_rate *= (count / len(class_is_j_x))
            # 如果用拉普拉斯修正，公式如下
            # j_rate *= ((count+1) / (len(class_is_j_x)+每个属性可能的取值数))
            # 求解每个属性可能的取值数
                # set是集合，去掉重复的项
                # set1 = set(class_is_j_x[:,k])
                # 每个属性可能的取值数
                # len1 = len(set1)
            # j_rate *= len([item for item in class_is_j_x if np.fabs(item[k] - test[k]) < 0.8])*1.0 / len(class_is_j_x)
        class_rate.append(j_rate) # 得出每个测试样本的标签是j的概率
        # 列表是0的概率，1的概率，2的概率......
    # 找到贝叶斯预测值最大的类别，作为该测试的预测类别，放到结果集中
    prediction.append(np.argmax(class_rate))
    # print(i, 'prediction:', prediction[-1], 'actual:', y_test[i])
end = time.perf_counter()
score= (np.sum(np.equal(prediction, y_test[:test_num])) / test_num) * 100
print("Mnist——The Accuracy of NaiveBayes:{:.2f}%".format(score))
print("Running time：{:.3f}minute".format((end - start)/60))