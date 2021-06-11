# 直接调用sklearn下的svm模块
# 模块简介https://www.jianshu.com/p/a9f9954355b3

import torchvision
from sklearn import svm
from sklearn.metrics import confusion_matrix
import torch

DEVICE_NUMBER = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu",DEVICE_NUMBER)
anomalynumber = 4
Dataset_Path = 'D:\C盘桌面文件\个人文件\研究生\高级人工智能\异常检测\mnist_anomaly_detection\PyTorch-MNIST\DataSets'
Relative_Path = 'PyTorch-MNIST\DataSets'

def MNIST_DATASET_TRAIN(downloads, train_amount):
    training_data = torchvision.datasets.MNIST(
              root = Dataset_Path,
              train = True,
              transform = torchvision.transforms.ToTensor(),
              download = downloads
              )
    
    train_data = training_data.train_data.numpy()[:train_amount]
    train_label = training_data.train_labels.numpy()[:train_amount]
    for i in range(len(train_label)):
        if train_label[i]==anomalynumber:
            train_label[i] = 1
        else:
            train_label[i] = 0
    train_data = train_data/255.0
    
    return train_data, train_label

def MNIST_DATASET_TEST(downloads, test_amount):
    testing_data = torchvision.datasets.MNIST(
              root = Dataset_Path,
              train = False,
              transform = torchvision.transforms.ToTensor(),
              download = downloads
              )
    test_data = testing_data.test_data.numpy()[:test_amount]
    test_label = testing_data.test_labels.numpy()[:test_amount]
    for i in range(len(test_label)):
        if test_label[i]==anomalynumber:
            test_label[i] = 1
        else:
            test_label[i] = 0
    test_data = test_data/255.0
    return test_data, test_label

if __name__=='__main__':
    train_data, train_label = MNIST_DATASET_TRAIN(False, 60000)
    test_data, test_label = MNIST_DATASET_TEST(False, 2000)

    training_features = train_data.reshape(60000,-1)
    test_features = test_data.reshape(2000,-1)

    # Training SVM
    print('------Training and testing SVM------')
    clf = svm.SVC(C=5, gamma=0.05,max_iter=50)
    clf.fit(training_features, train_label)
    
    #Test on Training data
    train_result = clf.predict(training_features)
    precision = sum(train_result == train_label)/train_label.shape[0]
    print('Training precision: ', precision)

    #Test on test data
    test_result = clf.predict(test_features)
    precision = sum(test_result == test_label)/test_label.shape[0]
    print('Test precision: ', precision)
    
    #Show the confusion matrix
    matrix = confusion_matrix(test_label, test_result)