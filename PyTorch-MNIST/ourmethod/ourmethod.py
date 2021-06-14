import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim



class ourmethod(nn.Module):
    def __init__(self):
        # 创建一个pytorch神经网络模型
        super(ourmethod, self).__init__()
        # 卷积层1，32通道输出，卷积核大小3*3，步长1*1，padding为1
        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        # 最大值池化，核大小2*2，步长2*2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.Conv4 = nn.ConvTranspose2d(128,64,6,5,1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.Conv5 = nn.ConvTranspose2d(64,32,5,5,3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.Conv6 = nn.ConvTranspose2d(32,1,4,4,0)
        self.pool6 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(func.relu(self.Conv1(x)))
        x = self.pool2(func.relu(self.Conv2(x)))
        x = self.pool3(func.relu(self.Conv3(x)))
        x = self.pool4(func.relu(self.Conv4(x)))
        x = self.pool5(func.relu(self.Conv5(x)))
        x = self.pool6(func.relu(self.Conv6(x)))
        return x

# from torchsummary import summary
# model = MyCNN()
# summary(model,(1,28,28),512)