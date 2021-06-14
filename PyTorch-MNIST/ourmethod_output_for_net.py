import math
import torch
import os
import torch.nn as nn
from utils import dataLoader
from ourmethod import ourmethod_result_net


DATA_PATH = './DataSets'
MODEL_PATH = './Models'
MDDEL_NAME = '/OUR_MNIST_net_result.pkl'
# MDDEL_NAME = '/OUR_MNIST.pkl'
DEVICE_NUMBER = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu",DEVICE_NUMBER)
BATCH_SIZE = 512
EPOCH = 1
anomalynumber = 0

print('Loading train set...')
train_set = dataLoader.loadTrain_set(DATA_PATH)
train_data = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
BATCH_NUM = math.ceil(len(train_set)/BATCH_SIZE)
print('Using ', DEVICE)

# 建立模型并载入设备
#CNN
# model_nodevice = ourmethod.ourmethod().to(DEVICE)
model_nodevice = ourmethod_result_net.ourmethod()
pretrained_dict = torch.load(MODEL_PATH + '/OUR_MNIST.pkl').state_dict()
model_dict = model_nodevice.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_nodevice.load_state_dict(model_dict)
# model = model_nodevice.to(DEVICE)
model = model_nodevice
print('Saving the model...')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
torch.save(model, MODEL_PATH + MDDEL_NAME)