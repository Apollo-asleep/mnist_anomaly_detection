import math
import torch
import os
import torch.nn as nn

from utils import dataLoader



class ourmethod_kmeans:
    def __init__(self) -> None:
        DATA_PATH = './DataSets'
        MODEL_PATH = './Models'
        MDDEL_NAME = '/OUR_MNIST.pkl'
        DEVICE_NUM = 4
        BATCH_SIZE = 1
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu",DEVICE_NUM)
        self.anomalynumber = 0 
        self.featurelist = []
        train_set = dataLoader.loadTrain_set(DATA_PATH)
        train_data = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        model = torch.load(MODEL_PATH + MDDEL_NAME).to(DEVICE)
        for batch_idx, (images, labels) in enumerate(train_data):
            images = images.to(DEVICE)  # BATCH_SIZE*28*28
            labels = labels.to(DEVICE)  # BATCH_SIZE*1
            outputs = model(images)
            print(labels[batch_idx])
            
            if labels[batch_idx]!= self.anomalynumber:
                self.featurelist.append(outputs)
        print(self.featurelist)

    def kmeans(self):

        return 0