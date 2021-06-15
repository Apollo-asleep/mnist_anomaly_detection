from types import AsyncGeneratorType
# from matplotlib.pyplot import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from tqdm import tqdm
from utils import dataLoader 
from sklearn.cluster import KMeans
from sklearn import svm

DATA_PATH = './DataSets'
MODEL_PATH = './Models'
BATCH_SIZE = 512
DEVICE = torch.device("cpu")

print('Loading train set...')

train_set = dataLoader.loadTrain_set(DATA_PATH)
train_data = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)

print('Using ', DEVICE)

print('Loading saved model...')
model = torch.load(MODEL_PATH + '/OUR_MNIST_net_result.pkl').to(DEVICE)

flags = False

for image,label in tqdm(train_data,leave=False):
    image = image.to(DEVICE)
    out = model(image)
    # print("out")
    # print(out.shape)
    out = out.view(BATCH_SIZE,-1)
    
    # print(out.shape)
    if flags == False:
        a = out
        flags = True
    else:

        a = torch.cat((a,out),dim=0)

b = a.cpu().detach().numpy()

kmeans = KMeans(n_clusters=9)
kmeans.fit(b)
y_kmeans = kmeans.predict(b)

# d = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
# for i in y_kmeans:
#     d[i] = d[i] + 1

# print(d)

svms=[]
for i in range(9):
    svms.append(svm.SVC(C=5, gamma=0.05,max_iter=20))

train_label = y_kmeans
for i in range(9):
    for j in range(len(y_kmeans)):
        if y_kmeans[j] == i:
            train_label[j] = 1
        else:
            train_label[j] = 0
    
    svms[i].fit(b, train_label)

