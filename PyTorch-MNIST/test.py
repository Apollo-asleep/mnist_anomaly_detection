import torch

from utils import dataLoader

DATA_PATH = './DataSets'
MODEL_PATH = './Models'
BATCH_SIZE = 512
DEVICE = torch.device("cuda" ,4)

print('Loading test set...')
test_set = dataLoader.loadTest_set(DATA_PATH)
test_data = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
print('Using ', DEVICE)
print('Loading saved model...')
model = torch.load(MODEL_PATH + '/MyCNN_MNIST.pkl').to(DEVICE)
print('Testing...')
anomalynumber = 0
num_correct = 0
for images, labels in test_data:
    #异常数字anomalynumber
    tensornum = labels.numel()
    for i in range(tensornum):
        if labels[i] == anomalynumber:
            labels[i] = 1
        else:
            labels[i] = 0    
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(images)
    pred = torch.max(outputs, 1)[1]
    num_correct += (pred == labels).sum().item()
print('Accuracy: {:.6f}%'.format(100 * num_correct / len(test_set)))
