模型最终存储在Models/OUR_MNIST_net_result。输出为[batchsize,128,3,3].
可以用torch.view改为（batchsize,784）。784 = 128X3X3。
直接读取Model文件即可



—————————————————————————————————————————————


文件夹ourmethod存储我们的方法里需要的网络模型
分别为
1.ourcnn.py
用于预训练出enoder的两层cnn
2.ourmethod_net.py
encoder+decoder存储在这里
3.ourmethod_result_net.py
提取出ourmethod_net.py中的encoder，成为最终的特征提取器


其他相关.py文件，均在PyTorch-MNIST文件夹下

1.ourmethod_output_for_net.py
最终调用的model用这个文件的代码，实现模型存储功能。
模型最终存储在Models/OUR_MNIST_net_result。输出为[batchsize,128,3,3].
可以用torch.view改为（batchsize,784）。784 = 128X3X3

2.ourmethod_pretrain_cnn.py
预训练enoder的两层cnn

3.outmethod_train.py
训练encoder+decoder
