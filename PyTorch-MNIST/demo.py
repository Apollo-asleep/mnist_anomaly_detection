import torch

n=torch.FloatTensor(4).fill_(0)
print(n)
print(n[0])
if n[0] == 0:
    n[0] = 10

print(n)
print(n.numel())
