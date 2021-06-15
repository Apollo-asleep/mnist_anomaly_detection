from torch.nn import functional as F
import torch
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = F.cosine_similarity(input1, input2)
print(output)
print(output.sum())