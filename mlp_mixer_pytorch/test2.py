import torch
img = torch.randn(32, 128, 19)
from torch import nn
from functools import partial

# m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
print(input.size())
output = m(input)
print(output.size())


x = torch.Tensor([1, 2, 3, 4, 5, 6,7,8,9,0,5,6]).view(2,3,2)
y = torch.mean(x, dim=1, keepdim=True)
print(y)