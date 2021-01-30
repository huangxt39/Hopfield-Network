# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
# torch.Size([100, 25, 512]) torch.Size([100, 46, 512])

source, target, s_mean = torch.load('./tensor.pt')
for i in range(5):   
    s = source[i].numpy()
    t = target[i].numpy()
    plt.subplot(1,2,1)
    plt.imshow(s.reshape(26,16), cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(t.reshape(46,16), cmap='gray')
    plt.show()
    
reproduce=True
if not os.path.exists('./weight.pt') or reproduce:
    weight = torch.matmul(target.transpose(0,1), source)
    print('='*10)   
    torch.save((weight,s_mean), './weight.pt')
else:
    weight, s_mean = torch.load('./weight.pt')
    
def retrieve(weight, vector, theta=0.0):

    retrieved = (weight * vector.unsqueeze(0)).sum(dim=1)
    vector = torch.where(retrieved > torch.tensor(theta), torch.tensor(1), torch.tensor(-1))
    
    return vector

num = source.size(0)
retrieved = []
for i in range(num):
    retrieved.append( retrieve(weight, source[i]) )
retrieved = torch.stack(retrieved)
retrieve_rate = ((retrieved - target) == 0).sum().item() / target.numel()
print(retrieve_rate)
    
