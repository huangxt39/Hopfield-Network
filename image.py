# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
#test_data = torchvision.datasets.CIFAR10(root='./cifar',train=False, download=True)

import pickle
with open('./cifar/cifar-10-batches-py/test_batch', 'rb') as fo:
    data_dic = pickle.load(fo, encoding='bytes')
    print(data_dic['data'.encode()].shape)
    im_array = data_dic['data'.encode()]

num_pictures = 10    # 3 perfect 4 cannot
im_array = im_array[:num_pictures].reshape(num_pictures, 3, 32, 32).transpose((0,2,3,1))

#compute weights
transformed_im = np.where(im_array.mean(axis=3) >= 128, 1, 0).reshape(num_pictures, 32*32) #100, 32*32
train_im = transformed_im * 2 - 1
weight = np.matmul(train_im.transpose(), train_im) * (1 - np.identity(32*32))

def retrieve(weight, vector, theta=0.5, time=20000):
    length = len(vector)
    e_lis = []
    for s in tqdm(range(time)):
        i = random.randint(0, length-1)
        u = np.dot(weight[i], vector)
        if u > theta:
            vector[i] = 1
        else:
            vector[i] = -1
        if s % 1000000 == 0:
            energy = (weight * np.matmul(np.expand_dims(vector, 1), np.expand_dims(vector, 0))).sum() * (-0.5) + theta * vector.sum()
            e_lis.append(energy)
    return vector, e_lis
        
rates = []
origin = []
retrieved = []
for i in range(num_pictures):
    origin_im = transformed_im[i]
    origin_im = origin_im * 2 - 1
    corrupted_im = origin_im.copy()
    corrupted_im[:16*32] = 0
    retrieved_im, e_lis = retrieve(weight, corrupted_im.copy())
    rate = ((retrieved_im * origin_im) == 1).sum() / 1024
    rates.append(rate)
    origin.append( (origin_im + 1) / 2 )
    retrieved.append( (retrieved_im + 1) / 2 )

plt.figure(figsize=(num_pictures,2))
for i,retrieved_im in enumerate(retrieved):
    plt.subplot(2,num_pictures,i+1)
    plt.imshow(retrieved_im.reshape(32,32),cmap='gray')
 
for i,origin_im in enumerate(origin):
    plt.subplot(2,num_pictures,i+num_pictures+1)
    plt.imshow(origin_im.reshape(32,32),cmap='gray')
    
plt.show()
print(np.array(rates).mean())

# =============================================================================
# while True:
#     a = int(input('select a picture from 0 - %d'%(num_pictures-1)))
#     if a < 0 or a >= num_pictures:
#         break
#     origin_im = transformed_im[a]
#     corrupted_im = origin_im * 2 - 1
#     corrupted_im[:16*32] = 0
#     retrieved_im, e_lis = retrieve(weight, corrupted_im.copy())
#     corrupted_im = (corrupted_im + 1) / 2
#     retrieved_im = (retrieved_im + 1) / 2
#     
#     plt.figure(figsize=(8,8))
#     plt.subplot(2,2,1)
#     plt.imshow(origin_im.reshape(32,32),cmap='gray')
#     plt.subplot(2,2,2)
#     plt.imshow(corrupted_im.reshape(32,32),cmap='gray')
#     plt.subplot(2,2,3)
#     plt.imshow(retrieved_im.reshape(32,32),cmap='gray')
#     plt.subplot(2,2,4)
#     plt.plot(np.arange(1, len(e_lis)+1), np.array(e_lis))
#     plt.show()
# =============================================================================
    
    

# =============================================================================
# im1 = im_array[1].reshape((3,32,32)).transpose((1,2,0))
# 
# plt.subplot(1,2,1)
# plt.imshow(im1)
# plt.subplot(1,2,2)
# im_trans = np.where(im1.mean(axis=2) >= 128, 255, 0)
# plt.imshow(im_trans,cmap='gray')
# =============================================================================
