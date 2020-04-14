# -*- coding: UTF-8 -*-
import torch as t
from PIL import Image
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from ga import GaNet, GsNet
import os
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
log_dir1 = '/pytorch_test_py3.6/pth/net1.pth'
log_dir2 = '/pytorch_test_py3.6/pth/net2.pth'
# 定义对数据的预处理
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize(norm_mean, norm_std),
])   # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])   # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量


# 训练集
img_data = tv.datasets.ImageFolder('D:/256_ObjectCategories', transform=train_transform)
train_data_loader = t.utils.data.DataLoader(img_data, batch_size=4, shuffle=True, num_workers=0)

net1= GaNet().cuda()
net2 = GsNet().cuda()
print(net1)
print(net2)
from torch import optim
criterion = nn.MSELoss()
# optimizer = optim.SGD(net1.parameters(),net1.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD([
                {'params': net1.parameters(), 'lr': 1e-4},
                {'params': net2.parameters(), 'lr': 1e-4}
            ], momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times
running_loss = 0.0
epoch = 0
for i, data in enumerate(train_data_loader,0):
    # get the inputs
    inputs, labels = data
    inputs = inputs.cuda()
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs1 = net1(inputs)
    outputs = net2(outputs1)
    loss = criterion(outputs, inputs)
    loss = loss.cuda()
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    if i % 500 == 499:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 500))
        running_loss = 0.0
# state = {'model': net1.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
t.save(net1, log_dir1)
# t.save(net1.state_dict(), log_dir1)
t.save(net2, log_dir2)
# t.save(net2.state_dict(), log_dir2)
print('Finished Training')


