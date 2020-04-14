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
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
log_dir1 = '/pytorch_test_py3.6/pth/net1.pth'
log_dir2 = '/pytorch_test_py3.6/pth/net2.pth'

# 定义对数据的预处理

# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     # transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     # transforms.Normalize(norm_mean, norm_std),
# ])   # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量
# processing test data
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])   # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量
img_data = tv.datasets.ImageFolder('D:/256_ObjectCategories', transform=test_transform)
train_data_loader = t.utils.data.DataLoader(img_data, batch_size=1, shuffle=True, num_workers=0)

#tensor -> PIL_image
unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
# loading GaNet as model1 and GsNet as model2
model1 = t.load(log_dir1)
# model1.load_state_dict(t.load(log_dir1))
model2 = t.load(log_dir2)
# model2.load_state_dict(t.load(log_dir2))

for i, data in enumerate(train_data_loader, 0):
    inputs, labels = data
    if i == 5:
        break
    #image_original = tensor_to_PIL((inputs + 1) / 2)
    image_original = tensor_to_PIL(inputs)
    image_original.save("/home/256_origin/" +  str(i)  + ".jpg")
    image1 = model1(inputs.to(device))
    image2 = model2(image1.to(device))
    #image = tensor_to_PIL((image2+1) / 2)
    image = tensor_to_PIL(image2)
    image.save("/home/256_reconstruct/" + str(i)  + ".jpg")
    print("Iamge-{} is ok".format(i))
