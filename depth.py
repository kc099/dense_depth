# import utils
from data import *
from model import PTModel
import torch.nn.functional as F
import torch
import torch.nn as nn 
import torchvision.transforms as T
from PIL import Image
# import torchvision.transforms as transforms


path = 'Unet_depth20.pth'
model = PTModel()
model.load_state_dict(torch.load(path))
print('Model successfully loaded.')

with open('test.txt', 'r') as f:
    lines = f.readlines()

x = Image.open('/home/krishna/DenseDepth/PyTorch/i0.jpg')
transform = T.Compose([T.ToTensor()])
x = transform(x)
y = model(x.unsqueeze(0))

print(y.shape)
