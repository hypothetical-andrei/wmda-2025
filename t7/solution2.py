# Exercise 2: Transfer Learning with ResNet
import torch
import torch.nn as nn
from torchvision import models

# TODO

x = torch.randn(1, 3, 224, 224)
logits = resnet(x)
print("Output shape:", logits.shape)  # torch.Size([1, 2])
