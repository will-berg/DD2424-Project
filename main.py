import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


model = models.resnet18(pretrained=True)
