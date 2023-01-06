import cv2
import numpy as np
import os

from torch import optim, nn
import  torch
import efficient
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.feature import hog
import bagOfWords
import torch.nn.functional as F
from collections import OrderedDict
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd

import zipfile

#import torch.nn
class con_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(con_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class googlenet(nn.Module):
    def __init__(self, in_channels=3, classes=6):
        super(googlenet, self).__init__()
        self.conv1 = con_block(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = con_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inc1 = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inc2 = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inc3 = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inc4 = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inc5 = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inc6 = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inc7 = Inception_block(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inc8 = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inc9 = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, classes)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.maxpool3(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.inc5(x)
        x = self.inc6(x)
        x = self.inc7(x)

        x = self.maxpool4(x)
        x = self.inc8(x)
        x = self.inc9(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = con_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            con_block(in_channels, red_3x3, kernel_size=1),
            con_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            con_block(in_channels, red_5x5, kernel_size=1),
            con_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            con_block(in_channels, out_1x1pool, kernel_size=1)

        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


device = "cuda"
model = googlenet().to(device)

model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

model.fc1 = nn.Linear(in_features=1024, out_features=2, bias=True)

model.load_state_dict( torch.load("lastSiames_InceptionWieghts.pth"))
