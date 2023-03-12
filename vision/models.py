#!/usr/bin/env python3

import torch
from tqdm import tqdm
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as nnF
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvFun
from torch.utils.data import DataLoader, sampler
from torch import optim
from skimage import io, transform
import os


class MultiFilterLayer(nn.Module):
    """Maintains the size, applying 3 filterbanks of different sizes, then do a batch norm, and finally a mixing filter (1x1 convolution) that also subsamples. Generally inspired by the ResNet architecture."""

    def __init__(
        self,
        nchannels_in,
        nchannels_out,
        nonlin,
        norm_layer,
        filterSizes=(3, 5, 7),
        stride=2,
    ):
        super(MultiFilterLayer, self).__init__()
        self.norm = norm_layer(sum(nchannels_out[0:3]))
        self.nonlin = nonlin
        self.conv1 = nn.Conv2d(
            nchannels_in, nchannels_out[0], kernel_size=filterSizes[0], padding="same"
        )
        self.conv2 = nn.Conv2d(
            nchannels_in, nchannels_out[1], kernel_size=filterSizes[1], padding="same"
        )
        self.conv3 = nn.Conv2d(
            nchannels_in, nchannels_out[2], kernel_size=filterSizes[2], padding="same"
        )
        self.conv_next = nn.Conv2d(
            sum(nchannels_out[0:3]), nchannels_out[3], kernel_size=1, stride=stride
        )

    def forward(self, x):
        x = torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), dim=1)
        x = self.norm(x)
        x = self.nonlin(x)
        x = self.conv_next(x)
        return x


class FullyConnected(nn.Module):
    """a fully connected set of layers, where `nodes_per_layer` is a list of the number of nodes in the ith layer for i>0, while the 0th entry is the size of the input. Between each layer is an application of the function `nonlin`."""

    def __init__(self, nodes_per_layer, nonlin=None):
        super(FullyConnected, self).__init__()
        if nonlin is None:
            nonlin = nn.ReLU6()
        self.layers = nn.ModuleList(
            [
                nn.Linear(nodes_per_layer[ii - 1], node)
                for (ii, node) in enumerate(nodes_per_layer)
                if ii > 0
            ]
        )
        self.nonlin = nonlin

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.nonlin(x)
        return nnF.log_softmax(x, dim=1)


class Animal10(nn.Module):
    """a set of multi-filter size convolutional layers, defined by `nchannels_multifilters` and `filterSizes`, followed by a set of fully connected layers, defined by `nfully_connected` (the first entry of n_fully_connected corresponds to the size of ). The nonlinearity `nonlin` is used univerally between all layers, while `norm_layer` defines the kind of batch norm used by the `MulitiFilterLayer`s."""

    def __init__(
        self,
        nchannels_multifilters,
        filterSizes,
        nfully_connected,
        nonlin=None,
        norm_layer=None,
        strides=None,
    ):
        super(Animal10, self).__init__()
        if nonlin is None:
            nonlin = nn.ReLU()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if strides is None:
            strides = [2 for _ in range(len(filterSizes))]

        # define the conv layers
        multiFilters = []
        for (ii, nchannels) in enumerate(nchannels_multifilters[:-1]):
            previous_out_channels = nchannels[-1]
            current_nchannels = nchannels_multifilters[ii + 1]
            multiFilters.append(
                MultiFilterLayer(
                    previous_out_channels,
                    current_nchannels,
                    nonlin,
                    norm_layer,
                    filterSizes[ii],
                    strides[ii],
                )
            )
        self.multiFilters = nn.ModuleList(multiFilters)
        self.adaptiveAve = nn.AdaptiveAvgPool2d((1, 1))
        # define the fully connected layers
        self.fullyConnected = FullyConnected(
            [nchannels_multifilters[-1][-1], *nfully_connected], nonlin
        )

    def forward(self, x):
        for layer in self.multiFilters:
            x = layer(x)
        x = self.adaptiveAve(x)
        x = torch.flatten(x, 1)  # drop the spatial components
        x = self.fullyConnected(x)
        return x
