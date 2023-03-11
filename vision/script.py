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

batch_size = 5
target_size = 800
running_average_size = 100
split_seed = 42
test_split = 0.2
n_epochs = 60

working_data_dir = "/fasterHome/workingDataDir/kaggle/animals10"
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}
for folder in os.listdir(os.path.join(working_data_dir, "raw-img")):
    contained = os.path.join(working_data_dir, "raw-img", folder)
    if folder in translate:
        os.rename(
            contained, os.path.join(working_data_dir, "raw-img", translate[folder])
        )


all_images = tv.datasets.ImageFolder(
    os.path.join(working_data_dir, "raw-img"), transform=tv.transforms.ToTensor(),
)


def needed_pad(x: torch.Tensor, max_size: torch.Size):
    """pad `x` so that it's last two dimensions have size `max_size`"""
    diff = np.array(max_size) - np.array(x.shape[-2:])
    diff = diff / 2
    return nnF.pad(
        x,
        (
            int(np.floor(diff[1])),
            int(np.ceil(diff[1])),
            int(np.floor(diff[0])),
            int(np.ceil(diff[0])),
        ),
    )


def collate_unevenly_sized(batch_iter):
    """collation function. first it checks that all images are smaller than `target_size`, and subsamples so that they are approximately below `target_size` using 2D average pooling. Then it pads all images so they match the largest in the batch"""
    images, labels = zip(*batch_iter)
    max_size, _ = torch.max(torch.tensor([t.shape[-2:] for t in images]), dim=0)
    if (max_size >= target_size).any():
        tmp = [image for image in images]
        for ii, image in enumerate(images):
            sz = torch.tensor(image.shape[-2:])
            if (sz >= target_size).any():
                rate = torch.ceil(sz / target_size).to(int)
                pooler = nn.AvgPool2d((rate[0], rate[1]))
                tmp[ii] = pooler(image)
        images = tmp
    max_size, _ = torch.max(torch.tensor([t.shape[-2:] for t in images]), dim=0)
    images = torch.cat([needed_pad(x, max_size).unsqueeze(0) for x in images], dim=0)
    labels = nnF.one_hot(torch.tensor(labels), num_classes=10).to(torch.float)
    return images, labels


# generate a simple train/test split
dataset_size = len(all_images)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.seed(split_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_loader = DataLoader(
    all_images,
    collate_fn=collate_unevenly_sized,
    batch_size=batch_size,
    sampler=train_sampler,
)
test_loader = DataLoader(
    all_images,
    collate_fn=collate_unevenly_sized,
    batch_size=batch_size,
    sampler=test_sampler,
)


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


n_channels = (
    [3],
    [16, 16, 16, 64],
    [16, 16, 16, 128],
    [16, 16, 16, 256],
    [32, 32, 32, 512],
)
filterSizes = ([3, 5, 7], [3, 5, 5], [3, 3, 5], [3, 3, 3])
fully_connected = (512, 512, 256, 128, 10)
strides = [2, 4, 4, 4]
animal_classifier = Animal10(n_channels, filterSizes, fully_connected, strides=strides)
device = torch.device("cuda")
animal_classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(animal_classifier.parameters(), lr=0.001, momentum=0.9)

loss_record: list[float] = []
for epoch in range(n_epochs):
    running_loss = 0.0
    for ii, data in enumerate(train_loader):
        image_batch, labels = data
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        class_pred = animal_classifier(image_batch)
        loss = criterion(class_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        del loss, image_batch, class_pred
        if ii % running_average_size == running_average_size - 1:
            print(
                f"[{epoch}, {ii + 1:5d}] loss: {running_loss / running_average_size:.3f}"
            )
            loss_record.append(running_loss / running_average_size)
            running_loss = 0
now = dt.datetime.now()
loss_record = [x / running_average_size for x in loss_record]
torch.save(animal_classifier.state_dict(), f"runs/model{now}.pth")
np.save(f"runs/lossRecord{now}.npy", np.array(loss_record))
np.save(f"runs/nChannels{now}.npy", np.array(n_channels))
np.save(f"runs/filterSizes{now}.npy", np.array(filterSizes))
np.save(f"runs/fully_connected{now}.npy", np.array(fully_connected))
np.save(f"runs/strides{now}.npy", np.array(strides))


# evaluating on the original data
top1 = 0
top3 = 0
classes = all_images.classes
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
total = 0
with torch.no_grad():
    for ii, data in enumerate(train_loader):
        image_batch, labels = data
        image_batch = image_batch.to(device)

        _, sorted_pred = torch.sort(animal_classifier(image_batch), 1)
        top_3_pred = sorted_pred[:, -3:].to("cpu")
        top_pred = sorted_pred[:, -1].to("cpu")
        _, label_index = torch.max(labels, 1)
        total += labels.size(0)
        top1 += (top_pred == label_index).sum().item()
        top3 += (top_3_pred.T == label_index).any(0).sum().item()
        for label, prediction in zip(label_index, top_pred):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

correct_class = {
    classname: correct_pred[classname] / total_pred[classname] for classname in classes
}
print(f"total accuracy: {top1/total}")
print(f"top-3 accuracy: {top3/total}")
print(correct_class)

for data in train_loader:
    break

# plot a smoothed error
from scipy.signal import savgol_filter

plt.plot(np.array(range(len(loss_record))) / 41, savgol_filter(loss_record, 101, 3))
plt.show()

# tmpAnimal = Animal10(
#     [[3], [16, 16, 16, 128]], filterSizes, fully_connected, strides=strides
# )
# tmpAnimal.load_state_dict(torch.load(f"runs/model{now}.pth"))
# thing = torch.load(f"runs/model{now}.pth")
