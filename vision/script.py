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
import sys

sys.path.append(".")
from models import MultiFilterLayer, FullyConnected, Animal10

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


def full_train_test_run(n_channels, filterSizes, fully_connected, strides, n_epochs):
    animal_classifier = Animal10(
        n_channels, filterSizes, fully_connected, strides=strides
    )
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
        classname: correct_pred[classname] / total_pred[classname]
        for classname in classes
    }
    print(f"total accuracy: {top1/total}")
    print(f"top-3 accuracy: {top3/total}")
    print(correct_class)


# n_channels = (
#     [3],
#     [16, 16, 16, 64],
#     [16, 16, 16, 64],
#     [16, 16, 16, 128],
#     [32, 32, 32, 256],
# )
# filterSizes = ([3, 5, 7], [3, 5, 5], [3, 3, 5], [3, 3, 3])
# fully_connected = (256, 256, 256, 10)
# strides = [2, 4, 4, 4]
# n_epochs = 63
# full_train_test_run(n_channels, filterSizes, fully_connected, strides, n_epochs)


# increasing the number of output convolutional channels
# n_channels = (
#     [3],
#     [16, 16, 16, 64],
#     [16, 16, 16, 128],
#     [16, 16, 16, 256],
#     [32, 32, 32, 512],
# )
# filterSizes = ([3, 5, 7], [3, 5, 5], [3, 3, 5], [3, 3, 3])
# fully_connected = (512, 512, 256, 128, 10)
# strides = [2, 4, 4, 4]
# n_epochs = 59
# full_train_test_run(n_channels, filterSizes, fully_connected, strides, n_epochs)


# decreasing the number of fully connected layers
# n_channels = (
#     [3],
#     [16, 16, 16, 64],
#     [16, 16, 16, 128],
#     [16, 16, 16, 256],
#     [32, 32, 32, 512],
# )
# filterSizes = ([3, 5, 7], [3, 5, 5], [3, 3, 5], [3, 3, 3])
# fully_connected = (256, 256, 256, 10)
# strides = [2, 2, 4, 4]
# n_epochs = 27
# full_train_test_run(n_channels, filterSizes, fully_connected, strides, n_epochs)


n_channels = (
    [3],
    [32, 16, 16, 64],
    [64, 32, 32, 128],
    [128, 16, 64, 256],
    [256, 128, 128, 512],
)
filterSizes = ([3, 5, 7], [3, 5, 7], [3, 3, 5], [3, 3, 3])
fully_connected = (512, 512, 256, 128, 10)
strides = [2, 4, 4, 4]
n_epochs = 59
full_train_test_run(n_channels, filterSizes, fully_connected, strides, n_epochs)
