import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

import math
from collections import OrderedDict

import numpy as np
from ray import tune
from torchvision import datasets, transforms

import workloads.common as com
DATA_PATH, RESULTS_PATH = com.detect_paths()


class VGG(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3,
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            torch.nn.Conv2d(3,8,3,padding=1),
            # TODO: max-pooling layer, size 2
            torch.nn.MaxPool2d(2),
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            torch.nn.Conv2d(8,16,3,padding=1),
            # TODO: max-pooling layer, size 2
            torch.nn.MaxPool2d(2),
            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            torch.nn.Conv2d(16,32,3,padding=1),
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            torch.nn.Conv2d(32,32,3,padding=1),
            # TODO: max-pooling layer, size 2
            torch.nn.MaxPool2d(2),
            # Stage 4
            # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
            torch.nn.Conv2d(32,64,3,padding=1),
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            torch.nn.Conv2d(64,64,3,padding=1),
            # TODO: max-pooling layer, size 2
            torch.nn.MaxPool2d(2),
            # Stage 5
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            torch.nn.Conv2d(64,64,3,padding=1),
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            torch.nn.Conv2d(64,64,3,padding=1),
            # TODO: max-pooling layer, size 2
            torch.nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
            torch.nn.Linear(64,64),
            torch.nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x



def model_creator(config):
    return VGG(config)


def exp_metric():
    return dict(metric="val_accuracy", mode="max")

def create_stopper():
    return com.MetricStopper(0.92, **exp_metric())

def data_creator(config):
    print('==> Preparing data..')
    traindir = os.path.join(DATA_PATH, 'train')
    valdir = os.path.join(DATA_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
         pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False, pin_memory=True)














# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def train(trainloader, net, criterion, optimizer, device):
    #optimizer = optim(net.parameters())
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # TODO: zero the parameter gradients
            optimizer.zero_grad()
            # TODO: forward pass

            outputs = net(images)

            loss = criterion(outputs, labels)

            # TODO: backward pass
            loss.backward()
            # TODO: optimize the network
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)

    net = VGG().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005,eps=1e-08, betas=(0.9, 0.999))

    train(trainloader, net, criterion, optimizer, device)

    test(testloader, net, device)


if __name__== "__main__":
    main()
