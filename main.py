import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing
from pathlib import Path
import torch.optim
import timeit
import matplotlib.pyplot as plt
import sklearn




def eval_net(net, val_loader, device='cuda'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on validaton set: %d %%' % (
        100 * correct / total))
    return (correct/total)


def train_net(net, trainloader, valloader, optimizer, criterion, weights_dir, epochs=5):
    logger = logging.getLogger(__name__)
    weights_dir.mkdir(exist_ok=True, parents=True)
    val_accuracy = 0.
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    for epoch in range(epochs):

        # Training
        net = net.train()
        epoch_star_time = timeit.default_timer()


        correct = 0
        total = 0

        for step, (train_x, train_y) in enumerate(trainloader):
            train_x, train_y = train_x.to(device, non_blocking=True), train_y.to(device, non_blocking=True)
            N = train_x.size(0)



            optimizer.zero_grad()
            logits = net(train_x)

            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            total += train_y.size(0)
            correct += (predicted == train_y).sum().item()

            if step % 250 == 0 or step == len(trainloader) - 1:
                time_per_step = (timeit.default_timer()-epoch_star_time)/(step+1)
                logger.warning("Train acc: " + str(correct / total) + " Train loss: " + str(loss.item()) + " | Time per step: " + str(time_per_step) +
                               " Estimated time remaining: "+ str(time_per_step*(len(trainloader)-step)))




        # VALIDATION
        val_accuracy = eval_net(net, valloader)





        torch.save(net, str(weights_dir/"{:03d}.pth").format(epoch+1))

    return val_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--nimrod_key")
    ap.add_argument("--network", default="SqueezeNet")

    args = ap.parse_args()


    #baseline_config = config.NetworkConfig()
    baseline_network = None



    if args.network == "ResNet50":
        baseline_network = torchvision.models.resnet50()
    elif args.network =="ResNet152":
        baseline_network = torchvision.models.resnet152()
    elif args.network == 'SqueezeNet':
        baseline_network = torchvision.models.squeezenet1_1()
    elif args.network == 'DenseNet121':
        baseline_network = torchvision.models.densenet121()
    elif args.network == 'DenseNet201':
        baseline_network = torchvision.models.densenet201()
    elif args.network == 'DeepLabv3+':
        baseline_network = torchvision.models.segmentation.deeplabv3_resnet50()
    else:
        print("Unsupported network!")
        SystemExit(1)

    results_dir = Path("results") / args.network
    results_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(str(results_dir/'file.log'))
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    batch_size = 16
    patches_dir = Path(args.data_dir) / "training" / "patches"

    patch_transforms = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(hue=0.2, saturation=0.2),
        transforms.ToTensor()

    ])

    patches_dataset = ImageFolder(patches_dir, transform=patch_transforms)

    train_split, val_split = torch.utils.data.random_split(patches_dataset, [len(patches_dataset)-5000, 5000])

    print(len(train_split))
    print(len(val_split))

    samples_per_class = {}
    for c in patches_dataset.classes:
        samples_per_class[c] = 0
    for i in train_split.indices:
        sample = patches_dataset.samples[i]
        class_of_sample = patches_dataset.classes[sample[1]]
        samples_per_class[class_of_sample] += 1

    smallest_class = (None, float("inf"))
    for k, v in samples_per_class.items():
        print(k, v)
        if v < smallest_class[1]:
            smallest_class = (k,v)
    logger.warning(samples_per_class)
    logger.warning("Undersampling to match: " + str(smallest_class))
    print(smallest_class[1]-3)
    #smallest_class = (None, 5000)
    patch_sampler = WeightedRandomSampler(weights= [smallest_class[1],smallest_class[1]],replacement=True, num_samples=int(smallest_class[1]*2)-5)
    train_loader = DataLoader(train_split, sampler=patch_sampler, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_split)

    opt = torch.optim.SGD(baseline_network.parameters(), lr=0.01, momentum=0.9)
    train_net(baseline_network, train_loader, val_loader, criterion=nn.CrossEntropyLoss(),
              weights_dir=results_dir / "models",optimizer=opt, epochs=20)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.freeze_support()
    main()