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
from astropy.io import fits
import DeepSkyLinearIntegrationDataset
import models.multisacle_denoise, models.models
from astropy.io import fits
import kornia.losses




def train_net(net, trainloader, valloader, optimizer, criterion, results_dir, epochs=5):
    logger = logging.getLogger(__name__)
    weights_dir = results_dir / ('models')
    image_dir = results_dir / ('sample_imgs')
    weights_dir.mkdir(exist_ok=True, parents=True)
    image_dir.mkdir(exist_ok=True, parents=True)
    print(image_dir)
    print(weights_dir)
    val_accuracy = 0.
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    for epoch in range(epochs):

        # Training
        net = net.train()
        epoch_star_time = timeit.default_timer()



        for step, (train_x, train_y) in enumerate(trainloader):
            #if you want to view images, do it before tensors are sent to GPU
            #f, ax = plt.subplots(1, 2)
            #ax[0].imshow(train_x[0,0])
            #ax[0].set_title("X[0]")
            #ax[1].imshow(train_y[0,0])
            #ax[1].set_title("Y[0]")
            #plt.show()

            train_x, train_y = train_x.to(device, non_blocking=True), train_y.to(device, non_blocking=True)
            N = train_x.size(0)



            optimizer.zero_grad()
            denoised = net(train_x)

            loss = criterion(denoised, train_y)
            loss.backward()
            optimizer.step()



            if step % 16 == 0 or step == len(trainloader) - 1:
                time_per_step = (timeit.default_timer()-epoch_star_time)/(step+1)
                logger.warning("Train loss: " + str(loss.item()) + " | Time per step: " + str(time_per_step) +
                               " Estimated time remaining: "+ str(time_per_step*(len(trainloader)-step)))





        # VALIDATION
        with torch.no_grad():
            for data in valloader:
                val_x, val_y = data
                val_x, val_y = val_x.to(device), val_x.to(device)
                denoised = net(val_x)

                val_loss =criterion(denoised, val_y)
                logger.warning("Val loss: " + str(val_loss))


        x_ssim = criterion(val_x,val_y).item()
        denoised_ssim = criterion(denoised,val_y).item()
        y_ssim = criterion(val_y,val_y).item()

        displayable_input = val_x.cpu().detach().numpy()
        displayable_result = denoised.cpu().detach().numpy()
        displayable_y = val_y.cpu().detach().numpy()


        f, ax = plt.subplots(1, 3)
        ax[0].imshow(displayable_input[0, 0])
        ax[1].imshow(displayable_result[0, 0])
        ax[2].imshow(displayable_y[0, 0])
        plt.suptitle("SSIM X,Y " + str(x_ssim) + "\nSSIM Denoised,Y: "+ str(denoised_ssim)+ "\nSSIM Y,Y: " + str(y_ssim))
        plt.savefig(str(image_dir / (str(epoch)+'.png')))
        plt.close(f)

        patch_path = str(image_dir / (str(epoch)+'_x.fits'))
        print("Saving a patch to: " + str(patch_path))

        patch_fits = fits.PrimaryHDU(data=displayable_input)
        patch_fits.writeto(patch_path, overwrite=True)

        patch_path = str(image_dir / (str(epoch)+'_denoised.fits'))
        print("Saving a patch to: " + str(patch_path))

        patch_fits = fits.PrimaryHDU(data=displayable_result)
        patch_fits.writeto(patch_path, overwrite=True)


        patch_path = str(image_dir / (str(epoch)+'_y.fits'))
        print("Saving a patch to: " + str(patch_path))

        patch_fits = fits.PrimaryHDU(data=displayable_y)
        patch_fits.writeto(patch_path, overwrite=True)


        torch.save(net, str(weights_dir/"{:03d}.pth").format(epoch+1))

    return val_accuracy

def load_patch(dir: Path):
    integration_path = dir / 'int'
    sub_path = dir / 'sub'

    if not integration_path.exists() or not sub_path.exists():
        AssertionError("missing required structure to load patches :(")

    x= []
    for sub in sub_path.glob("*.fit*"):
        sd = fits.open((sub))[0].data
        x.append(sd)

    y = []
    for integration in integration_path.glob("*.fit*"):
        integrationd = fits.open((integration))[0].data
        y.append(integrationd)

    return (x,y)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--network", default="Multiscale")
    ap.add_argument("--loss_func", default="MSE")
    ap.add_argument("--batch_size", default=16, type=int)
    args = ap.parse_args()

    data_path = Path(args.data_dir)

    #baseline_config = config.NetworkConfig()
    baseline_network = None



    if args.network == "Multiscale":
        baseline_network = models.multisacle_denoise.multiscale_denoise()
    elif args.network == "DnCNN":
        baseline_network = models.models.DnCNN()
    elif args.network == "FastARCNN":
        baseline_network = models.models.FastARCNN()
    else:
        print("Unsupported network!")
        SystemExit(1)

    loss_func = None
    if args.loss_func == "MSE":
        loss_func = torch.nn.MSELoss()
    elif args.loss_func == "SSE":
        loss_func = "Potato"
    elif args.loss_func == "SSIM":
        loss_func = kornia.losses.SSIM(window_size=11,reduction='mean')
    elif args.loss_func == "PSNR":
        loss_func = kornia.losses.PSNRLoss(max_val=1)
    else:
        print("Unsupported loss function!")
        SystemExit(1)

    results_dir = Path("results") / str(args.network + "_"+args.loss_func)
    results_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(str(results_dir/'file.log'))
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


    patches_dir = Path(args.data_dir) / "training" / "patches"

    patch_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        torchvision.transforms.RandomCrop(224),
        transforms.ToTensor()

    ])

    dslids = DeepSkyLinearIntegrationDataset.DeepSkyLinearIntegrationDataset(data_path, transform=patch_transforms)

    some_patches = dslids[0]
    train_size = int(0.9 * len(dslids))
    val_size = len(dslids) - train_size

    train_split, val_split = torch.utils.data.random_split(dslids, [train_size, val_size])

    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_split)

    opt = torch.optim.Adam(baseline_network.parameters(), lr=0.01)
    train_net(baseline_network, train_loader, val_loader, optimizer=opt, criterion=loss_func,
              results_dir=results_dir, epochs=20)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.freeze_support()
    main()