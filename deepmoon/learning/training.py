import pathlib
import shutil

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from deepmoon.learning.moondata import MoonCrater
from deepmoon.learning.model import Crater_VNet


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().detach().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def load_split_datasets(dataset, validataion_size, batch_size, shuffle,
                        num_worker):
    ## fix validataionsize to percent.
    # if validateion size ggt one
    if validataion_size > 1:
        # then divide the value so long with 10
        # until the value is smaller than 1 (1 eq 100 percent)
        validataion_size_ = validataion_size
        while (validataion_size_ > 1.):
            validataion_size_ /= 10
        validataion_size = validataion_size_

    image_number = len(dataset)
    indices = list(range(image_number))

    # caluclate the set split
    split = int(np.floor(validataion_size * image_number))

    if shuffle:
        np.random.shuffle(indices)

    train_indices, validation_indeces = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indeces)

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=num_worker,
                              sampler=train_sampler)

    validation_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=num_worker,
                                   sampler=validation_sampler)

    return (train_loader, validation_loader)


def save_checkpoint(state,
                    is_best,
                    path,
                    prefix,
                    filename='checkpoint.pth.tar'):
    prefix_save = pathlib.Path(path) / prefix
    if not prefix_save.is_dir():
        prefix_save.mkdir(parents=True, exist_ok=True)
    name = str(prefix_save) + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def training(path, img_size, learning_rate, batch_size, num_worker, epoch,
             split, shuffle, filter_len, number_of_filters, dropout, output):

    best_prec1 = 100.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    moon_crater_dataset = MoonCrater(root=path,
                                     transform=transforms.ToTensor(),
                                     image_size=img_size)

    (moon_crater_training,
     moon_crater_validation) = load_split_datasets(dataset=moon_crater_dataset,
                                                   validataion_size=split,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_worker=num_worker)

    model = Crater_VNet("relu", dropout)
    model.to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(model)

    writer = SummaryWriter(f'{output}/runs/')
    first_run = True

    for epoche in range(epoch):
        total_loss_train = 0
        total_loss_val = 0

        t = v = 0

        pbar = tqdm.tqdm(moon_crater_training,
                         desc=f"Training of epoch {epoche+1}/{epoch}")
        for (input_image, target_image) in pbar:
            t += 1
            # to device
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            if first_run:
                writer.add_graph(model, input_image)
                first_run=False

            # forward + backward + optimize
            nn_image = model(input_image)
            loss = criterion(nn_image, target_image)
            total_loss_train += loss.item()

            pbar.set_postfix({"current training loss": loss.item()})

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        save_checkpoint(
            {
                'epoch': epoche,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1
            },
            is_best=False,
            path=output,
            prefix="crater_vnet")

        pbar = tqdm.tqdm(moon_crater_validation,
                         desc=f"Validataion of epoch {epoche+1}/{epoch}")
        target_ = None
        with torch.no_grad():
            for image, mask in pbar:
                v += 1
                image = image.to(device)
                mask = mask.to(device)

                model.eval()

                target_ = model(image)
                val_loss = criterion(target_, mask)
                total_loss_val += val_loss.item()
                pbar.set_postfix({"current validataion loss": val_loss.item()})

        # tensorboard images
        img_grid = make_grid(target_[0])
        matplotlib_imshow(img_grid, one_channel=True)
        writer.add_image('crater_images', img_grid, epoche)

        writer.add_scalar("Loss/Train", total_loss_train/t, epoche)
        writer.add_scalar("Loss/Validataion", total_loss_val/v, epoche)

        print(
            f"[{epoche+1}] train_loss: {total_loss_train} \t validataion_loss: {total_loss_val}"
        )
    writer.close()
