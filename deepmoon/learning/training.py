import numpy as np

import torch
from torch.utils.data import (DataLoader, SubsetRandomSampler)
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from deepmoon.learning.moondata import (MoonCrater, MoonCraterH5)
from deepmoon.learning.model import Crater_VNet


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


def training(path, img_size, learning_rate, batch_size, num_worker, epoch,
             split, shuffle, filter_len, number_of_filters, dropout, output, h5=False, checkpoint=None):

    if h5:
        moon_crater_dataset = MoonCraterH5(root=path,
                                     transform=ToTensor())
    else:
        moon_crater_dataset = MoonCrater(root=path,
                                     transform=ToTensor(),
                                     image_size=img_size)


    (moon_crater_training,
     moon_crater_validation) = load_split_datasets(dataset=moon_crater_dataset,
                                                   validataion_size=split,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_worker=num_worker)

    checkpointModel = ModelCheckpoint(dirpath=f"{output}/logs/checkpoints/",
                                filename='DeepMoon-{epoch:02d}-{val_loss:.2f}',
                                monitor="val_loss",
                                verbose=True,
                                save_top_k=3)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epoch,
                         logger=pl_loggers.TensorBoardLogger(f'{output}/logs/tb/'),
                         resume_from_checkpoint=f'{output}/logs/checkpoints/{checkpoint}.ckpt' if checkpoint is not None else None,
                         #default_root_dir=f'{output}/checkpoints',
                         callbacks=[checkpointModel]
                        )

    trainer.fit(Crater_VNet("relu", dropout, lr=learning_rate),
                train_dataloader=moon_crater_training,
                val_dataloaders=moon_crater_validation)
