#!/usr/bin/env python

import os
import sys
import click

import torch
import pytorch_lightning as pl

if __debug__:
    scriptPath = os.path.realpath(os.path.dirname(__file__))
    sys.path.append(scriptPath + "/..")

from deepmoon.learning.training import training


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--shuffle",
              "-s",
              default=True,
              is_flag=True,
              help="set shuffle for dataloader")
@click.option("--num_worker",
              "-c",
              default=4,
              type=int,
              help="number of workers")
@click.option("--batch_size", "-b", default=16, type=int, help="batch size")
@click.option("--epoch",
              "-e",
              default=100,
              type=int,
              help="number of learning epochs")
@click.option("--learning_rate",
              "-l",
              default=10e-5,
              type=float,
              help="learning rate")
@click.option("--split",
              "-x",
              default=.4,
              type=float,
              help="split the dataset in a validation and trainings set")
@click.option("--filter_len",
              "-f",
              default=3,
              type=int,
              help="filter size")
@click.option("--number_of_filters",
              "-n",
              type=int,
              default=3,
              help="filter size")
@click.option("--img_size",
              "-z",
              default=256,
              type=int,
              help="image size")
@click.option("--dropout", "-d", default=.15, type=float, help="dropout value")
@click.argument("input_files_root_path")
@click.argument("output")
@click.argument("checkpoint", re)
def main(shuffle, num_worker, batch_size, epoch, learning_rate, split,
         filter_len, number_of_filters, dropout, img_size, input_files_root_path, output, checkpoint):

    if checkpoint is None:
        checkpoint = None

    print("\n\n")
    print("*" * 25, "   DEEPMOON   ", "*" * 25)
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print("__pytorch DEVICE: ", "cuda" if torch.cuda.is_available() else "cpu")
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    print('Current cuda device ', torch.cuda.current_device())
    print("pytorch lighning", pl.__version__)
    print("*" * 60)
    print("\n\n")

    training(input_files_root_path, img_size, learning_rate, batch_size, num_worker,
             epoch, split, shuffle, filter_len, number_of_filters, dropout, output, checkpoint)


if __name__ == '__main__':
    main()