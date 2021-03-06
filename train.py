import sys
sys.path.append('src')

import os
import torch
import argparse

import multiprocessing as mp
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from dataset.customDataset import CustomDataset, CustomDataloader
from model.networks import UNet
from model.solver import Trainer


def run(parser):
    batch_size = parser.batch_size
    patience = parser.patience
    n_epochs = parser.n_epochs
    val_epoch = parser.val_epoch
    lr = parser.learning_rate
    weight_decay = parser.reg
    step_size = parser.step_size
    gamma = parser.gamma

    print('Preparing Dataset')
    models_dir = os.path.join('data', 'models')
    checkpoint_dir = os.path.join(models_dir, 'UNet')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # load data
    base = os.path.join(os.getcwd(), 'data', 'VOCtrainval_11-May-2009')
    dataset_folder = os.path.join(base, 'VOCdevkit', 'VOC2009')

    labels_dir = os.path.join(dataset_folder, 'ImageSets', 'Segmentation')
    train_txt = os.path.join(labels_dir, 'train.txt')
    val_txt = os.path.join(labels_dir, 'trainval.txt')
    test_txt = os.path.join(labels_dir, 'val.txt')

    dataset_txt = [train_txt, val_txt, test_txt]
    datasets = [CustomDataset(base, txt) for txt in dataset_txt]

    dataloaders = CustomDataloader(datasets, batch_size=batch_size, num_workers=mp.cpu_count())
    dataloaders.setup()

    # train
    optim_hparams = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    scheduler_params = {
        "step_size": step_size,
        "gamma": gamma,
    }
    n_classes = 21
    unet_model = UNet(n_classes)
    optimizer = Adam(unet_model.parameters(), **optim_hparams)
    scheduler = StepLR(optimizer, **scheduler_params)

    trainer = Trainer(unet_model, dataloaders, optimizer,
                      cuda=torch.cuda.is_available(),
                      batch_size=batch_size,
                      checkpoint_dir=checkpoint_dir, patience=patience, scheduler=scheduler)

    print('Starting Training...')
    print('run \'tensorboard --logdir=runs\' to open tensorboard')
    trainer.train(n_epochs=n_epochs, val_epoch=val_epoch, logs_dir='UNetModel')


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Train semantic segmentation using UNet on VOC 2012 dataset',
        usage='python train.py',
        description='This module trains, evaluates, and tests semantic segmentation model using U-Net on VOC 2012 dataset',
        add_help=True
    )

    parser.add_argument('-e', '--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-v', '--val_epoch', type=int, default=1, help='frequency of running validation epoch')
    parser.add_argument('-p', '--patience', type=int, default=5, help='patience to auto-stop training')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='training rate')
    parser.add_argument('-r', '--reg', type=float, default=1e-5, help='L2 regularization')
    parser.add_argument('-s', '--step_size', type=int, default=10, help='step size of the lr scheduler')
    parser.add_argument('-g', '--gamma', type=float, default=0.5, help='gamma of the lr scheduler')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    print("training args:")
    print(parser)
    run(parser)
