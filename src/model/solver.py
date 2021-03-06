from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from utils.visualization_utils import visualize_torch


class Trainer:
    def __init__(self, model, dataloaders=None, optimizer=None, cuda=False, batch_size=32,
                 patience=5, checkpoint_dir=None, scheduler=None):
        print("Cuda is available") if cuda else print("Cuda is not avaliable")

        self.model = model.to('cuda') if cuda else model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.batch_size = batch_size
        self.cuda = cuda
        self.loader = {
            'train': dataloaders.train_dataloader(),
            'valid': dataloaders.val_dataloader(),
            'test': dataloaders.test_dataloader(),
        } if dataloaders is not None else None
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler

        self.epoch = 0
        self.writer = None
        self.bad_epoch = 0
        self.min_loss = np.inf

    def train_epoch(self, mode='train'):
        dataloader = self.loader[mode]
        validate = False if mode == 'train' else True

        epoch = self.epoch
        running_loss = 0.0

        self.model.train()
        if validate:
            self.model.eval()

        with tqdm(dataloader, total = len(dataloader)) as epoch_pbar:
            for i, data in enumerate(epoch_pbar):
                # get the inputs; data is a list of [inputs, labels]
                in_imgs, target_segs = data
                if self.cuda:
                    in_imgs, target_segs = in_imgs.cuda(), target_segs.cuda()
                # zero the parameter gradients
                if not validate:
                    self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.criterion(self.model(in_imgs), target_segs.long())

                if not validate:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.detach().cpu().item()

                if not validate:
                    desc = f'Epoch Train {epoch} - loss {running_loss / (i + 1):.4f}'
                else:
                    desc = f'Validate - loss {running_loss / (i + 1):.4f}'

                epoch_pbar.set_description(desc)

        epoch_loss = running_loss / len(dataloader)
        log_name = 'Training' if not validate else 'Validate'
        output_imgs, seg_img = self.inference(loader=dataloader)
        output_imgs, seg_img = output_imgs.detach().cpu(), seg_img.detach().cpu()

        if self.writer:
            # log scaler to Tensorboard
            self.writer.add_scalar(f'{log_name} loss', epoch_loss, epoch)
            self.writer.add_figure(log_name, visualize_torch(output_imgs), global_step=epoch)
            self.writer.add_figure(log_name+' gt', visualize_torch(seg_img), global_step=epoch)

        if validate:
            if self.patience > 0:
                if epoch_loss > self.min_loss:
                    self.bad_epoch += 1
                elif epoch_loss < self.min_loss:
                    self.bad_epoch -= 1 if self.bad_epoch > 0 else 0

            if epoch_loss < self.min_loss:
                self.min_loss = epoch_loss
                if self.checkpoint_dir:
                    if not os.path.exists(self.checkpoint_dir):
                        os.mkdir(self.checkpoint_dir)
                    self.save(os.path.join(self.checkpoint_dir, "checkpoint.model"), inference=False)

        return epoch_loss

    def evaluate_epoch(self):
        with torch.no_grad():
            self.train_epoch(mode='valid')

    def train(self, n_epochs, logs_dir="", val_epoch=1):
        self.bad_epoch = 0
        total_loss = 0
        count = 0

        while os.path.exists('runs') and logs_dir in os.listdir('runs'):
            logs_dir = logs_dir.split('_')[0] + f"_{count}"
            count += 1
        self.writer = SummaryWriter(f'runs/{logs_dir}')

        for epoch in range(self.epoch, self.epoch+n_epochs):
            self.epoch = epoch
            loss = self.train_epoch()
            total_loss += loss
            if (epoch+1) % val_epoch == 0:
                self.evaluate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.bad_epoch == self.patience:
                print("Patience reached.")
                break

        print(f'Total loss {total_loss/n_epochs}.')
        print('Finished Training')

    def test(self):
        with torch.no_grad():
            return self.train_epoch(mode='test')

    def inference(self, images=None, loader=None):
        seg_img = None
        if images is None:
            if loader is None:
                try:
                    loader = self.loader['valid']
                except:
                    print('No Dataloader was specified')
            in_imgs, seg_img = next(iter(loader))
        else:
            in_imgs = images

        with torch.no_grad():
            self.model.eval()
            output = self.model(in_imgs)
            F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            return pred, seg_img

    def save(self, path, inference=True):
        if inference:
            torch.save(self.model.state_dict(), path)
            print('Model saved. ', path)
        else:
            torch.save({
                'epoch': self.epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'criterion_state_dict': self.criterion.state_dict(),
            }, path)
            print('Checkpoint saved. ', path)

    def load(self, path, inference=True):
        device = 'cuda' if self.cuda else 'cpu'
        if inference:
            self.model.load_state_dict(torch.load(path, map_location=torch.device(device))['model_state_dict'])
            print('Model loaded. ', path)
        else:
            checkpoint = torch.load(path, map_location=torch.device(device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Checkpoint loaded. ', path)
