from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, cuda=False, batch_size=32,
                 patience=5, checkpoint_dir=None):
        print("Cuda is available") if cuda else print("Cuda is not avaliable")

        self.model = model.to('cuda') if cuda else model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.batch_size = batch_size
        self.cuda = cuda
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.epoch = 0
        self.writer = None
        self.bad_epoch = 0
        self.min_loss = np.inf

    def train_epoch(self, validate=False):
        dataloader = self.train_loader if not validate else self.valid_loader
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

                running_loss += loss.detach().item()

                if not validate:
                    desc = f'Epoch Train {epoch} - loss {running_loss / (i + 1):.4f}'
                else:
                    desc = f'Validate - loss {running_loss / (i + 1):.4f}'

                epoch_pbar.set_description(desc)

        epoch_loss = running_loss / len(dataloader)
        log_name = 'Training' if not validate else 'Validate'

        if self.writer:
            # log scaler to Tensorboard
            self.writer.add_scalar(f'{log_name} loss', epoch_loss, epoch)

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
            self.train_epoch(validate=True)

    def train(self, n_epochs, logs_dir="", val_epoch=1):
        self.bad_epoch = 0
        total_loss= 0
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

            if self.bad_epoch == self.patience:
                print("Patience reached.")
                break

        print(f'Total loss {total_loss/n_epochs}.')
        print('Finished Training')

    def evaluate(self):
        with torch.no_grad():
            return self.train_epoch(validate=True)

    def inference(self, images=None, loader=None):
        if images is None:
            if loader is None:
                try:
                    loader = self.valid_loader
                except:
                    print('No Dataloader was specified')
            in_imgs, _ = next(iter(loader))
        else:
            in_imgs = images

        with torch.no_grad():
            self.model.eval()
            output = self.model(in_imgs)
            _, pred = torch.max(output.data, 1)
            return pred

    def save(self, path, inference=True):
        if inference:
            torch.save(self.model.state_dict(), path)
            print('Model saved. ', path)
        else:
            torch.save({
                'epoch': self.epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'criterion_state_dict': self.criterion.state_dict(),
            }, path)
            print('Checkpoint saved. ', path)

    def load(self, path, inference=True):
        if inference:
            self.model.load_state_dict(torch.load(path))
            print('Model loaded. ', path)
        else:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Checkpoint loaded. ', path)
