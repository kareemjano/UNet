from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from utils.dataset_utils import read_image

class CustomDataloader():

    def __init__(self, datasets, batch_size=32, num_workers=1, input_size=572):
        """
        :param dataset: LfwImagesDataset(), if manual_split==True than this is the LfwImagesPairsDataset train set
        :param batch_size: default value: 32
        :param num_workers: if manual_split==True this must be the validation LfwImagesPairsDataset
        :param input_size: (int) size of the input image. assuming width and height are the same
        """
        super().__init__()
        self.batch_size = batch_size
        self.datasets = datasets
        self.num_workers = num_workers
        self.input_size = input_size

    def setup(self):
        # transforms
        my_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size))
        ])

        (dataset.set_transform(my_transform) for dataset in self.datasets)

        self.train_dataset, self.val_dataset, self.test_dataset = self.datasets

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           sampler=None,
                                           collate_fn=None
                                           )

class CustomDataset(Dataset):
    # define the constructor of this dataset object
    def __init__(self, dataset_folder, labels_map_path, my_transform=None, input_size=572, output_size=388):
        """
        Create VOC 2012 pytorch Dataset
        :param dataset_folder: main VOC 2012 directory
        :param labels_map_path: path to labels_map txt file
        :param my_transform: pytorch transforms to be applied
        :param input_size: input size of the model
        :param output_size: output size of the model
        """
        self.base = os.path.join(dataset_folder, 'VOCdevkit', 'VOC2009')
        self.images_dir = os.path.join(self.base, 'JPEGImages')
        self.segs_dir = os.path.join(self.base, 'SegmentationClass')
        self.output_size = output_size
        self.input_size = input_size
        self.transform = my_transform

        with open(labels_map_path) as f:
            label_map = f.readlines()

        # label_map = label_map[:int(len(label_map)/4)]
        label_map = label_map[:5]
        self.labels_map = [x.strip() for x in label_map]

    def __len__(self):
        return len(self.labels_map)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.labels_map[idx] + ".jpg")
        seg_path = os.path.join(self.segs_dir, self.labels_map[idx] + ".png")

        image = read_image(img_path, size=self.input_size, transform=self.transform)

        seg = Image.open(seg_path).resize((self.output_size, self.output_size))
        seg = np.array(seg).astype(dtype=np.uint8)

        return image, torch.tensor(seg)
