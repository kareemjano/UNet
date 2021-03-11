import urllib.request
import os
from tqdm import tqdm
import tarfile
import numpy as np
from PIL import Image

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _download_zip(url, zip_filename, target_dir):
    zip_path = os.path.join(target_dir, zip_filename)
    if zip_filename not in os.listdir(target_dir):
        print('\ndownloading zip file...')

        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    else:
        print('Dir is not empty')

    return zip_path

def get_dataset(dataset_url):
    print('\ncreating directories...')
    data_path = os.path.join(os.getcwd(), 'data')
    if 'data' not in os.listdir():
        print('creating', data_path)
        os.mkdir(data_path)
    else:
        print('data dir exists')

    url = dataset_url
    zip_filename = url.split('/')[-1]
    filename = zip_filename.split('.')[0]
    target_dir = os.path.join(data_path, filename)
    if filename not in os.listdir(data_path):
        print('creating', target_dir)
        os.mkdir(target_dir)
    else:
        print(target_dir, ' exists')

    zip_path = _download_zip(url, zip_filename, target_dir)

    if os.path.exists(zip_path):
        with tarfile.open(zip_path) as tar:
            tar.extractall(path=target_dir)
        os.remove(zip_path)
    else:
        print('file doesnt exist')

    return target_dir

def read_image(img_path, size, transform=None):
    image = np.array(Image.open(img_path).resize((size, size))) / 255
    image = np.transpose(image, [2, 0, 1])
    if transform:
        image = transform(image)
    return image

