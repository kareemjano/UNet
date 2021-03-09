import sys

sys.path.append('src')

from utils.dataset_utils import *

if __name__ == '__main__':
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar'
    dataset_dir = get_dataset(url)