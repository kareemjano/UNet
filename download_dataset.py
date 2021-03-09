from utils.dataset_utils import *
import os

print(os.getcwd())
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar'
dataset_dir = get_dataset(url)