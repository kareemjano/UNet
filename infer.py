import sys
sys.path.append('src')

import argparse
import os
import torch
import numpy as np

from model.networks import UNet
from model.solver import Trainer
from utils.dataset_utils import read_image
from utils.visualization_utils import decode_segmap
from PIL import Image

class InferenceException(Exception):
    """main inference error class"""

def run(parser):
    checkpoint = parser.checkpoint
    image_path = parser.image

    # loading model
    n_classes = 21
    unet_model = UNet(n_classes)

    if os.path.exists(image_path):
        image = read_image(image_path, unet_model.input_size)
        image = torch.tensor(image).unsqueeze(0)
    else:
        raise InferenceException("Image path doesn't exist")

    trainer = Trainer(unet_model)
    if os.path.exists(checkpoint):

        trainer.load(checkpoint)
    else:
        print("Checkpoint path doesn't exist")

    #perfom inference and save image
    out, _ = trainer.inference(images=image)
    out_image = decode_segmap(out.squeeze(0))
    print(np.unique(out_image))

    out_path = os.path.join(os.getcwd(), "data", "output")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    save_path = os.path.join(out_path, 'output.png')
    im = Image.fromarray(out_image)
    im.save(save_path)
    print(f"image saved to {save_path}")

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Semantic segmentation inference using U-Net',
        usage='python infer.py',
        description='This module performs semantic segmentation inference using U-Net checkpoint on a given image.',
        add_help=True
    )

    checkpoint_default = os.path.join(os.getcwd(), 'data', 'models', 'UNet', 'checkpoint.model')
    parser.add_argument('-c', '--checkpoint', type=str, default=checkpoint_default, help='checkpoint file to use')
    parser.add_argument('-i', '--image', type=str, required=True, help='image to perform inference one')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    print("inference args:")
    print(parser)
    run(parser)