
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms
import pandas as pd

from PIL import Image
import pickle
import random
from collections import OrderedDict, Counter

def get_bounding_box(image):
    """
    Get the bounding box of a white box in a black image.

    Args:
        image (torch.Tensor): A binary image of shape (H, W) where the white region (value=1) represents the box.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates of the bounding box.
    """
    # Find indices of all non-zero (white) pixels
    non_zero_indices = torch.nonzero(image, as_tuple=True)  # Returns (y_indices, x_indices)

    # Get bounding box coordinates
    y_min, y_max = torch.min(non_zero_indices[0]).item(), torch.max(non_zero_indices[0]).item()
    x_min, x_max = torch.min(non_zero_indices[1]).item(), torch.max(non_zero_indices[1]).item()

    return x_min, y_min, x_max, y_max


class NaturalSceneDataset(Dataset):
    '''
    Dataset to load in ObjectArray data and targets
    A sample consists of target image(cropped to only show the object) and the
    context image(entire image), the bounding box coordinates of the target object([xmin, ymin, w, h])

    '''
    def __init__(self, gt_dir, stimuli_dir, target_dir, normalize_means=None, normalize_stds=None):

        self.stimuli_dir = stimuli_dir
        self.target_dir = target_dir
        self.gt_dir = gt_dir

        if normalize_means is not None and normalize_stds is not None:
            self.normalize_means = normalize_means
            self.normalize_stds = normalize_stds
            self.normalize = True
        else:
            self.normalize = False


    def __len__(self):
        return 240

    def __getitem__(self, idx):
        # Load the target and context images
        target_img = Image.open(f'{self.target_dir}/t{(idx + 1):03d}.jpg')
        stimuli_img = Image.open(f'{self.stimuli_dir}/img{(idx + 1):03d}.jpg')
        gt_img = Image.open(f'{self.gt_dir}/gt{idx+1}.jpg')

        # Determine the bounding box of the target object

        xmin, ymin, xmax, ymax = get_bounding_box(to_tensor(gt_img))


        # Convert the images to tensors
        target_img = to_tensor(target_img)
        context_img = to_tensor(stimuli_img)

        # Normalize the images
        if self.normalize:
            transforms.Normalize(target_img, self.normalize_means, self.normalize_stds)
            transforms.Normalize(context_img, self.normalize_means, self.normalize_stds)

        return context_img, target_img, (xmin, ymin, xmax, ymax) #(bounding box given as (xmin, ymin, xmax, ymax))

