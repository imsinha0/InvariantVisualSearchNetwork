import matplotlib.pyplot as plt
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
        image (torch.Tensor): A tensor of shape (3, H, W) representing an RGB image
                              where the white region is the same across all channels.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates of the bounding box.
    """
    # Combine the three channels into a single-channel binary mask
    # Assume white pixels have the same value across all channels and are significantly brighter
    grayscale = image.sum(dim=0)  # Sum across the channels, resulting in shape (H, W)
    binary_mask = grayscale > 0  # Threshold: Non-black pixels become True (1)

    # Find the coordinates of the white region
    white_pixels = torch.nonzero(binary_mask)  # shape: (N, 2), where N is the number of white pixels

    # Extract the min and max coordinates
    y_min, x_min = white_pixels.min(dim=0).values
    y_max, x_max = white_pixels.max(dim=0).values

    return x_min.item(), y_min.item(), x_max.item(), y_max.item()


index_order = ["1_1", "1_2", "1_3", "1_4", "1_5", "1_6", "1_7", "1_8", "1_9", "1_10", "1_11", "1_12",
               "2_1", "2_2", "2_3", "2_4", "2_5", "2_6", "2_7", "2_8", "2_9", "2_10", "2_11", "2_12",
                "3_1", "3_2", "3_3", "3_4", "3_5", "3_6", "3_7", "3_8", "3_9", "3_10", "3_11",
                "4_1", "4_2", "4_3", "4_4", "4_5", "4_6", "4_7", "4_8", "4_9", "4_10", "4_11",
                "5_1", "5_2", "5_3", "5_4", "5_5", "5_6", "5_7", "5_8", "5_9", "5_10", "5_11", "5_12",
                "6_1", "6_2", "6_3", "6_4", "6_5", "6_6", "6_7", "6_8", "6_9", "6_10"]


class WaldoDataset(Dataset):
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
        return len(index_order)

    def __getitem__(self, idx):
        # Load the target and context images
        target_img = Image.open(f'{self.target_dir}/waldo.JPG')
        stimuli_img = Image.open(f'{self.stimuli_dir}/cropped_{index_order[idx]}.jpg')
        gt_img = Image.open(f'{self.gt_dir}/gt_{index_order[idx]}.jpg')

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

