
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


image_positions = {1:(56, 539), 2:(54, 234), 3:(330,85), 4:(592, 230), 5:(584, 530), 6:(337, 675)} #centers


class ArrayDataset(Dataset):
    '''
    Dataset to load in ObjectArray data and targets
    A sample consists of target image(cropped to only show the object) and the
    context image(entire image), the bounding box coordinates of the target object([xmin, ymin, w, h])

    '''
    def __init__(self, gt_positions, stimuli_dir, target_dir, target_size, normalize_means=None, normalize_stds=None):

        self.stimuli_dir = stimuli_dir
        self.target_dir = target_dir
        self.gt_positions = pd.read_csv(gt_positions) # ground truth results
        self.target_size = target_size

        if normalize_means is not None and normalize_stds is not None:
            self.normalize_means = normalize_means
            self.normalize_stds = normalize_stds
            self.normalize = True
        else:
            self.normalize = False


    def __len__(self):
        return len(self.gt_positions)

    def __getitem__(self, idx):
        # Load the target and context images
        target_img = Image.open(f'{self.target_dir}/target_{idx+1}.jpg')
        stimuli_img = Image.open(f'{self.stimuli_dir}/array_{(idx+1)%300}.jpg')

        # Load the ground truth position
        gtpos = self.gt_positions.loc[self.gt_positions["Trial"] == idx+1, "Ground Truth Position"].iloc[0]


        # Convert the images to tensors
        target_img = to_tensor(target_img)
        context_img = to_tensor(stimuli_img)

        # Normalize the images
        if self.normalize:
            transforms.Normalize(target_img, self.normalize_means, self.normalize_stds)
            transforms.Normalize(context_img, self.normalize_means, self.normalize_stds)

        return context_img, target_img, (max(image_positions[gtpos][0] - self.target_size/2, 0) , max(image_positions[gtpos][1] - self.target_size/2,0), self.target_size, self.target_size)

