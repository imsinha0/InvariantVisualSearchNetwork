
#here we implement a dataloader to read in coco18 data
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
import json
import random


class COCO_TPDataset(Dataset):

    def __init__(self, normalize):
        self.normalize = normalize
        self.data_dir = "coco18Data/coco_search18_images_TP"

        # Load JSON data once during initialization
        json_path = "coco18Data/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json"
        with open(json_path, 'r') as f:
            self.fixation_data = json.load(f)
            
        # Create lookup dictionary for faster access
        self.fixation_lookup = {entry["name"]: entry for entry in self.fixation_data}
        
        # Pre-index all files
        self.image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Only include image files
                    if file not in self.fixation_lookup:
                        continue
                    full_path = os.path.join(root, file)
                    self.image_files.append((full_path, file))

    
        
        # Define normalization transform
        if self.normalize:
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path and filename
        image_path, image_filename = self.image_files[idx]
        
        # Load and convert image to tensor
        image = Image.open(image_path).convert('RGB')
        image_tensor = to_tensor(image)
        
        # Apply normalization if requested
        if self.normalize:
            image_tensor = self.normalize_transform(image_tensor)
        
        # Get matching fixation data
        matching_data = self.fixation_lookup.get(image_filename, None)
        if matching_data is None:
            raise ValueError(f"No fixation data found for image {image_filename}")
            
        return image_tensor, matching_data
    


class COCO_TADataset(Dataset):

    def __init__(self, normalize):
        self.normalize = normalize
        self.data_dir = "coco18Data/coco_search18_images_TA"

        # Load JSON data once during initialization
        json_path = "coco18Data/coco18TAfixations.json"
        with open(json_path, 'r') as f:
            self.fixation_data = json.load(f)

        # Create lookup dictionary for faster access
        self.fixation_lookup = {entry["name"]: entry for entry in self.fixation_data}
        
        # Pre-index all files
        self.image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Only include image files
                    if file not in self.fixation_lookup:
                        continue
                    full_path = os.path.join(root, file)
                    self.image_files.append((full_path, file, root.rsplit("/", 1)[-1]))
            

        
        # Define normalization transform
        if self.normalize:
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path and filename
        image_path, image_filename, root = self.image_files[idx]
        
        # Load and convert image to tensor
        image = Image.open(image_path).convert('RGB')
        image_tensor = to_tensor(image)
        
        # Apply normalization if requested
        if self.normalize:
            image_tensor = self.normalize_transform(image_tensor)
        
        # Get matching fixation data
        matching_data = self.fixation_lookup.get(image_filename, None)
        if matching_data is None:
            raise ValueError(f"No fixation data found for image {image_filename}")
        
        #Get the target image by selecting random image from root directory in targetPics

        
        target_pics_dir = os.path.join("coco18Data", "targetPics", root)
        target_images = [f for f in os.listdir(target_pics_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not target_images:
            raise ValueError(f"No target images found in directory {target_pics_dir}")
        
        random_target_image = random.choice(target_images)
        target_image_path = os.path.join(target_pics_dir, random_target_image)
        
        # Load the target image
        target_image = Image.open(target_image_path).convert('RGB')
        target_image_tensor = to_tensor(target_image)
        
            
        return image_tensor, target_image_tensor, matching_data
