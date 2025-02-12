
#here we implement a dataloader to read in coco18 data
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
import json

class COCO_TPDataset(Dataset):

    def __init__(self, normalize):
        self.normalize = normalize
        self.data_dir = "coco18Data/coco_search18_images_TP"
        
        # Pre-index all files
        self.image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Only include image files
                    full_path = os.path.join(root, file)
                    self.image_files.append((full_path, file))
        
        # Load JSON data once during initialization
        json_path = "coco18Data/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json"
        with open(json_path, 'r') as f:
            self.fixation_data = json.load(f)
            
        # Create lookup dictionary for faster access
        self.fixation_lookup = {entry["name"]: entry for entry in self.fixation_data}
        
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