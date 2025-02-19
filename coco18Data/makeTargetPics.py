

import os
import json
from PIL import Image

# Define directories
target_present_dir = 'coco18Data/coco_search18_images_TP'
fixation_file_path = 'coco18Data/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json'
fixation_file_path2 = 'coco18Data/COCOSearch18-fixations-TP/coco_search18_fixations_TP_validation_split1.json'
target_pics_dir = 'coco18Data/targetPics'


with open(fixation_file_path, 'r') as f:
    fixations = json.load(f)

with open(fixation_file_path2, 'r') as f:
    fixations2 = json.load(f)


bboxes = {}

for fixation in fixations:
    if fixation['name'] not in bboxes:
        bboxes[fixation['name']] = fixation['bbox']
for fixation in fixations2:
    if fixation['name'] not in bboxes:
        bboxes[fixation['name']] = fixation['bbox']

for subdir, _, files in os.walk(target_present_dir):
    for image_name in files:
        if image_name.endswith('.jpg'):
            if image_name not in bboxes:
                continue
            print(image_name)
            bbox = bboxes[image_name]


            image_path = os.path.join(subdir, image_name)
            image = Image.open(image_path)

            #bbox has format (x_min, y_min, width, height)

            # Crop the image using the bbox
            cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

            # Create a subdirectory in targetPics with the same name as the image's subdirectory
            subdir_name = os.path.basename(subdir)
            target_subdir = os.path.join(target_pics_dir, subdir_name)
            os.makedirs(target_subdir, exist_ok=True)

            # Save the cropped image
            cropped_image.save(os.path.join(target_subdir, image_name))
