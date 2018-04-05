"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import numpy as np
import cv2
import os
import skimage.color
import skimage.io

from config import Config
import utils

class SixdConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "sixd"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    #IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 128
    #IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 5
    
    
class SixdDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_sixd(self, count, coco_dataset_dir, coco_subset, coco_year = DEFAULT_DATASET_YEAR, sixd_dataset_dir):
        # Add classes
        self.add_class("sixd", 1, "cup")
        self.add_class("sixd", 2, "carton")
        
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        image_ids = np.random.choice(list(coco.imgs.keys()), count)

        for coco_id in image_ids:
            instance_paths, class_ids = choose_random_objects(sixd_dataset_dir)
            bg_path = os.path.join(image_dir, coco.imgs[coco_id]['file_name'])
            image_def = generate_image_def(bg_path, instance_paths)
            
            self.add_image(
                "sixd", image_id=coco_id,
                path=bg_path,
                width=coco.imgs[coco_id]["width"],
                height=coco.imgs[coco_id]["height"], image_def, class_ids)
        
            
    def load_image(self, image_id):
        # Load image
        info = self.image_info[image_id]
        image_def = info['image_def']
        image, _, _ = generate_image(image_def)

        return image

    def image_reference(self, image_id):
        """Return the image_def data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sixd":
            return info["sixd"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for cups/cartons of the given image ID.
        """
        info = self.image_info[image_id]
        image_def, class_ids = info['image_def'], info['class_ids']
        _, _, mask = generate_image(image_def)
        return mask, class_ids.astype(np.int32)