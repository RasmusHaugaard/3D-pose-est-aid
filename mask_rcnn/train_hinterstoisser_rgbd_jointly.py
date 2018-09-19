import os
import sys
import numpy as np
import skimage
from pathlib import Path

sys.path.append('./Mask_RCNN')
from mrcnn import utils
from mrcnn import model as modellib
from hinterstoisser import HinterstoisserConfig, HinterstoisserDatasetRGB


class Config(HinterstoisserConfig):
    NAME = "hinterstoisser_rgbd_bottom_up_jointly"
    IMAGE_CHANNELS = 4
    MEAN_PIXELS = np.zeros(4)


class HinterstoisserDatasetRGBD(HinterstoisserDatasetRGB):
    def load_image(self, image_id):
        info = self.image_info[image_id]
        rgb = skimage.io.imread(info['path'])
        depth = skimage.io.imread(str(
            Path(info['path']).parent.parent / 'depth' / '{:04}.png'.format(info['id'])
        )).reshape((*rgb.shape[:2], 1))
        depth = (depth.astype(np.float32) - 950.) / 460.
        rgb = (rgb.astype(np.float32) - [84., 79., 78.]) / 62.
        return np.concatenate((rgb, depth), axis=-1)


file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_dir = project_dir + '/datasets/sixd/hinterstoisser'

dataset_train = HinterstoisserDatasetRGBD()
dataset_train.load_hinterstoisser(dataset_dir, 2, 'train_set')
dataset_train.prepare()

dataset_val = HinterstoisserDatasetRGBD()
dataset_val.load_hinterstoisser(dataset_dir, 2, 'val_set')
dataset_val.prepare()

config = Config()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='all')
