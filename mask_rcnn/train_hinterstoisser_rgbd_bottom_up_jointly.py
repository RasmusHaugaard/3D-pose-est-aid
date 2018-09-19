import os
import sys
import numpy as np

sys.path.append('./Mask_RCNN')
from mrcnn import utils
from mrcnn import model as modellib
from hinterstoisser import HinterstoisserConfig, HinterstoisserDatasetRGBD


class Config(HinterstoisserConfig):
    NAME = "hinterstoisser_rgbd_bottom_up_jointly"
    IMAGE_CHANNELS = 4
    MEAN_PIXELS = np.array([123.7, 116.8, 103.9, 0.0])

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
