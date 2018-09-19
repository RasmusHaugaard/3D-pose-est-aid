import os
import sys

sys.path.append('./Mask_RCNN')
from mrcnn import utils
from mrcnn import model as modellib
from hinterstoisser import HinterstoisserConfig, HinterstoisserDatasetRGBD


class Config(HinterstoisserConfig):
    NAME = "hinterstoisser_rgbd_pretrained_rgb_d"
    SECONDARY_MODE = "before_rpn"
    SECONDARY_CHANNELS = 1


file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_dir = project_dir + '/datasets/sixd/hinterstoisser'

DEPTH_MODEL_PATH = file_dir + '/mask_rcnn_hinterstoisser_0100_secondary.h5'
COCO_MODEL_PATH = file_dir + '/Mask_RCNN/mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

dataset_train = HinterstoisserDatasetRGBD()
dataset_train.load_hinterstoisser(dataset_dir, 2, 'train_set')
dataset_train.prepare()

dataset_val = HinterstoisserDatasetRGBD()
dataset_val.load_hinterstoisser(dataset_dir, 2, 'val_set')
dataset_val.prepare()

exclude=["mrcnn.*", "fpn.*", "rpn.*"]

config = Config()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=exclude)
model.load_weights(DEPTH_MODEL_PATH, by_name=True, exclude=exclude)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads')
