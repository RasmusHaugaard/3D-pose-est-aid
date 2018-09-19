import os
import sys

sys.path.append('./Mask_RCNN')
from mrcnn import utils
from mrcnn import model as modellib
from hinterstoisser import HinterstoisserConfig, HinterstoisserDatasetRGB


class HinterstoisserConfigRGB(HinterstoisserConfig):
    NAME = 'hinterstoisser_rgb_coco'

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_dir = project_dir + '/datasets/sixd/hinterstoisser'

dataset_train = HinterstoisserDatasetRGB()
dataset_train.load_hinterstoisser(dataset_dir, 2, 'train_set')
dataset_train.prepare()

dataset_val = HinterstoisserDatasetRGB()
dataset_val.load_hinterstoisser(dataset_dir, 2, 'val_set')
dataset_val.prepare()


COCO_MODEL_PATH = file_dir + '/Mask_RCNN/mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = HinterstoisserConfigRGB()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')

model.load_weights(COCO_MODEL_PATH, by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads')
