import os
import sys

sys.path.append('./Mask_RCNN')
from Mask_RCNN import utils
from Mask_RCNN import model as modellib
from Overlay3D import Overlay3DConfig, Overlay3DDataset


file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_dir = project_dir + '/datasets/sixd/overlay_3D/4k'

COCO_MODEL_PATH = file_dir + '/Mask_RCNN/mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

dataset_train = Overlay3DDataset()
dataset_train.load_overlay3d(dataset_dir + '/train')
dataset_train.prepare()

dataset_val = Overlay3DDataset()
dataset_val.load_overlay3d(dataset_dir + '/val')
dataset_val.prepare()

config = Overlay3DConfig()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads')
