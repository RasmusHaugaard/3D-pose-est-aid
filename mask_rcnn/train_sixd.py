import os
import sys

sys.path.append('./Mask_RCNN')
from Mask_RCNN import utils
from Mask_RCNN import model as modellib
from Sixd import SixdConfig, SixdDataset


file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_coco_dir = project_dir + '/datasets/coco'
dataset_sixd_dir = project_dir + '/datasets/sixd/doumanoglou/train'

COCO_MODEL_PATH = 'Mask_RCNN/mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

dataset_train = SixdDataset()
dataset_train.load_sixd(3000, dataset_coco_dir, 'train', dataset_sixd_dir)
dataset_train.prepare()

dataset_val = SixdDataset()
dataset_val.load_sixd(250, dataset_coco_dir, 'val', dataset_sixd_dir)
dataset_val.prepare()

config = SixdConfig()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')
