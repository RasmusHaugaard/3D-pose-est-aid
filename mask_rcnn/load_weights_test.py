import os
import sys

sys.path.append('./Mask_RCNN')
from mrcnn import utils
from mrcnn import model as modellib
from hinterstoisser_rgbd import HinterstoisserConfig, HinterstoisserDataset

os.environ['CUDA_VISIBLE_DEVICES'] = ''

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

dataset_dir = project_dir + '/datasets/sixd/hinterstoisser'

DEPTH_MODEL_PATH = file_dir + '/mask_rcnn_hinterstoisser_0100_secondary.h5'
COCO_MODEL_PATH = file_dir + '/Mask_RCNN/mask_rcnn_coco.h5'
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

exclude=["mrcnn.*", "fpn.*", "rpn.*"]

config = HinterstoisserConfig()
model = modellib.MaskRCNN(mode='training', config=config, model_dir='logs')

exclude_coco = exclude
print('LOAD COCO:', exclude_coco)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=exclude_coco)

exclude_depth = exclude
print('LOAD DEPTH:', exclude_depth)
model.load_weights(DEPTH_MODEL_PATH, by_name=True, exclude=exclude_depth)
