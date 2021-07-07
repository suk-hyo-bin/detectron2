import torch, torchvision
torch.__version__
import detectron2
import pdb

from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("fruits_nuts", {}, "/data/workspace/hyobin/data/trainval.json", "/data/workspace/hyobin/data/images")
# #pdb.set_trace()
# fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
# dataset_dicts = DatasetCatalog.get("fruits_nuts")

from detectron2.data.datasets.geococo import load_geococo_dicts
dataset_dicts = load_geococo_dicts("/data/datasets/yet_another_DOTAv2/GeoCOCO/Train", "/data/datasets/yet_another_DOTAv2/GeoCOCO/Train/GeoCOCO.json")

# import random
# a = 0

# for d in random.sample(dataset_dicts, 2):
#     a= a+1
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imwrite(('test %d.jpg' % a),vis.get_image()[:, :, ::-1])
#     #cv2.imshow('test.jpg',vis.get_image()[:, :, ::-1])

print("finish")
#pdb.set_trace()
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
#cfg.merge_from_file("/data/workspace/hyobin/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
cfg.merge_from_file("/data/workspace/hyobin/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TRAIN = ("dota_train",)

cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00002
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 50  # 3 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#pdb.set_trace()
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

