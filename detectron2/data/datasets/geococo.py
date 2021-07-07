"""
GEOCOCO 데이터셋 사용에 필요한 함수입니다.
"""

import os
from typing import Dict, List

from detectron2.structures import BoxMode

#from rpointrcnn.utils import convert_angle_to_det2_format, normalize_range

from detectron2.utils.image_utils import normalize_range

def load_geococo_dicts(image_dir: str, json_path: str) -> List[Dict]:
    """
    이미지 경로와 json으로부터 데이터셋을 불러옵니다.

    @param image_dir: 이미지 경로
    @param json_path: GEOCOCO json 파일
    @return: 데이터셋
    """
    from pygeococotools.geococo import GeoCOCO

    geococo_api = GeoCOCO(json_path)

    # sort indices for reproducible results
    img_ids = sorted(geococo_api.imgs.keys())
    imgs = geococo_api.loadImgs(img_ids)
    anns = [geococo_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    dataset_dicts = list()
    for (img_dict, anno_dict_list) in imgs_anns:
        record = dict()
        record["file_name"] = os.path.join(image_dir, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]
        record["scene_id"] = img_dict["scene_meta"]["scene_id"]
        record["scene_bounds_imcoords"] = img_dict["scene_meta"]["scene_bounds_imcoords"]

        objs = list()
        for anno in anno_dict_list:
            bbox = BoxMode.convert(anno["properties"]["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            #rbox = convert_angle_to_det2_format(anno["properties"]["rbbox"])

            obj = {
                "category_id": anno["properties"]["category_id"],
                "bbox": bbox,
                #"rbox": rbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                #"rbox_mode": BoxMode.XYWHA_ABS,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# For Testing
# python3 -m rpointrcnn.dataset.geococo (at RPointRCNN/)
if __name__ == "__main__":
    import argparse
    import pickle
    import random
    from pathlib import Path

    import cv2
    import numpy as np
    from detectron2.data.catalog import Metadata
    from skimage.exposure import equalize_adapthist

    from ..visualizer import RPointVisualizer
    from .builtin import CATEGORIES, ROOT_PATH, SPLITS

    parser = argparse.ArgumentParser(description="Save Sample of Dataset")
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()

    dataset_paths = SPLITS[args.dataset_name]
    dataset_pre_name = args.dataset_name.split("_")[0]

    meta = Metadata().set(thing_classes=[c["name"] for c in CATEGORIES[dataset_pre_name]])
    dataset_dicts = load_geococo_dicts(
        image_dir=os.path.join(ROOT_PATH, dataset_paths[0]),
        json_path=os.path.join(ROOT_PATH, dataset_paths[1]),
    )

    out_path = "./sample/{}".format(args.dataset_name)
    Path(out_path).mkdir(exist_ok=True, parents=True)
    for d in random.sample(dataset_dicts, 10):
        file_path = d["file_name"] 
        if os.path.splitext(file_path)[1] == ".pkl": # if pkl file is exist
            img = normalize_range(
                pickle.load(open(file_path, "rb")), range_max=255, dst_type=np.uint8
            )
            img = (equalize_adapthist(img) * 255).astype(np.uint8)
        else:
            img = cv2.imread(file_path)[:, :, ::-1]
        visualizer = RPointVisualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        vis.save(os.path.join(out_path, os.path.basename(os.path.splitext(file_path)[0] + ".png")))