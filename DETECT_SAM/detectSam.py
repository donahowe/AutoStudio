from typing import List

import io
import time

import os
current_file_path = os.path.abspath(__file__)
dpath = os.path.dirname(current_file_path)

import sys
sys.path.append(f"{dpath}/YOLO-World/")
sys.path.append(f"{dpath}/Grounding-DINO/")

import cv2
import time
import contextlib
import numpy as np
from PIL import Image
import supervision as sv
import torch

from groundingdino.util.inference import Model, load_image

#from mmengine.runner import Runner
#from mmengine.config import Config
#from mmengine.dataset import Compose
#from yoloWorld import run_image, parse_args

from tqdm import tqdm
from efficient_sam import load, inference_with_boxes

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    RESULTS = "results"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EFFICIENT_SAM_MODEL = load(device=DEVICE)
    GROUNDING_DINO_MODEL = Model(f"{dpath}/Grounding-DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{dpath}/Grounding-DINO/groundingdino_swint_ogc.pth", DEVICE)

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
    MASK_ANNOTATOR = sv.MaskAnnotator()
    LABEL_ANNOTATOR = sv.LabelAnnotator()

def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

create_directory(directory_path=RESULTS)

def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]

def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
) -> np.ndarray:
    def trans(class_id):
        if isinstance(class_id, np.ndarray):
            class_id[class_id == None] = 0
        elif isinstance(class_id, list):
            class_id = [trans(i) for i in class_id]
        elif class_id is None:
            class_id = 0
        return class_id
    detections.class_id = trans(detections.class_id)
    labels = [
        (
            f"{categories[trans(class_id)]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[trans(class_id)]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_image(
    detect_model,
    same_model,
    input_image,
    categories: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    with_segmentation: bool = True,
    with_confidence: bool = False,
    device = DEVICE
):
    
    categories = process_categories(categories)
    input_image, _ = load_image(input_image)#, type='image')

    if device != DEVICE:
        detect_model.model = detect_model.model.to(device)

    detections = detect_model.predict_with_classes(
        image=input_image,
        classes=categories,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if with_segmentation:
        detections.mask = inference_with_boxes(
            image=input_image,
            xyxy=detections.xyxy,
            model=same_model,
            device=device
        )
    output_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence
    )
    output_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)) 
    return output_image, detections


if __name__ == '__main__':
    # For Test
    box_threshold = 0.05
    text_threshold = 0.05
    with_segmentation = True
    with_confidence = True
    with_class_agnostic_nms = False

    input_image = Image.open(f"{dpath}/Grounding-DINO/assert/test1.png")
    image_categories_text = "girl"

    output_image, _=process_image(
        GROUNDING_DINO_MODEL,        #YOLO_RUNNER,
        EFFICIENT_SAM_MODEL,
        input_image,
        image_categories_text,
        box_threshold,
        text_threshold,
        with_segmentation,
        with_confidence
    )
    output_image.save(f'{dpath}/Grounding-DINO/assert/result1.png')
    ##cv2.imwrite(f'{dpath}/output.png', output_image)
