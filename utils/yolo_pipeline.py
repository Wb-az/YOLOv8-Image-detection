"""
Class for object detection in optimize openvino model
"""

import random
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
import cv2


from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.plotting import colors

import torch
import openvino as ov


class YoloPipeline(object):
    """
    A YOLOV8 detection ousing openvino optomized models
    """

    def __init__(self, ov_model=None, path=None, device=None, labels=None):

        self.initialized = False

        # Parameters for inference
        self.ov_model = ov_model
        self.quantized = False

        self.ov_path = None
        self.device = device
        # Parameters for pre-processing
        self.frame = None
        self.path = path
        self.imgsz = 640
        self.stride = 32

        # Parameters for post-processing
        self.conf = 0.4
        self.iou = 0.70
        self.max_det = 300
        self.agnostic = False
        self.labels = labels

    def initialize(self):

        if self.ov_model is None:
            core = ov.Core()
            self.initialized = True
            self.ov_model = core.read_model(self.ov_path)
            if self.device != "CPU":
                self.ov_model.reshape({0: [1, 3, self.imgsz, self.imgsz]})
            self.ov_model = core.compile_model(self.ov_model, self.device)

        return self.ov_model

    def preprocess_image(self):

        # Load an image or a frame (image array)
        if self.frame is not None:
            img0 = self.frame

        else:
            img0 = Image.open(self.path).convert("RGB")
            img0 = ImageOps.exif_transpose(img0)
            img0 = np.array(img0)

        # Resize image
        img = LetterBox(self.imgsz, stride=self.stride)(image=img0.copy())
        # Convert HWC to CHW
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32)  # uint8 to fp/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        return img0, img

    def inference(self, model_input):

        compiled_model = self.initialize()
        start_time = datetime.now()
        result = compiled_model(model_input)
        end_time = datetime.now()

        return result, (end_time - start_time).total_seconds()

    def postprocess(self, img0, img, inference_output):

        boxes = inference_output[0]

        predictions = torch.from_numpy(boxes)
        preds = ops.non_max_suppression(predictions,
                                        conf_thres=self.conf,
                                        iou_thres=self.iou,
                                        agnostic=self.agnostic,
                                        max_det=self.max_det,
                                        nc=len(self.labels))
        log_string = ''
        results = []
        for _, pred in enumerate(preds):

            if len(pred) == 0:
                results.append({"det": []})
                continue
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
            results.append({"det": pred})

        if not results:
            return log_string + 'No detection found.'
        return results

    def detect(self):
        preprocessed_data = self.preprocess_image()
        org_input, model_input = preprocessed_data
        inference_output, time = self.inference(model_input)
        detections = self.postprocess(org_input, model_input, inference_output)
        return org_input, detections, time


def plot_one_box(box, img, color, label, line_thickness=5):

    tl = max(line_thickness,
             round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)

    return img


def draw_results(results, source_image, labels=None):
    """
    Helper function for drawing bounding boxes on the image
    Parameters:
        results (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id] \
        source_image (np.ndarray): input image for drawing label_map; (Dict[int, str]): \
        labels (dict): to class name mapping
    Returns:
        Image with boxes
    """

    boxes = results["det"]

    for (*xyxy, conf, lb) in boxes:
        label = f'{labels[int(lb)]} {conf:.2f}'
        source_image = plot_one_box(xyxy, source_image, label=label, color=colors(int(lb)),
                                    line_thickness=1)
    return source_image