#!/usr/bin/env python3
"""
Object detection through OpenCV's DNN module with a user-supplied ONNX model.
Supports the YOLO export layouts: (1, 4+classes, N) from YOLOv8/YOLO11 and (1, N, 5+classes) from YOLOv5. No model file ships with EyeMap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO_CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,"
    "parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,"
    "tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,"
    "tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,"
    "hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,"
    "keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,"
    "hair drier,toothbrush"
).split(",")


@dataclass
class Detection:
    """One detected object in pixel coordinates of the original frame."""

    label: str
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        """Return the box center (u, v)."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def load_class_names(path: str | Path) -> list[str]:
    """Read class names from a text file, one per line, in model output order."""
    names_path = Path(path)
    if not names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {names_path}")
    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Class names file is empty: {names_path}")
    return names


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square; return (padded, scale, (pad_x, pad_y))."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(image, (new_w, new_h))
    return padded, scale, (pad_x, pad_y)


def decode_yolo_output(
    output: np.ndarray,
    class_names: list[str],
    scale: float,
    pad: tuple[int, int],
    image_size: tuple[int, int],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Turn a raw YOLO output tensor into detections in original-image pixels.

    image_size is (width, height). Rows are (cx, cy, w, h, [objectness,] class scores...).
    """
    rows = np.asarray(output, dtype=np.float32)
    while rows.ndim > 2 and rows.shape[0] == 1:  # drop the batch axis only, never a real one
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"Unsupported detector output shape: {np.shape(output)}")
    n_classes = len(class_names)
    if rows.shape[1] not in (4 + n_classes, 5 + n_classes) and rows.shape[0] in (4 + n_classes, 5 + n_classes):
        rows = rows.T
    if rows.shape[1] == 5 + n_classes:  # YOLOv5 layout with an objectness column
        scores = rows[:, 5:] * rows[:, 4:5]
    elif rows.shape[1] == 4 + n_classes:
        scores = rows[:, 4:]
    else:
        raise ValueError(
            f"Detector output {np.shape(output)} does not match {n_classes} class names; "
            "pass the model's class list with --detect-classes."
        )

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []
    boxes_cxcywh, class_ids, confidences = rows[keep, :4], class_ids[keep], confidences[keep]

    width, height = image_size
    x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0 - pad[0]) / scale
    y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0 - pad[1]) / scale
    x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0 - pad[0]) / scale
    y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0 - pad[1]) / scale
    x1, x2 = np.clip(x1, 0, width), np.clip(x2, 0, width)
    y1, y2 = np.clip(y1, 0, height), np.clip(y2, 0, height)

    detections: list[Detection] = []
    for class_id in np.unique(class_ids):  # suppress overlaps within each class only
        idx = np.flatnonzero(class_ids == class_id)
        xywh = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in idx]
        kept = cv2.dnn.NMSBoxes(xywh, [float(confidences[i]) for i in idx], conf_threshold, iou_threshold)
        for k in np.asarray(kept).reshape(-1):
            i = idx[int(k)]
            if x2[i] - x1[i] < 1.0 or y2[i] - y1[i] < 1.0:
                continue
            detections.append(
                Detection(
                    class_names[int(class_id)],
                    int(class_id),
                    float(confidences[i]),
                    (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                )
            )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class OnnxDetector:
    """Run a YOLO-style ONNX detector with cv2.dnn."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str] | None = None,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        net: Any = None,
    ) -> None:
        """Load the model. `class_names` defaults to the 80 COCO classes; `net` is injectable for tests."""
        self.class_names = list(class_names) if class_names else list(COCO_CLASSES)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        if net is None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Detection model not found: {path}")
            net = cv2.dnn.readNetFromONNX(str(path))
        self.net = net

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame, most confident first."""
        padded, scale, pad = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        height, width = image.shape[:2]
        return decode_yolo_output(
            output, self.class_names, scale, pad, (width, height), self.conf_threshold, self.iou_threshold
        )
