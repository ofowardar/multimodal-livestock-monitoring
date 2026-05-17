"""
vision/detector.py
YOLOv8 + ByteTrack tabanlı hayvan tespit ve takip modülü.

Her kare için tespit edilen hayvanları track_id ile birlikte döndürür.
Raspberry Pi 5 optimizasyonu: verbose=False, persist=True (tracker state saklanır).
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

import conf


@dataclass
class Detection:
    """Tek bir hayvana ait tespit sonucu."""
    track_id: int           # ByteTrack tarafından atanan kalıcı ID
    bbox: tuple             # (x1, y1, x2, y2) piksel koordinatları
    class_id: int           # COCO sınıf ID'si
    class_name: str         # Türkçe sınıf adı (conf.py'den)
    confidence: float       # Tespit güven skoru [0-1]
    center: tuple           # Bounding box merkezi (cx, cy)


class AnimalDetector:
    """
    YOLOv8 ile hayvan tespiti ve ByteTrack ile takip sınıfı.

    Kullanım:
        detector = AnimalDetector()
        detections = detector.detect(frame)
    """

    def __init__(self):
        print(f"[Detector] Model yükleniyor: {conf.MODEL_PATH}")
        self.model = YOLO(conf.MODEL_PATH)
        print("[Detector] Model hazır.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Verilen karede hayvan tespiti ve takip yapar.

        Args:
            frame: BGR formatında numpy array (OpenCV frame)

        Returns:
            Detection listesi. Track ID alamamış nesneler dahil edilmez.
        """
        results = self.model.track(
            source=frame,
            persist=True,               # Tracker state'i kareler arasında sakla
            tracker="bytetrack.yaml",   # ByteTrack algoritması
            conf=conf.CONFIDENCE_THRESHOLD,
            iou=conf.IOU_THRESHOLD,
            classes=conf.ANIMAL_CLASS_IDS,
            verbose=False,              # Raspberry Pi'de konsol çıktısını azalt
            stream=False,
        )

        detections: List[Detection] = []

        if results is None or len(results) == 0:
            return detections

        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections

        for box in boxes:
            # Track ID yoksa atla (tracker henüz ID atamamış)
            if box.id is None:
                continue

            track_id    = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls         = int(box.cls.item())
            conf_score  = float(box.conf.item())
            class_name  = conf.ANIMAL_CLASS_NAMES.get(cls, "Hayvan")
            cx          = (x1 + x2) // 2
            cy          = (y1 + y2) // 2

            detections.append(Detection(
                track_id   = track_id,
                bbox       = (x1, y1, x2, y2),
                class_id   = cls,
                class_name = class_name,
                confidence = conf_score,
                center     = (cx, cy),
            ))

        return detections
