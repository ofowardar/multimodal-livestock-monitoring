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
        self.track_class_history = {}  # track_id -> son 4 karedeki class_id listesi
        self.track_bbox_history = {}   # track_id -> son pürüzsüzleştirilmiş bbox (x1, y1, x2, y2)
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

        if boxes is None:
            return detections

        # Bellek temizliği için aktif ID'leri topla
        active_ids = set()

        # Geçici takip dışı ID sayacı
        temp_id_counter = -1

        for box in boxes:
            x1_raw, y1_raw, x2_raw, y2_raw = map(int, box.xyxy[0].tolist())
            cls         = int(box.cls.item())
            conf_score  = float(box.conf.item())
            
            # --- Takip ID Kontrolü ve Fallback ---
            if box.id is not None:
                track_id = int(box.id.item())
                active_ids.add(track_id)
            else:
                # Takipçi henüz ID atayamamışsa (örn: çok hızlı koşan inek), tespiti kaybetmemek için geçici ID atıyoruz
                track_id = temp_id_counter
                temp_id_counter -= 1

            # --- 1. BBOX EMA YUMUŞATMA FİLTRESİ (Titreme Engelleyici) ---
            alpha = 0.55  # 0.0 (tam durağan) - 1.0 (filtresiz ham koordinat)
            if track_id in self.track_bbox_history:
                prev_x1, prev_y1, prev_x2, prev_y2 = self.track_bbox_history[track_id]
                x1 = int(alpha * x1_raw + (1 - alpha) * prev_x1)
                y1 = int(alpha * y1_raw + (1 - alpha) * prev_y1)
                x2 = int(alpha * x2_raw + (1 - alpha) * prev_x2)
                y2 = int(alpha * y2_raw + (1 - alpha) * prev_y2)
            else:
                x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw
            
            # Güncel pürüzsüz koordinatları geçmişe yaz
            self.track_bbox_history[track_id] = (x1, y1, x2, y2)

            # --- 2. SINIF STABİLİZASYONU (Çoğunluk Oylaması Filtresi) ---
            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            
            self.track_class_history[track_id].append(cls)
            if len(self.track_class_history[track_id]) > 4:  # Son 4 kareyi sakla
                self.track_class_history[track_id].pop(0)
            
            # Son karelerdeki en popüler sınıfı bul
            smoothed_cls = max(set(self.track_class_history[track_id]), key=self.track_class_history[track_id].count)
            class_name  = conf.ANIMAL_CLASS_NAMES.get(smoothed_cls, "Hayvan")
            
            cx          = (x1 + x2) // 2
            cy          = (y1 + y2) // 2

            detections.append(Detection(
                track_id   = track_id,
                bbox       = (x1, y1, x2, y2),
                class_id   = smoothed_cls,
                class_name = class_name,
                confidence = conf_score,
                center     = (cx, cy),
            ))

        # Bellek sızıntısını önlemek için artık etkin olmayan eski track geçmişlerini temizle
        if len(self.track_class_history) > 50:
            inactive_ids = set(self.track_class_history.keys()) - active_ids
            for inactive_id in inactive_ids:
                self.track_class_history.pop(inactive_id, None)
                self.track_bbox_history.pop(inactive_id, None)

        return detections
