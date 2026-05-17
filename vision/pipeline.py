"""
vision/pipeline.py
Vision Pipeline — Tespit, takip, hareket analizi ve anomali tespitini birleştirir.

Tek giriş noktası: pipeline.process_frame(frame)
    → (annotated_frame, output_dict)

output_dict şeması:
    {
        "detections":      List[Detection],
        "motion_scores":   Dict[int, float],   # {track_id: score}
        "avg_motion_score": float,
        "animal_count":    int,
        "anomaly_status":  AnomalyStatus,
        "anomaly_reason":  str,
        "anomaly_color":   str,                # hex renk
        "anomaly_icon":    str,
    }
"""

from typing import Tuple, Dict, Any

import cv2
import numpy as np

from .detector import AnimalDetector, Detection
from .motion_analyzer import MotionAnalyzer
from .anomaly_detector import AnomalyDetector, AnomalyStatus, STATUS_COLORS


class VisionPipeline:
    """
    Tüm görüntü işleme adımlarını koordine eden ana sınıf.

    Kullanım:
        pipeline = VisionPipeline()
        annotated, output = pipeline.process_frame(frame)
    """

    def __init__(self):
        self.detector        = AnimalDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.anomaly_detector = AnomalyDetector()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Tek bir kareyi işle.

        Args:
            frame: BGR formatında numpy array

        Returns:
            (annotated_frame, output_dict)
        """
        # 1. Tespit + Takip
        detections = self.detector.detect(frame)

        # 2. Hareket analizi
        motion_scores  = self.motion_analyzer.update(detections)
        avg_motion     = self.motion_analyzer.get_avg_score(motion_scores)

        # 3. Anomali tespiti
        anomaly_status, anomaly_reason = self.anomaly_detector.update(avg_motion)

        # 4. Görsel çizimler
        annotated = self._draw_annotations(
            frame.copy(), detections, motion_scores, anomaly_status
        )

        output = {
            "detections":       detections,
            "motion_scores":    motion_scores,
            "avg_motion_score": avg_motion,
            "animal_count":     len(detections),
            "anomaly_status":   anomaly_status,
            "anomaly_reason":   anomaly_reason,
            "anomaly_color":    self.anomaly_detector.color,
            "anomaly_icon":     self.anomaly_detector.icon,
        }

        return annotated, output

    def reset(self):
        """Pipeline state'ini sıfırla (yeni video açıldığında)."""
        self.motion_analyzer  = MotionAnalyzer()
        self.anomaly_detector.reset()

    # ──────────────────────────────────────────
    # Görsel Çizimler
    # ──────────────────────────────────────────

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections,
        motion_scores: Dict[int, float],
        anomaly_status: AnomalyStatus,
    ) -> np.ndarray:
        """Bounding box, track ID, hareket skoru ve skor barını çiz."""

        # Anomali durumuna göre renk seç (hex → BGR)
        hex_color  = STATUS_COLORS[anomaly_status]
        box_color  = self._hex_to_bgr(hex_color)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            score = motion_scores.get(det.track_id, 0.0)

            # Bireysel hareket skoruna göre renk
            ind_color = self._score_to_bgr(score)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), ind_color, 2)

            # Üst etiket: "ID:3 İnek | M:0.42"
            label = f"ID:{det.track_id} {det.class_name} | M:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
            )
            label_y = max(y1, th + 10)

            # Etiket arka planı
            cv2.rectangle(
                frame,
                (x1, label_y - th - 8),
                (x1 + tw + 6, label_y - 2),
                ind_color,
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 3, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (10, 10, 10), 1, cv2.LINE_AA,
            )

            # Bounding box altında küçük hareket çubuğu
            bar_w = x2 - x1
            filled = int(bar_w * score)
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 7), (40, 40, 40), -1)
            if filled > 0:
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 7), ind_color, -1)

        # Sağ üst köşede genel durum overlay
        self._draw_status_overlay(frame, anomaly_status, box_color)

        return frame

    def _draw_status_overlay(self, frame, status: AnomalyStatus, color):
        """Sağ üst köşeye anomali durum etiketi çiz."""
        h, w = frame.shape[:2]
        label = f"  {status.value}  "
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        x1 = w - tw - 15
        y1 = 10
        x2 = w - 10
        y2 = th + 22

        # Yarı saydam arka plan (siyah)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), color, 2)
        cv2.putText(frame, label, (x1, y2 - 5), font, scale, color, thickness, cv2.LINE_AA)

    # ──────────────────────────────────────────
    # Yardımcı Dönüşümler
    # ──────────────────────────────────────────

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        """#RRGGBB → (B, G, R)"""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    @staticmethod
    def _score_to_bgr(score: float):
        """Hareket skoruna göre yeşil→sarı→kırmızı gradyanı."""
        if score < 0.35:
            return (30, 230, 100)    # Yeşil
        elif score < 0.65:
            return (30, 165, 255)    # Turuncu
        else:
            return (50, 50, 240)     # Kırmızı
