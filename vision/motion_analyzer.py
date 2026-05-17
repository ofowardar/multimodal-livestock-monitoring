"""
vision/motion_analyzer.py
Track bazlı hareket skoru hesaplama modülü.

Her hayvanın merkez koordinatları geçmiş N karede saklanır.
Ortalama kare-başı yer değiştirme, normalize edilerek [0.0–1.0] skora dönüştürülür.

İleride ses modülüyle entegrasyon için:
    motion_scores dict'i doğrudan AnomalyDetector'a verilir.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np

import conf
from .detector import Detection


class MotionAnalyzer:
    """
    Her track_id için geçmiş pozisyonları saklar ve hareket skoru hesaplar.

    Hareket Skoru Formülü:
        - Son N karenin merkez koordinatları saklanır.
        - Ardışık kareler arasındaki Öklid mesafeleri hesaplanır.
        - Ortalama piksel/kare hızı MOTION_NORMALIZE_PIXELS'e bölünür.
        - Sonuç [0.0, 1.0] aralığına kırpılır.

    Yüksek skor → hızlı/agresif hareket → stres işareti
    """

    def __init__(self):
        # {track_id: deque[(cx, cy), ...]}
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=conf.MOTION_HISTORY_FRAMES)
        )
        # Son hesaplanan skorlar (pipeline için erişilebilir)
        self._last_scores: Dict[int, float] = {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def update(self, detections: List[Detection]) -> Dict[int, float]:
        """
        Tespit listesiyle pozisyon geçmişini güncelle, hareket skorlarını döndür.

        Args:
            detections: O karedeki Detection listesi

        Returns:
            {track_id: motion_score} — sadece aktif hayvanlar
        """
        active_ids = set()

        for det in detections:
            self._history[det.track_id].append(det.center)
            active_ids.add(det.track_id)

        # Uzun süredir görülmeyen track'leri temizle
        stale_ids = set(self._history.keys()) - active_ids
        for tid in stale_ids:
            del self._history[tid]
        for tid in stale_ids:
            self._last_scores.pop(tid, None)

        # Skor hesapla
        scores: Dict[int, float] = {}
        for tid in active_ids:
            scores[tid] = self._compute_score(self._history[tid])

        self._last_scores = scores
        return scores

    def get_avg_score(self, scores: Dict[int, float]) -> float:
        """Tüm aktif hayvanların ortalama hareket skorunu döndür."""
        if not scores:
            return 0.0
        return float(np.mean(list(scores.values())))

    def get_score(self, track_id: int) -> float:
        """Belirli bir hayvanın son hareket skorunu döndür."""
        return self._last_scores.get(track_id, 0.0)

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _compute_score(self, positions: deque) -> float:
        """
        Pozisyon geçmişinden normalize hareket skoru hesapla.

        Args:
            positions: (cx, cy) tuple'larından oluşan deque

        Returns:
            float in [0.0, 1.0]
        """
        if len(positions) < 2:
            return 0.0

        pts = list(positions)
        # Ardışık çerçeveler arası mesafe toplamı
        total_dist = sum(
            float(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i])))
            for i in range(len(pts) - 1)
        )
        # Ortalama piksel/kare hızı
        avg_speed = total_dist / (len(pts) - 1)
        # Normalize et ve [0, 1]'e kırp
        score = avg_speed / conf.MOTION_NORMALIZE_PIXELS
        return float(min(score, 1.0))
