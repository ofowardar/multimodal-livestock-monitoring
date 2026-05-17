"""
vision/anomaly_detector.py
Eşik tabanlı anomali tespit modülü.

Şu an sadece hareket skoru kullanılıyor.
İleride ses (MFCC) skoru eklenerek multimodal füzyon yapılabilir:
    fused_score = alpha * motion_score + beta * audio_score

Anomali durumu 3 seviyeli:
    NORMAL  → Yeşil  (skor < WARNING eşiği)
    WARNING → Sarı   (WARNING ≤ skor < ALERT eşiği)
    ALERT   → Kırmızı (skor ≥ ALERT eşiği)
"""

from enum import Enum
from collections import deque
from typing import Tuple

import conf


class AnomalyStatus(Enum):
    NORMAL  = "NORMAL"
    WARNING = "UYARI"
    ALERT   = "ANOMALİ"


# Durum için görsel renk eşlemeleri (GUI'de kullanılır)
STATUS_COLORS = {
    AnomalyStatus.NORMAL:  "#00e676",   # yeşil
    AnomalyStatus.WARNING: "#ffc107",   # sarı/turuncu
    AnomalyStatus.ALERT:   "#f44336",   # kırmızı
}

STATUS_ICONS = {
    AnomalyStatus.NORMAL:  "●",
    AnomalyStatus.WARNING: "⚠",
    AnomalyStatus.ALERT:   "🚨",
}


class AnomalyDetector:
    """
    Hareket (ve ileride ses) skoruna göre anomali durumu hesaplar.

    Ani tepkileri önlemek için 'confirm_frames' mekanizması vardır:
    Anomali durumu yalnızca ardışık N kare boyunca koşul sağlanırsa güncellenir.
    """

    def __init__(self):
        self._status = AnomalyStatus.NORMAL
        self._reason = "Normal aktivite"
        # Son N karedeki raw durum geçmişi (confirm için)
        self._raw_history: deque = deque(maxlen=conf.ANOMALY_CONFIRM_FRAMES)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def update(
        self,
        avg_motion_score: float,
        audio_score: float = 0.0,       # Ses modülü eklendiğinde dolu gelecek
        alpha: float = 1.0,             # Hareket ağırlığı
        beta:  float = 0.0,             # Ses ağırlığı (şimdilik 0)
    ) -> Tuple[AnomalyStatus, str]:
        """
        Anomali durumunu güncelle ve döndür.

        Args:
            avg_motion_score: Sürünün ortalama hareket skoru [0-1]
            audio_score:      Ses anomali skoru [0-1] — şimdilik 0
            alpha:            Hareket ağırlığı
            beta:             Ses ağırlığı

        Returns:
            (AnomalyStatus, açıklama_metni)
        """
        # Füzyon (şimdilik sadece motion)
        fused = alpha * avg_motion_score + beta * audio_score

        # Ham durum hesapla
        if fused >= conf.MOTION_ALERT_THRESHOLD:
            raw = AnomalyStatus.ALERT
            reason = f"Yüksek aktivite tespit edildi (skor: {fused:.2f})"
        elif fused >= conf.MOTION_WARNING_THRESHOLD:
            raw = AnomalyStatus.WARNING
            reason = f"Orta düzey aktivite (skor: {fused:.2f})"
        else:
            raw = AnomalyStatus.NORMAL
            reason = f"Normal aktivite (skor: {fused:.2f})"

        self._raw_history.append(raw)

        # Confirm mekanizması: son N karenin tamamı aynı duruma işaret ediyorsa güncelle
        if len(self._raw_history) == conf.ANOMALY_CONFIRM_FRAMES:
            if all(s == raw for s in self._raw_history):
                self._status = raw
                self._reason = reason
        elif raw.value > self._status.value:
            # Acil yükseltme — ALERT anında göster
            self._status = raw
            self._reason = reason

        return self._status, self._reason

    def reset(self):
        """Anomali durumunu sıfırla (video değiştiğinde çağrılır)."""
        self._status = AnomalyStatus.NORMAL
        self._reason = "Normal aktivite"
        self._raw_history.clear()

    @property
    def status(self) -> AnomalyStatus:
        return self._status

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def color(self) -> str:
        """Güncel durumun hex rengi."""
        return STATUS_COLORS[self._status]

    @property
    def icon(self) -> str:
        """Güncel durumun ikonu."""
        return STATUS_ICONS[self._status]
