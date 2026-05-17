"""
conf.py — Global Konfigürasyon
Çiftlik Hayvanlarında Stres Tespiti Projesi

Raspberry Pi 5 ve masaüstü için ortak ayarlar.
"""

# ─────────────────────────────────────────────
# MODEL AYARLARI
# ─────────────────────────────────────────────

# YOLOv8 model ağırlıkları
# "yolov8n.pt" → Nano  (en hızlı, en düşük doğruluk)
# "yolov8s.pt" → Small (daha iyi doğruluk, Raspberry Pi 5'te ~10-14 FPS)
# "yolov8m.pt" → Medium (en iyi, Raspberry Pi'de yavaş)
MODEL_PATH = "yolov8s.pt"   # <— Small: nano'dan %20+ daha iyi AP

# Tespit güven eşiği (0.0 – 1.0)
# Düşük = daha fazla tespit ama yanlış pozitif artabilir
CONFIDENCE_THRESHOLD = 0.30

# NMS IOU eşiği
IOU_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# VİDEO KAYNAĞI
# ─────────────────────────────────────────────

# Varsayılan video kaynağı
# None  → GUI'den dosya seç
# 0     → Webcam
# "test_video/farm_test.mp4" → Test videosu
DEFAULT_VIDEO_SOURCE = "vision/test_video/farm_test.mp4"

# ─────────────────────────────────────────────
# GÖRÜNTÜ İŞLEME
# ─────────────────────────────────────────────

# İşleme çözünürlüğü (YOLOv8'e gönderilen frame boyutu)
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Frame atlama — Raspberry Pi optimizasyonu
# 1 = her kare, 2 = bir atla (15 FPS'i 7-8'e indirir ama CPU'yu yarıya düşürür)
FRAME_SKIP = 2

# ─────────────────────────────────────────────
# HAYVAN SINIFLARI (COCO dataset)
# ─────────────────────────────────────────────
# Sadece inek (cow) — COCO class ID: 19
# Diğer sınıfları eklemek için:
#   14: bird  | 15: cat   | 16: dog   | 17: horse | 18: sheep

ANIMAL_CLASS_IDS = [19]   # Sadece cow

ANIMAL_CLASS_NAMES = {
    19: "İnek",
}

# ─────────────────────────────────────────────
# HAREKET ANALİZİ
# ─────────────────────────────────────────────

# Her track için saklanacak geçmiş pozisyon sayısı (kare)
MOTION_HISTORY_FRAMES = 25

# Normalizasyon sabiti (piksel/kare — bu değerin üzerindeki hareket = 1.0 skor)
# 640x480 çözünürlükte orta hızlı bir hareket ~30-40 piksel/kare
MOTION_NORMALIZE_PIXELS = 40.0

# ─────────────────────────────────────────────
# ANOMALİ TESPİTİ EŞİKLERİ
# ─────────────────────────────────────────────

# Hareket skoru eşikleri [0.0 – 1.0]
MOTION_WARNING_THRESHOLD = 0.35   # Sarı uyarı
MOTION_ALERT_THRESHOLD   = 0.65   # Kırmızı alarm

# Anomali durumu gecikmesi (kaç kare üst üste anomali olursa bildirim çıkar)
ANOMALY_CONFIRM_FRAMES = 5

# ─────────────────────────────────────────────
# GUI AYARLARI
# ─────────────────────────────────────────────

# GUI güncelleme hızı (ms) — 33ms ≈ 30 FPS
GUI_UPDATE_MS = 33

# Pencere boyutu
GUI_WIDTH  = 1200
GUI_HEIGHT = 720
