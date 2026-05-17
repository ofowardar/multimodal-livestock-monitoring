"""
train_finetune.py
YOLOv8s modelini çiftlik sığırı dataseti üzerinde fine-tune eder.

Kullanım:
    python train_finetune.py

Gereksinim:
    - Roboflow'dan indirilen dataset (YOLOv8 formatında)
    - data/cattle.yaml dosyası
    - GPU önerilir (CPU'da çok yavaş)
"""

from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# AYARLAR — İhtiyaca göre değiştir
# ─────────────────────────────────────────────

BASE_MODEL   = "yolov8s.pt"            # Başlangıç ağırlıkları (COCO pre-trained)
DATA_YAML    = "data/cattle/data.yaml"  # download_dataset.py'nin oluşturduğu yol
OUTPUT_DIR   = "runs/finetune"          # Eğitim çıktı dizini
PROJECT_NAME = "cattle_detector"

# Eğitim Hiperparametreleri
EPOCHS      = 50     # Epoch sayısı
IMG_SIZE    = 640    # Görüntü boyutu
BATCH_SIZE  = 8      # GPU varsa 16, CPU'da 4-8 önerilir
PATIENCE    = 15     # Early stopping
LR0         = 0.001  # Fine-tune için düşük öğrenme hızı
FREEZE      = 10     # Backbone'un ilk N katmanını dondur

# ─────────────────────────────────────────────


def train():
    print("=" * 60)
    print("  YOLOv8 Fine-Tuning — Çiftlik Sığırı Tespiti")
    print("=" * 60)

    # Dataset yaml kontrolü
    if not Path(DATA_YAML).exists():
        print(f"\n[HATA] Dataset dosyası bulunamadı: {DATA_YAML}")
        print("Lütfen önce dataset indirme adımlarını tamamlayın.")
        print("Detay için README veya fine-tuning rehberini okuyun.")
        return

    print(f"\n[1/3] Model yükleniyor: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"[2/3] Eğitim başlıyor...")
    print(f"      Dataset : {DATA_YAML}")
    print(f"      Epochs  : {EPOCHS}")
    print(f"      Img Size: {IMG_SIZE}")
    print(f"      Batch   : {BATCH_SIZE}")
    print()

    # GPU varsa otomatik kullan, yoksa CPU'ya geç
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    workers = 4 if torch.cuda.is_available() else 0  # Windows CPU'da workers=0 gerekli
    print(f"      Cihaz   : {'GPU (CUDA)' if device == '0' else 'CPU (yavaş olabilir)'}")

    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        patience    = PATIENCE,
        lr0         = LR0,
        freeze      = FREEZE,       # Backbone'u dondur, sadece head eğit
        project     = OUTPUT_DIR,
        name        = PROJECT_NAME,
        exist_ok    = True,
        verbose     = True,
        device      = device,
        workers     = workers,
        augment     = True,         # Veri artırma (mosaic, flip vb.)
        cos_lr      = True,         # Cosine learning rate schedule
        save        = True,
        save_period = 10,           # Her 10 epoch'ta checkpoint kaydet
    )

    print("\n[3/3] Eğitim tamamlandı!")

    # En iyi modelin yolu
    best_model_path = Path(OUTPUT_DIR) / PROJECT_NAME / "weights" / "best.pt"
    print(f"\n✅ En iyi model: {best_model_path}")
    print(f"\nconf.py içinde şunu güncelle:")
    print(f'    MODEL_PATH = "{best_model_path}"')

    return best_model_path


def validate(model_path: str = None):
    """Fine-tuned modeli test seti üzerinde doğrula."""
    path = model_path or str(
        Path(OUTPUT_DIR) / PROJECT_NAME / "weights" / "best.pt"
    )
    print(f"\n[Validate] Model: {path}")
    model = YOLO(path)
    metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE)
    print(f"  mAP50   : {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall  : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    best = train()
    if best and best.exists():
        validate(str(best))
