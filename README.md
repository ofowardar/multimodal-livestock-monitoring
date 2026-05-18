# 🐄 Multimodal Livestock Monitoring System

> Çiftlik hayvanlarında stres tespiti için görüntü ve ses verilerinin geç füzyon (late fusion) yöntemiyle birleştirildiği çok modlu izleme sistemi.

**Tez Çalışması** | Bilgisayar Mühendisliği | 2026

---

## 📋 Proje Özeti

Bu sistem, çiftlik hayvanlarının (sığır) davranışsal stres belirtilerini tespit etmek amacıyla:
- **Görüntü modülü:** YOLOv8s + ByteTrack ile gerçek zamanlı davranış tespiti (yatan / duran / yürüyen)
- **Ses modülü:** MFCC tabanlı ses anomali tespiti *(geliştirme aşamasında)*
- **Füzyon:** Late fusion ile iki modaliteden elde edilen skorların birleştirilmesi

üzerine inşa edilmiştir. Sistem Raspberry Pi 5 üzerinde çalışacak şekilde optimize edilmiştir.

---

## 🏗️ Sistem Mimarisi

```
Kamera (görüntü)           Mikrofon (ses)
      ↓                          ↓
  YOLOv8s (fine-tuned)      MFCC Özellik Çıkarma
  ByteTrack (takip)         Anomali Skoru
      ↓                          ↓
  motion_score [0-1]        audio_score [0-1]
      ↓                          ↓
      └──────── Late Fusion ──────┘
                    ↓
         fused_score = α·motion + β·audio
                    ↓
          NORMAL / UYARI / ANOMALİ
```

---

## 📊 Model Performansı

| Metrik | Base YOLOv8s | Fine-Tuned (RTX 5060) | Gelişim (Değişim) |
|---|---|---|---|
| **mAP@0.5** | **3.9%** | **97.7%** | **+93.8pp** 🚀 |
| **mAP@0.5:0.95** | 1.7% | 85.8% | +84.1pp |
| **Precision** | 6.0% | 94.0% | +88.0pp |
| **Recall** | 28.5% | 96.6% | +68.0pp |
| **F1 Score** | 9.9% | 95.2% | +85.3pp |

### Sınıf Bazlı Performans (mAP@0.5)

| Sınıf | Base YOLOv8s | Fine-Tuned (RTX 5060) | Gelişim (Değişim) |
|---|---|---|---|
| **stand (duran)** | 8.4% | 97.6% | +89.2pp |
| **lie (yatan)** | 3.1% | 98.4% | +95.3pp 🛌 |
| **walk (yürüyen)** | 0.1% | 97.0% | +96.9pp |

---

## 📁 Proje Yapısı

```
multimodal-livestock-monitoring/
│
├── vision/
│   ├── detector.py          # YOLOv8 + ByteTrack tespiti
│   ├── motion_analyzer.py   # Hareket skoru hesaplama
│   ├── anomaly_detector.py  # Anomali sınıflandırma
│   └── pipeline.py          # Vision pipeline birleştirici
│
├── main_gui.py              # Tkinter GUI (dark theme)
├── train_finetune.py        # YOLOv8 fine-tuning scripti
├── evaluate_model.py        # Baseline / Fine-tuned karşılaştırma
├── download_dataset.py      # Roboflow dataset indirici
├── conf.py                  # Global konfigürasyon
└── requirements.txt         # Bağımlılıklar
```

---

## 🚀 Kurulum

### PC (Windows)
```powershell
git clone https://github.com/ofowardar/multimodal-livestock-monitoring.git
cd multimodal-livestock-monitoring

python -m venv .myenv
.myenv\Scripts\activate
pip install -r requirements.txt
```

### Raspberry Pi 5
```bash
git clone https://github.com/ofowardar/multimodal-livestock-monitoring.git
cd multimodal-livestock-monitoring

python3 -m venv .pienv
source .pienv/bin/activate
pip install ultralytics opencv-python-headless numpy
```

---

## 🎯 Kullanım

### GUI Çalıştırma
```bash
python main_gui.py
```

### Model Fine-Tuning
```bash
# 1. Dataset indir
python download_dataset.py

# 2. Baseline metrikleri al
python evaluate_model.py --model yolov8s.pt --tag baseline

# 3. Fine-tuning yap
python train_finetune.py

# 4. Fine-tuned metrikleri al
python evaluate_model.py --model runs/finetune/cattle_detector/weights/best.pt --tag finetuned
```

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Nesne Tespiti | YOLOv8s (Ultralytics) |
| Takip | ByteTrack |
| Derin Öğrenme | PyTorch 2.11 + CUDA 12.8 |
| Görüntü İşleme | OpenCV |
| GUI | Tkinter |
| Dataset | Roboflow (cow-detection-7yhtu) |
| Edge Deploy | Raspberry Pi 5 |

---

## 📚 Referanslar

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*
- Roboflow Universe: [cow-detection-7yhtu](https://universe.roboflow.com/lambda-t1w0j/cow-detection-7yhtu)

---

*Bu proje bir lisans tezi kapsamında geliştirilmektedir.*
