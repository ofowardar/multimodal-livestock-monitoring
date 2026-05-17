"""
evaluate_model.py
=================
Base YOLOv8s ve Fine-Tuned modeli karşılaştırmalı değerlendirir.
Tez için metrik tablosu üretir.

Kullanım:
    # Baseline (fine-tune öncesi):
    python evaluate_model.py --model yolov8s.pt --tag baseline

    # Fine-tuned (eğitim sonrası):
    python evaluate_model.py --model runs/finetune/cattle_detector/weights/best.pt --tag finetuned
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────
# Ayarlar
# ─────────────────────────────────────────────
DATA_YAML   = "data/cattle/data.yaml"   # Dataset
IMG_SIZE    = 640
RESULTS_DIR = "results/metrics"         # Çıktı klasörü


def evaluate(model_path: str, tag: str):
    print("\n" + "=" * 60)
    print(f"  Model Değerlendirme — {tag.upper()}")
    print("=" * 60)
    print(f"  Model  : {model_path}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Tarih  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60 + "\n")

    # Model yükle
    model = YOLO(model_path)

    # Validation set üzerinde değerlendir
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split="test",       # test seti kullan
        plots=True,         # confusion matrix, PR curve vb. kaydet
        save_json=True,
    )

    # Metrikleri çıkar
    results = {
        "tag": tag,
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "dataset": DATA_YAML,
        # Genel metrikler
        "mAP50":    round(float(metrics.box.map50), 4),   # mAP@0.5
        "mAP50_95": round(float(metrics.box.map),   4),   # mAP@0.5:0.95
        "precision": round(float(metrics.box.mp),   4),   # Mean Precision
        "recall":    round(float(metrics.box.mr),   4),   # Mean Recall
        # Sınıf bazlı metrikler
        "per_class": {}
    }

    # F1 skoru hesapla
    p = results["precision"]
    r = results["recall"]
    results["F1"] = round(2 * p * r / (p + r + 1e-8), 4)

    # Sınıf bazlı metrikler
    class_names = {0: "lie (yatan)", 1: "stand (duran)", 2: "walk (yürüyen)"}
    if hasattr(metrics.box, "ap_class_index"):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            results["per_class"][cls_name] = {
                "AP50": round(float(metrics.box.ap50[i]), 4),
                "AP50_95": round(float(metrics.box.ap[i]), 4),
            }

    # Sonuçları ekrana yazdır
    print("\n" + "─" * 50)
    print("  📊 SONUÇLAR")
    print("─" * 50)
    print(f"  mAP@0.5      : {results['mAP50']:.4f}  ({results['mAP50']*100:.1f}%)")
    print(f"  mAP@0.5:0.95 : {results['mAP50_95']:.4f}  ({results['mAP50_95']*100:.1f}%)")
    print(f"  Precision    : {results['precision']:.4f}  ({results['precision']*100:.1f}%)")
    print(f"  Recall       : {results['recall']:.4f}  ({results['recall']*100:.1f}%)")
    print(f"  F1 Score     : {results['F1']:.4f}  ({results['F1']*100:.1f}%)")

    if results["per_class"]:
        print("\n  Sınıf Bazlı AP@0.5:")
        for cls, vals in results["per_class"].items():
            print(f"    {cls:<20} : {vals['AP50']:.4f}  ({vals['AP50']*100:.1f}%)")

    # JSON olarak kaydet
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = Path(RESULTS_DIR) / f"{tag}_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Metrikler kaydedildi: {out_file}")

    # Eğer her iki metrik de varsa karşılaştır
    compare_if_both_exist()

    return results


def compare_if_both_exist():
    """Her iki JSON da varsa karşılaştırma tablosu yazar."""
    baseline_file  = Path(RESULTS_DIR) / "baseline_metrics.json"
    finetuned_file = Path(RESULTS_DIR) / "finetuned_metrics.json"

    if not (baseline_file.exists() and finetuned_file.exists()):
        return

    with open(baseline_file,  encoding="utf-8") as f:
        base = json.load(f)
    with open(finetuned_file, encoding="utf-8") as f:
        fine = json.load(f)

    print("\n" + "=" * 60)
    print("  📈 KARŞILAŞTIRMA TABLOSU (Tez Materyali)")
    print("=" * 60)
    print(f"  {'Metrik':<20} {'Baseline':>12} {'Fine-Tuned':>12} {'Değişim':>10}")
    print("  " + "─" * 56)

    metrics_to_compare = [
        ("mAP@0.5 (%)",    "mAP50",    100),
        ("mAP@0.5:0.95 (%)", "mAP50_95", 100),
        ("Precision (%)",  "precision", 100),
        ("Recall (%)",     "recall",    100),
        ("F1 Score (%)",   "F1",        100),
    ]

    for label, key, scale in metrics_to_compare:
        b_val = base[key] * scale
        f_val = fine[key] * scale
        delta = f_val - b_val
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {label:<20} {b_val:>11.1f}% {f_val:>11.1f}%  {arrow}{abs(delta):>6.1f}pp")

    print("  " + "─" * 56)

    # Karşılaştırmayı dosyaya da yaz
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "baseline": base,
        "finetuned": fine,
    }
    out_file = Path(RESULTS_DIR) / "comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Karşılaştırma kaydedildi: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Evaluator")
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="Model yolu (örn: yolov8s.pt veya runs/.../best.pt)"
    )
    parser.add_argument(
        "--tag", type=str, choices=["baseline", "finetuned"], default="baseline",
        help="baseline veya finetuned"
    )
    args = parser.parse_args()

    evaluate(args.model, args.tag)
