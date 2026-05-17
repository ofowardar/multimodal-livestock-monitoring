"""
download_dataset.py
Roboflow'dan cattle detection dataseti indirir.

Kullanım:
    1. API_KEY değişkenine kendi API key'ini yaz
    2. python download_dataset.py

Dataset indirilince: data/cattle/ klasörüne kaydedilir.
"""

from roboflow import Roboflow

# ──────────────────────────────────────────
# BURAYA KOYACAKSIN:
# roboflow.com → Sağ üst profil → Settings → API Key
# ──────────────────────────────────────────
API_KEY = "9VK8EUkQ8PNdckI6cQkR"

# ──────────────────────────────────────────
# Dataset bilgileri
# Kaynak: https://universe.roboflow.com/lambda-t1w0j/cow-detection-7yhtu
# ──────────────────────────────────────────
WORKSPACE  = "lambda-t1w0j"
PROJECT    = "cow-detection-7yhtu"
VERSION    = 1

def main():
    print("=" * 50)
    print("Roboflow Dataset İndirici")
    print("=" * 50)

    if API_KEY == "BURAYA_KENDI_API_KEY'INI_YAZ":
        print("\n❌ HATA: API_KEY boş!")
        print("   download_dataset.py dosyasını aç ve")
        print("   API_KEY = 'xxx...' satırını doldur.")
        return

    print(f"\n🔗 Roboflow'a bağlanılıyor...")
    rf = Roboflow(api_key=API_KEY)

    print(f"📂 Proje yükleniyor: {WORKSPACE}/{PROJECT}")
    project = rf.workspace(WORKSPACE).project(PROJECT)

    print(f"⬇️  Versiyon {VERSION} indiriliyor (YOLOv8 formatı)...")
    dataset = project.version(VERSION).download(
        model_format="yolov8",
        location="data/cattle",
        overwrite=True
    )

    print(f"\n✅ Dataset başarıyla indirildi!")
    print(f"   Konum : data/cattle/")
    print(f"   YAML  : data/cattle/data.yaml")
    print(f"\nŞimdi eğitimi başlatabilirsin:")
    print(f"   .myenv\\Scripts\\python.exe train_finetune.py")


if __name__ == "__main__":
    main()
