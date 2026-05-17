"""
main_gui.py
Tkinter GUI — Çiftlik Hayvanı Stres İzleme Sistemi

Layout:
    Sol (%65)  : Canlı video akışı (annotated frame)
    Sağ (%35)  : Analiz paneli (sayım, hareket, anomali, bireysel skorlar)

Video kamera thread'i → İşleme thread'i → GUI thread (after loop)
"""

import tkinter as tk
from tkinter import filedialog
import threading
import queue
import time

import cv2
from PIL import Image, ImageTk

import conf
from vision.pipeline import VisionPipeline
from vision.anomaly_detector import AnomalyStatus


# ─────────────────────────────────────────────────────────────
# Renk Paleti
# ─────────────────────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_CARD    = "#161b22"
BG_PANEL   = "#21262d"
ACCENT     = "#e94560"
TEXT_MAIN  = "#e6edf3"
TEXT_DIM   = "#8b949e"
COLOR_OK   = "#00e676"
COLOR_WARN = "#ffc107"
COLOR_ALERT= "#f44336"
COLOR_BLUE = "#58a6ff"


class FarmMonitorApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Çiftlik Hayvanı Stres İzleme Sistemi")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(900, 580)

        self.pipeline  = VisionPipeline()
        self.cap       = None
        self.running   = False

        self.frame_queue  = queue.Queue(maxsize=4)   # capture → process
        self.result_queue = queue.Queue(maxsize=4)   # process → GUI

        self._fps_counter = 0
        self._fps_time    = time.time()
        self._current_fps = 0.0

        self._build_ui()

    # ──────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────

    def _build_ui(self):
        # ── Başlık Barı ──
        title_bar = tk.Frame(self.root, bg=BG_CARD, height=48)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="🐄  Çiftlik Hayvanı Stres & Refah İzleme Sistemi",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD, fg=TEXT_MAIN,
        ).pack(side=tk.LEFT, padx=16, pady=10)

        self._clock_var = tk.StringVar(value="--:--:--")
        tk.Label(
            title_bar, textvariable=self._clock_var,
            font=("Consolas", 11), bg=BG_CARD, fg=TEXT_DIM,
        ).pack(side=tk.RIGHT, padx=16)

        # ── Ana İçerik ──
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        self._build_video_panel(content)
        self._build_info_panel(content)

    # ── Sol: Video Paneli ──────────────────────
    def _build_video_panel(self, parent):
        left = tk.Frame(parent, bg=BG_CARD, bd=0)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Video canvas — Canvas kullanıyoruz (Label yerine)
        # Sebebi: Label içinde image set edince widget boyutu büyür,
        # bu da her render'da daha büyük image üretir (çözen döngü).
        # Canvas'ta image daima canvas boyutuna sığıştırılır.
        self.video_canvas = tk.Canvas(
            left, bg="#000000",
            highlightthickness=0,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        # Başlangıç yazısı
        self._placeholder_text = self.video_canvas.create_text(
            400, 240,
            text="Video seçmek için aşağıdaki düğmeyi kullanın",
            font=("Segoe UI", 12), fill=TEXT_DIM,
        )

        # Alt kontrol barı
        ctrl = tk.Frame(left, bg=BG_CARD, height=46)
        ctrl.pack(fill=tk.X, padx=6, pady=6)
        ctrl.pack_propagate(False)

        self.btn_open = tk.Button(
            ctrl, text="📂  Video Aç",
            command=self._open_video,
            bg=ACCENT, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=14, pady=6,
            cursor="hand2", activebackground="#c73652",
        )
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_cam = tk.Button(
            ctrl, text="📷  Webcam",
            command=self._open_webcam,
            bg=BG_PANEL, fg=TEXT_MAIN,
            font=("Segoe UI", 10),
            relief=tk.FLAT, padx=14, pady=6,
            cursor="hand2", activebackground="#30363d",
        )
        self.btn_cam.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = tk.Button(
            ctrl, text="⏹  Durdur",
            command=self._stop,
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Segoe UI", 10),
            relief=tk.FLAT, padx=14, pady=6,
            cursor="hand2", state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT)

        self._fps_var = tk.StringVar(value="FPS: --")
        tk.Label(
            ctrl, textvariable=self._fps_var,
            font=("Consolas", 10), bg=BG_CARD, fg=TEXT_DIM,
        ).pack(side=tk.RIGHT, padx=8)

    # ── Sağ: Bilgi Paneli ─────────────────────
    def _build_info_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK, width=310)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # ── Kart 1: Hayvan Sayısı ──
        c1 = self._card(right, "🐾  Tespit Edilen Hayvan")
        self._count_var = tk.StringVar(value="0")
        tk.Label(
            c1, textvariable=self._count_var,
            font=("Segoe UI", 44, "bold"),
            bg=BG_CARD, fg=COLOR_BLUE,
        ).pack(pady=(4, 8))

        # ── Kart 2: Ortalama Hareket Skoru ──
        c2 = self._card(right, "🏃  Ortalama Hareket Skoru")
        self._motion_var = tk.StringVar(value="0.00")
        tk.Label(
            c2, textvariable=self._motion_var,
            font=("Segoe UI", 26, "bold"),
            bg=BG_CARD, fg=COLOR_BLUE,
        ).pack()
        self._motion_canvas = tk.Canvas(
            c2, height=16, bg="#0d1117", highlightthickness=0,
        )
        self._motion_canvas.pack(fill=tk.X, padx=10, pady=(4, 10))

        # ── Kart 3: Anomali Durumu ──
        c3 = self._card(right, "🚨  Anomali Durumu")
        self._anomaly_status_var = tk.StringVar(value="● NORMAL")
        self._anomaly_label = tk.Label(
            c3, textvariable=self._anomaly_status_var,
            font=("Segoe UI", 18, "bold"),
            bg=BG_CARD, fg=COLOR_OK,
        )
        self._anomaly_label.pack(pady=(4, 2))
        self._anomaly_reason_var = tk.StringVar(value="Normal aktivite")
        tk.Label(
            c3, textvariable=self._anomaly_reason_var,
            font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_DIM,
            wraplength=270,
        ).pack(pady=(0, 8))

        # ── Kart 4: Bireysel Takip ──
        c4 = self._card(right, "📋  Bireysel Hareket Skorları", expand=True)
        self._animals_frame = tk.Frame(c4, bg=BG_CARD)
        self._animals_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Alt bilgi
        tk.Label(
            right, text="Çiftlik Hayvanı Stres & Refah | Vision Modülü v1.0",
            font=("Segoe UI", 8), bg=BG_DARK, fg=TEXT_DIM,
        ).pack(side=tk.BOTTOM, pady=(4, 0))

    def _card(self, parent, title: str, expand: bool = False) -> tk.Frame:
        """Renkli başlıklı kart widget'ı oluştur."""
        outer = tk.Frame(parent, bg=BG_PANEL, bd=0)
        if expand:
            outer.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        else:
            outer.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            outer, text=title,
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL, fg=ACCENT, anchor=tk.W,
        ).pack(fill=tk.X, padx=10, pady=(7, 0))

        sep = tk.Frame(outer, bg=ACCENT, height=1)
        sep.pack(fill=tk.X, padx=10, pady=(2, 4))

        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        return inner

    # ──────────────────────────────────────────
    # Video Kontrol
    # ──────────────────────────────────────────

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.ts"),
                ("Tümü", "*.*"),
            ],
        )
        if path:
            self._start_capture(path)

    def _open_webcam(self):
        self._start_capture(0)

    def _start_capture(self, source):
        self._stop()
        self.pipeline.reset()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.video_canvas.delete("all")
            self.video_canvas.create_text(
                self.video_canvas.winfo_width() // 2 or 400,
                self.video_canvas.winfo_height() // 2 or 240,
                text="⚠ Video açılamadı!",
                font=("Segoe UI", 14, "bold"), fill=COLOR_ALERT,
            )
            return

        # Video meta: FPS
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._video_fps = raw_fps if raw_fps and raw_fps > 0 else 25.0
        print(f"[Capture] Video FPS: {self._video_fps:.1f}")

        self.running = True
        self.btn_stop.config(state=tk.NORMAL, fg=TEXT_MAIN)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_cam.config(state=tk.DISABLED)

        self._fps_counter = 0
        self._fps_time    = time.time()
        self._video_fps   = getattr(self, "_video_fps", 25.0)

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()
        self._gui_loop()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop.config(state=tk.DISABLED, fg=TEXT_DIM)
        self.btn_open.config(state=tk.NORMAL)
        self.btn_cam.config(state=tk.NORMAL)

    # ──────────────────────────────────────────
    # Thread Döngüleri
    # ──────────────────────────────────────────

    def _capture_loop(self):
        """Kamera/Video okuma thread'i.

        FPS-tabanlı zamanlama:
        - Video dosyası: gerçek video FPS'ine göre hız sınırla
        - Webcam: doğal hızda oku, queue dolunca frame at
        - İşleme yavaşsa frame düşer ama video süresi bozulmaz
        """
        is_file = isinstance(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), float) and \
                  self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0

        video_fps    = getattr(self, "_video_fps", 25.0)
        # Her FRAME_SKIP orijinal kare için ne kadar beklemeli
        frame_delay  = conf.FRAME_SKIP / video_fps
        next_time    = time.time()

        frame_idx = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            frame_idx += 1
            if frame_idx % conf.FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (conf.FRAME_WIDTH, conf.FRAME_HEIGHT))

            # ── FPS Hız Kontrolü ──
            # next_time'a kadar uyu — böylece video gerçek hızında oynar
            now = time.time()
            wait = next_time - now
            if wait > 0:
                time.sleep(wait)
            next_time += frame_delay
            # Çok geride kaldıysak (uzun inference) zamanlamayı sıfırla
            if time.time() - next_time > 2.0:
                next_time = time.time()

            # Non-blocking put: doluysa bu frame'i at (processing yavaş)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Frame düştü, video hızı korundu

    def _process_loop(self):
        """YOLOv8 + pipeline işleme thread'i.

        running=False olsa bile queue'da kalan frame'leri bitirmeye devam eder.
        """
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break   # Video/durdur bitti ve queue boş — çık
                continue    # Henüz frame gelmedi, bekle

            try:
                annotated, output = self.pipeline.process_frame(frame)
            except Exception as e:
                import traceback
                print(f"[Process] Hata: {e}")
                traceback.print_exc()
                continue

            # FPS hesapla
            self._fps_counter += 1
            now = time.time()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_time    = now

            # Result queue'ya koy — dolunca eski sonucu at, yenisini ekle
            try:
                self.result_queue.put((annotated, output), block=True, timeout=1.0)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()  # Eskiyi çıkar
                except queue.Empty:
                    pass
                self.result_queue.put_nowait((annotated, output))

    # ──────────────────────────────────────────
    # GUI Güncelleme
    # ──────────────────────────────────────────

    def _gui_loop(self):
        """Ana GUI güncelleme döngüsü (Tkinter ana thread'inde çalışır).

        Video bitince process_loop queue'yu boşaltmaya devam eder.
        Bu yüzden frame_queue da boşalana kadar bekliyoruz.
        """
        all_done = (
            not self.running
            and self.frame_queue.empty()
            and self.result_queue.empty()
        )
        if all_done:
            self.btn_stop.config(state=tk.DISABLED, fg=TEXT_DIM)
            self.btn_open.config(state=tk.NORMAL)
            self.btn_cam.config(state=tk.NORMAL)
            return

        try:
            annotated, output = self.result_queue.get_nowait()
            self._render_frame(annotated)
            self._update_panel(output)
        except queue.Empty:
            pass
        except Exception as e:
            # Herhangi bir render hatası loop'u öldürmesin
            import traceback
            print(f"[GUI] Hata: {e}")
            traceback.print_exc()

        self.root.after(conf.GUI_UPDATE_MS, self._gui_loop)

    def _clock_loop(self):
        """Saat etiketini her saniye günceller — video dursa da çalışır."""
        self._clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._clock_loop)

    def _render_frame(self, frame):
        """OpenCV frame'ini Canvas'a yerleştir.

        Canvas boyutunu layout belirler — image hiçbir zaman canvas'tan
        büyük olmaz, dolayısıyla büyüme döngüsü oluşmaz.
        En-boy oranı korunur (pillarbox/letterbox).
        """
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return  # Widget henüz boyutlanmadı

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # En-boy oranını koruyarak canvas'a sığdır
        iw, ih = img.width, img.height
        scale  = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img    = img.resize((nw, nh), Image.Resampling.BILINEAR)

        photo = ImageTk.PhotoImage(img)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            cw // 2, ch // 2, image=photo, anchor=tk.CENTER
        )
        self.video_canvas._photo = photo  # GC'den koruma

    def _update_panel(self, output: dict):
        """Sağ bilgi panelini güncelle."""

        # FPS
        self._fps_var.set(f"FPS: {self._current_fps:.1f}")

        # Hayvan sayısı
        self._count_var.set(str(output["animal_count"]))

        # Ortalama hareket skoru
        avg = output["avg_motion_score"]
        self._motion_var.set(f"{avg:.2f}")
        self._draw_motion_bar(avg)

        # Anomali durumu
        status: AnomalyStatus = output["anomaly_status"]
        color_map = {
            AnomalyStatus.NORMAL:  COLOR_OK,
            AnomalyStatus.WARNING: COLOR_WARN,
            AnomalyStatus.ALERT:   COLOR_ALERT,
        }
        icon_map = {
            AnomalyStatus.NORMAL:  "●  NORMAL",
            AnomalyStatus.WARNING: "⚠  UYARI",
            AnomalyStatus.ALERT:   "🚨 ANOMALİ!",
        }
        clr = color_map[status]
        self._anomaly_status_var.set(icon_map[status])
        self._anomaly_label.config(fg=clr)
        self._anomaly_reason_var.set(output["anomaly_reason"])

        # Bireysel hayvan listesi
        for w in self._animals_frame.winfo_children():
            w.destroy()

        for det in output["detections"]:
            score = output["motion_scores"].get(det.track_id, 0.0)
            ind_color = (
                COLOR_OK   if score < 0.35 else
                COLOR_WARN if score < 0.65 else
                COLOR_ALERT
            )
            row = tk.Frame(self._animals_frame, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1, padx=2)

            tk.Label(
                row,
                text=f"ID:{det.track_id:>3}  {det.class_name:<10}",
                font=("Consolas", 9), bg=BG_PANEL, fg=TEXT_MAIN,
                anchor=tk.W, width=20,
            ).pack(side=tk.LEFT, padx=(6, 0))

            # Küçük bar
            bar_canvas = tk.Canvas(row, height=12, width=70, bg="#0d1117", highlightthickness=0)
            bar_canvas.pack(side=tk.LEFT, padx=4)
            filled = int(70 * score)
            bar_canvas.create_rectangle(0, 0, 70, 12, fill="#1c2128", outline="")
            if filled > 0:
                bar_canvas.create_rectangle(0, 0, filled, 12, fill=ind_color, outline="")

            tk.Label(
                row, text=f"{score:.2f}",
                font=("Consolas", 9, "bold"),
                bg=BG_PANEL, fg=ind_color, width=5,
            ).pack(side=tk.RIGHT, padx=(0, 6))

    def _draw_motion_bar(self, score: float):
        """Ortalama hareket skoru için progress bar çiz."""
        self._motion_canvas.update_idletasks()
        w = self._motion_canvas.winfo_width()
        h = self._motion_canvas.winfo_height() or 16

        self._motion_canvas.delete("all")
        self._motion_canvas.create_rectangle(0, 0, w, h, fill="#1c2128", outline="")

        if score <= 0:
            return

        color = (
            COLOR_OK   if score < 0.35 else
            COLOR_WARN if score < 0.65 else
            COLOR_ALERT
        )
        filled = int(w * min(score, 1.0))
        self._motion_canvas.create_rectangle(0, 0, filled, h, fill=color, outline="")


# ─────────────────────────────────────────────────────────────
# Giriş Noktası
# ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Çiftlik Hayvanı Stres İzleme Sistemi")
    root.geometry(f"{conf.GUI_WIDTH}x{conf.GUI_HEIGHT}")
    root.configure(bg=BG_DARK)

    # Tkinter varsayılan temasını iyileştir
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    app = FarmMonitorApp(root)
    app._clock_loop()  # Saati hemen başlat (video beklemesin)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
