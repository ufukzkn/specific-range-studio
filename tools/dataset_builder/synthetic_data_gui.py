import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
import os
import json
import re
import random
import shutil
from pathlib import Path
import queue
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

class ToolTip(object):
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.delay, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        # Ikon uzerinde daha iyi hizalama icin koordinat hesaplama
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 10

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))

        # Temaya uygun renkler
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", foreground="#000000",
                      relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "9", "normal"),
                      padx=5, pady=3)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    ToolTip(widget, text)

class MaskEditor(tk.Toplevel):
    def __init__(self, parent, img_path, mask_path):
        super().__init__(parent)
        self.title("Maske Duzenleyici")
        self.geometry("1000x800")

        self.img_path = img_path
        self.mask_path = mask_path

        # Goruntuleri Yukle
        self.pil_img = Image.open(img_path).convert("RGB")
        self.pil_mask = Image.open(mask_path).convert("L")

        # Resize for display if too big
        self.display_size = (800, 800)
        self.scale = 1.0

        # Maskeyi duzenlemek icin bir kopya (Draw nesnesi)
        self.draw_mask = self.pil_mask.copy()
        self.draw = ImageDraw.Draw(self.draw_mask)

        # UI
        self._create_toolbar()
        self._create_canvas()

        self.brush_size = 5
        self.brush_color = 255 # 255=White (Add), 0=Black (Remove)

        self.update_display()

    def _create_toolbar(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(toolbar, text="Firca Boyutu:").pack(side=tk.LEFT)
        self.scale_size = tk.Scale(toolbar, from_=1, to=50, orient=tk.HORIZONTAL)
        self.scale_size.set(5)
        self.scale_size.pack(side=tk.LEFT, padx=5)

        self.var_mode = tk.StringVar(value="add")
        ttk.Radiobutton(toolbar, text="Ekle (Beyaz)", variable=self.var_mode, value="add", command=self._set_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(toolbar, text="Sil (Siyah)", variable=self.var_mode, value="remove", command=self._set_mode).pack(side=tk.LEFT, padx=5)

        ttk.Button(toolbar, text="Kaydet", command=self.save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="Iptal", command=self.destroy).pack(side=tk.RIGHT)

    def _create_canvas(self):
        self.canvas = tk.Canvas(self, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

    def _set_mode(self):
        if self.var_mode.get() == "add":
            self.brush_color = 255
        else:
            self.brush_color = 0

    def update_display(self):
        # Orijinal resim ve maskeyi karistirip goster
        # Maskeyi kirmizimsi goster
        mask_rgb = self.draw_mask.convert("RGB")

        # Overlay
        # 1. Image -> Numpy
        img_np = np.array(self.pil_img)
        mask_np = np.array(self.draw_mask)

        # Maskenin oldugu yerleri kirmizi yap
        # Basit overlay:
        overlay = img_np.copy()
        overlay[mask_np > 127] = [0, 0, 255] # Mavi (OpenCV BGR degil, PIL RGB oldugu icin Mavi=Blue.. pardon RGB burasi. 0,0,255 mavidir.)
        # Kirmizi yapmak istersek [255, 0, 0]

        # Alpha blend (manuel)
        alpha = 0.4
        blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(blended))

        self.canvas.delete("all")
        # Center image
        cw = self.winfo_width()
        ch = self.winfo_height()
        # Canvas ortasina koy
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def paint(self, event):
        x, y = event.x, event.y
        r = self.scale_size.get()

        # PIL uzerinde ciz
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=self.brush_color, outline=None)

        # Canvas uzerinde gorsel geri bildirim (Hizli olmasi icin sadece daire ciz, tum resmi update etme)
        color = "red" if self.brush_color == 255 else "blue" # Ekleme kirmizi, Silme mavi (gecici vizualizasyon)
        # Ama gercek overlay ile uyumlu olsun diye:
        # Ekle = Kirmizi maske, Sil = Orijinal resim (bunu yapmak zor)
        # Basitce uzerine renkli daire cizelim, mouse birakinca update_display yapariz.
        # Veya surekli update_display? (Yavas olabilir)

        # Basit cizim
        outline_color = "red" if self.brush_color == 255 else "black"
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=outline_color, outline=outline_color)

    def save(self):
        try:
            self.draw_mask.save(self.mask_path)
            messagebox.showinfo("Basarili", "Maske kaydedildi.")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatasi: {e}")


class SyntheticDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentetik Veri ve Egri Analiz Kontrol Paneli")
        self.root.geometry("1200x850")

        # Icon
        icon_path = "logo.ico"
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Icon yuklenemedi: {e}")

        # Stil ayarlari
        style = ttk.Style()
        style.theme_use('clam')

        # --- DEGISKENLER ---
        # 1. Sentetik Veri
        self.num_images_var = tk.StringVar(value="5000")
        self.num_workers_var = tk.StringVar(value="6")

        # Augmentation (Gelismis)
        self.aug_rotation_var = tk.StringVar(value="3.0")
        self.aug_noise_var = tk.StringVar(value="15.0")
        self.aug_jpeg_var = tk.StringVar(value="70")
        self.aug_blur_var = tk.StringVar(value="0")       # Yeni
        self.aug_shadow_var = tk.StringVar(value="0")
        self.aug_flip_var = tk.StringVar(value="0")       # 0: off, 1:H, 2:V, 3:Both
        self.aug_shear_var = tk.StringVar(value="0.0")
        self.aug_hue_var = tk.StringVar(value="0")
        self.aug_sat_var = tk.StringVar(value="0")
        self.aug_cutout_var = tk.StringVar(value="0")
        self.aug_motion_var = tk.StringVar(value="0")

        # Colab Entegrasyonu
        self.chk_colab_train_var = tk.BooleanVar(value=False)
        self.colab_link_var = tk.StringVar(value="https://colab.research.google.com/")
        self.aug_prob_var = tk.StringVar(value="0.7")

        # 2. Egitim
        self.epochs_var = tk.StringVar(value="50")
        self.batch_size_var = tk.StringVar(value="8")
        self.dataset_path_var = tk.StringVar(value="dataset_production")
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.image_size_var = tk.StringVar(value="256")
        self.early_stopping_patience_var = tk.StringVar(value="10")
        self.early_stopping_mode_var = tk.StringVar(value="Otomatik")

        # 3. Inference
        self.model_path_var = tk.StringVar(value="checkpoints/best_model.pth")
        self.input_dir_var = tk.StringVar(value="test_images/")
        self.pdf_path_var = tk.StringVar()
        self.pdf_pages_var = tk.StringVar(value="all") # "1-10" or "all"
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.output_dir_var = tk.StringVar(value="segmentation_results/")
        self.correction_dir_var = tk.StringVar(value="segmentation_results/")
        self.inference_mode_var = tk.StringVar(value="folder") # "folder" or "pdf"

        # Checkboxlar
        self.chk_gen_data_var = tk.BooleanVar(value=True)
        self.chk_train_model_var = tk.BooleanVar(value=True)
        self.chk_inference_var = tk.BooleanVar(value=True)
        self.chk_clean_output_var = tk.BooleanVar(value=True) # Varsayilan temizle
        self.chk_extract_data_var = tk.BooleanVar(value=True)
        self.chk_error_correction_var = tk.BooleanVar(value=True)

        # New Feature: Clean Dataset
        self.chk_clean_data_var = tk.BooleanVar(value=False)

        self.is_running = False
        self.process = None
        self.log_queue = queue.Queue()
        self.plot_queue = queue.Queue()

        # Preview State
        self.current_preview_img = None
        self.current_preview_mask = None # Mask path

        # Sequential Preview
        self.preview_files = [] # List of image paths
        self.preview_index = 0

        # Plot Data
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.epochs = []

        self._create_widgets()
        self._create_menu()
        self._check_queue()
        self._check_plot_queue()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Theme Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Gorunum", menu=view_menu)
        view_menu.add_command(label="Acik Tema", command=lambda: self.toggle_theme("light"))
        view_menu.add_command(label="Koyu Tema", command=lambda: self.toggle_theme("dark"))

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Yardim", menu=help_menu)
        help_menu.add_command(label="Parametre Rehberi", command=self.show_help_window)
        help_menu.add_separator()
        help_menu.add_command(label="Hakkinda", command=self.show_about_window)

    def _on_help_close(self):
        """Yardim penceresi kapatildiginda referansi temizle"""
        try:
            self._help_win.destroy()
        except Exception:
            pass
        self._help_win = None

    def show_help_window(self):
        # Zaten aciksa one cikar, yeni pencere acma
        if hasattr(self, '_help_win') and self._help_win is not None:
            try:
                self._help_win.lift()
                self._help_win.focus_force()
                return
            except tk.TclError:
                pass  # Pencere kapanmis, yeni ac

        help_win = tk.Toplevel(self.root)
        self._help_win = help_win
        help_win.protocol("WM_DELETE_WINDOW", self._on_help_close)
        help_win.title("Parametre Rehberi")
        help_win.geometry("800x700")

        # Temaya gore renkleri belirle
        is_dark = getattr(self, 'current_theme', 'light') == 'dark'
        win_bg   = "#1e1e1e" if is_dark else "#f5f5f5"
        txt_bg   = "#1e1e1e" if is_dark else "#ffffff"
        txt_fg   = "#d4d4d4" if is_dark else "#000000"

        help_win.configure(bg=win_bg)

        txt = scrolledtext.ScrolledText(
            help_win, wrap=tk.WORD, padx=16, pady=12,
            font=("Consolas", 10), bg=txt_bg, fg=txt_fg,
            insertbackground=txt_fg, relief="flat", borderwidth=0
        )
        txt.pack(fill=tk.BOTH, expand=True)

        # ── Renk etiketleri: koyu temada parlak VS Code renkleri, acik temada yuksek kontrast ──
        if is_dark:
            txt.tag_config("banner",   foreground="#00d4ff", font=("Consolas", 12, "bold"))
            txt.tag_config("subtitle", foreground="#a0a0a0", font=("Consolas", 10, "italic"))
            txt.tag_config("section",  foreground="#ffd700", font=("Consolas", 11, "bold"))
            txt.tag_config("subsec",   foreground="#4ec9b0", font=("Consolas", 10, "bold"))
            txt.tag_config("key",      foreground="#9cdcfe", font=("Consolas", 10))
            txt.tag_config("val",      foreground="#f3deb7", font=("Consolas", 10))  # daha parlak portakal-sari
            txt.tag_config("bullet",   foreground="#ffffff", font=("Consolas", 10))  # tam beyaz
            txt.tag_config("check",    foreground="#6a9955", font=("Consolas", 10, "bold"))
            txt.tag_config("new",      foreground="#f44747", font=("Consolas",  9, "bold"))
            txt.tag_config("sep",      foreground="#5c5c5c", font=("Consolas",  9))  # daha belirgin gri
        else:
            txt.tag_config("banner",   foreground="#003366", font=("Consolas", 12, "bold")) # Koyu Lacivert
            txt.tag_config("subtitle", foreground="#111111", font=("Consolas", 10, "italic", "bold")) # Neredeyse tam siyah
            txt.tag_config("section",  foreground="#800000", font=("Consolas", 11, "bold")) # Bordo
            txt.tag_config("subsec",   foreground="#004d00", font=("Consolas", 10, "bold")) # Koyu Yesil
            txt.tag_config("key",      foreground="#000080", font=("Consolas", 10, "bold")) # Koyu Mavi
            txt.tag_config("val",      foreground="#000000", font=("Consolas", 10))         # Tam Siyah (Normal Metin)
            txt.tag_config("bullet",   foreground="#000000", font=("Consolas", 10, "bold")) # Tam Siyah Kalin
            txt.tag_config("check",    foreground="#006600", font=("Consolas", 10, "bold")) # Canli Koyu Yesil
            txt.tag_config("new",      foreground="#cc0000", font=("Consolas",  9, "bold")) # Kirmizi
            txt.tag_config("sep",      foreground="#404040", font=("Consolas",  9, "bold")) # Koyu Gri Koyu Cizgi
        txt.tag_config("nl",       font=("Consolas", 4))   # kucuk bosluk

        def w(text, tag="bullet"):
            txt.insert(tk.END, text, tag)
        def nl(n=1):
            txt.insert(tk.END, "\n" * n)
        def sep():
            w("  " + "─" * 70 + "\n", "sep")

        # ── BANNER ─────────────────────────────────────────────────────────
        nl()
        w("  " + "="*60 + "\n", "sep")
        w("  SENTETIK VERI URETIMI VE DERIN OGRENME PLATFORMU\n", "banner")
        w("  " + "="*60 + "\n", "sep")
        nl()

        # ── [1] PROGRAMIN AMACI VE KULLANIM SENARYOSU ──
        w("  [1] PROGRAMIN AMACI VE KULLANIM SENARYOSU\n", "section")
        sep()
        w("  Bu yazilim, sentetik ucak grafikleri olusturup, U-Net modeli kullanarak\n  grafiklerdeki egrileri (curve) otomatik olarak segmente etmek ve bu egrileri\n  sayisallastirarak Excel formatinda cikti almak icin gelistirilmis tam bir sistemdir.\n", "bullet")
        nl()
        w("  Nasil Calisir?\n", "subsec")

        steps = [
            ("1. Adim (Veri Uretimi)", "Modeli egitmek icin 5000+ sentetik sahte grafik uretir. Gercek dunya sartlari (bulaniklik, gurultu, golge) simule edilerek COCO formatinda ayrilir."),
            ("2. Adim (U-Net Egitimi)", "Uretilen verilerle U-Net modeli (Semantic Segmentation) egitilir. Model egrileri test, val ve train olarak ayirarak en optimum sekilde arka plandan ayirmayi ogrenir."),
            ("3. Adim (PDF/PNG Inference)", "Egitilmis model, gercek PDF dosyalarindaki veya resimlerdeki grafikleri otomatik keser, OCR ile motor/agirlik bilgilerini okur ve egrileri bulur."),
            ("4. Adim (Excel'e Cikarim)", "Bulunan egriler dijitallestirilerek gercek degerleri (X, Y) ile beraber Excel formatinda disa aktarilir."),
            ("5. Adim (Hata Duzeltme)", "Excel'deki veriler uzerindeki havacilik duzeltmeleri (ISA, Mach, Weight) geri alinarak ham baseline verisi elde edilir.")
        ]

        for name, desc in steps:
            w(f"  ▸ ", "sep")
            w(f"{name}\n", "key")
            w(f"    {desc}\n", "val")
            nl()

        # ── [2] ARAYUZ VE PARAMETRELERIN DETAYLI ACIKLAMASI ──
        w("  [2] ARAYUZ VE PARAMETRELERIN DETAYLI ACIKLAMASI\n", "section")
        sep()
        nl()

        panels = [
            ("BOLUM 1: SENTETIK VERI URETIMI (Sol Panel)", [
                ("Aktif Et", "Bu adimi calistirmak istiyorsaniz isaretleyin."),
                ("Uretim Oncesi SIL", "Disk tasarrufu ve temiz baslangic icin eski verileri siler."),
                ("Grafik Sayisi", "Modelin tatmin edici sekilde ogrenmesi icin onerilen uretim miktari (Orn: 5000)"),
                ("Worker Sayisi", "Coklu islem icin CPU cekirdek sayisi. (Ne kadar yuksekse, o kadar hizli)"),
            ]),
            ("Gelismis Veri Artirma (Augmentation)", [
                ("Olasilik", "Belirtilen bozulma efektlerinin uygulanma ihtimali (0.0 hic, 1.0 kesinlikle)"),
                ("Cevirme (Flip)", "Goruntuyu yatay veya dikey cevirir (0:Kapali, 1:Yatay, 2:Dikey, 3:Her ikisi)"),
                ("Kaykilma (Shear)", "Goruntuye perspektif egriligi katar (Orn: 0.1)"),
                ("Hue / Sat", "Renk tonu ve doygunlugunu degistirir (Orn: Hue:10, Sat:20)"),
                ("Cutout", "Goruntuye rastgele siyah engeller koyar (Orn: 3 adet kapatma)"),
                ("Motion Blur", "Hareket kaynakli bulaniklik siddeti (Orn: 5)"),
            ]),
            ("BOLUM 2: MODEL EGITIMI (Sol Panel)", [
                ("Dataset Yolu", "Uretilen ve JSON ile paketlenen egitim verisi klasoru."),
                ("Model Kapasitesi", "U-Net feature sayisi (64 veya 128 olabilir)."),
                ("Epoch Sayisi", "Modelin egitim tur sayisi. (Oneri: 50-100 arasi)."),
                ("Batch Size", "Ekran karti VRAM kapasitesine gore. Hata alirsaniz dusurun (Orn: 4-8-16)."),
                ("Learning Rate", "Egitimin adim araligi. Standart: 1e-3 (0.001)"),
                ("Image Size", "Goruntu standardizasyon boyutu. Standart 256'dir."),
                ("Erken Durdurma", "Eger model belirli sayida(10-15) epoch boyunca daha iyi sonuc veremezse, asiri ogrenmeyi engellemek icin kendini durdurur."),
            ]),
            ("BOLUM 3: EGRI CIKARIMI (Sol Panel)", [
                ("Model (.pth)", "Egitimi tamamlanmis agirlik dosyasi (checkpoints/best_model.pth)."),
                ("Giris (PDF/Klasor)", "Islenecek PDF Raporu veya resim dolu klasor."),
                ("Sayfalar", "Butun PDF'i islemek yerine '10-50' seklinde sayfa filtresi verebilirsiniz."),
                ("Esik Degeri", "Egri tespit eminlik yuzdesi (Threshold). Varsayilan 0.5'tir. Ince cizgiler icin 0.3'e cekilebilir."),
            ]),
            ("BOLUM 4: EXCEL AKTARIMI (Sol Panel)", [
                ("Aktif Et", "Tespit edilen egrileri dijitallestirip Excel dosyasina (.xlsx) aktarir."),
            ]),
            ("BOLUM 5: HATA DUZELTME (Sol Panel)", [
                ("Aktif Et", "Excel'deki verilerin fiziksel duzeltmelerini (ISA, Mach, Weight) geri alarak 'curve_data_HAM_Baseline.xlsx' olusturur."),
            ]),
            ("SAG PANEL OZELLIKLERI VE ARACLAR", [
                ("Ana Butonlar", "Secili tum modulleri arka arkaya otonom sekilde baslatabilir veya aniden durdurabilirsiniz."),
                ("Islem Loglari", "Kodlarin ciktilarini, OCR hatalarini, basarili islemleri okuyabilirsiniz."),
                ("Canli Egitim Testi", "Egitim surecini IoU(Basarim) ve Loss(Kayip) olarak bir tablo grafiginde isler."),
                ("Maske Editoru", "Cikan kotu maskelere firca moduyla gorsel onizlemeler kismindan ufak dokunuslar atabilirsiniz."),
                ("DIS ARACLAR", "Otomatik kullanim icin 'Poppler' ve 'Tesseract OCR' kurulu olup sistem PATH'inde tanimli olmalidir."),
            ]),
        ]

        for title, items in panels:
            w(f"  ■  {title}\n", "subsec")
            for key, val in items:
                w(f"     • ", "sep")
                w(f"{key}", "key")
                w(f" :  ", "sep")
                w(f"{val}\n", "val")
            nl()

        sep()
        nl()
        txt.configure(state='disabled')

    def _create_widgets(self):
        # --- Ana PanedWindow (Ayarlanabilir Ayirici) ---
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # --- Sol Panel (Ayarlar) - Scrollable Yapi ---
        left_container = ttk.Frame(self.main_paned, padding="0")
        self.main_paned.add(left_container, weight=1) # Baslangicta esit pay alsin

        # Canvas ve Scrollbar Pakiplemesi (Ust Kisim)
        canvas_frame = ttk.Frame(left_container)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas ve Scrollbar
        canvas = tk.Canvas(canvas_frame, width=380) # Genislik sabit
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self.left_scrollbar = scrollbar  # Tema degisiminde erisim icin sakla

        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Sol Panel Alt Kisim (Progress Bar) ---
        status_frame = ttk.Frame(left_container, padding="5", relief="sunken")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = ttk.Label(status_frame, text="Hazir", font=("Arial", 9, "bold"))
        self.lbl_status.pack(anchor=tk.W)

        self.progress_var = tk.DoubleVar()
        self.progressbar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill=tk.X, pady=2)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        content_frame = ttk.Frame(self.scrollable_frame, padding="5")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 1. SENTETIK VERI URETIMI GRUBU
        self.gen_frame = ttk.LabelFrame(content_frame, text="1. Sentetik Veri Uretimi", padding="5")
        self.gen_frame.pack(fill=tk.X, pady=2)

        chk_gen_frame = ttk.Frame(self.gen_frame)
        chk_gen_frame.pack(anchor=tk.W)
        chk_gen = ttk.Checkbutton(chk_gen_frame, text="Aktif Et", variable=self.chk_gen_data_var, command=self._update_ui_state)
        chk_gen.pack(side=tk.LEFT)
        self._add_info_icon(chk_gen_frame, "Bu modulu (Veri Uretimi) aktif veya pasif hale getirir.")

        self.gen_params_frame = ttk.Frame(self.gen_frame)
        self.gen_params_frame.pack(fill=tk.X)

        # Clean Dataset Checkbox
        chk_clean_frame = ttk.Frame(self.gen_params_frame)
        chk_clean_frame.pack(anchor=tk.W, padx=20)
        chk_clean = ttk.Checkbutton(chk_clean_frame, text="Uretim Oncesi Eski Verileri SIL", variable=self.chk_clean_data_var)
        chk_clean.pack(side=tk.LEFT)
        self._add_info_icon(chk_clean_frame, "DIKKAT: Isaretlenirse 'dataset_production' klasorundeki tum eski veriler kalici olarak silinir!")

        self._add_entry(self.gen_params_frame, "Grafik Sayisi:", self.num_images_var, help_text="Uretilecek toplam sentetik goruntu sayisi (orn: 5000).")
        self._add_entry(self.gen_params_frame, "Worker Sayisi:", self.num_workers_var, help_text="Paralel islemci sayisi. CPU cekirdek sayisina gore ayarlayin.")

        # Augmentation Alt Grubu
        aug_frame = ttk.LabelFrame(self.gen_params_frame, text="Gelismis Veri Artirma", padding="5")
        aug_frame.pack(fill=tk.X, pady=5)

        self._add_entry(aug_frame, "Dondurme (Max °):", self.aug_rotation_var, help_text="Rastgele dondurme acisi (+/- derece).")
        self._add_entry(aug_frame, "Gurultu (Seviye):", self.aug_noise_var, help_text="Goruntuye eklenen karincalanma/gurultu miktari.")
        self._add_entry(aug_frame, "JPEG Kalite (Min):", self.aug_jpeg_var, help_text="Jpeg sikistirma kalitesi (10-100). Dusuk degerler daha fazla bozulma yapar.")
        self._add_entry(aug_frame, "Bulaniklik (Max Kernel):", self.aug_blur_var, help_text="0 Kapali. Aciklik secenekleri: 3, 5, 7 vb.")
        self._add_entry(aug_frame, "Golge Efekti (0/1):", self.aug_shadow_var, help_text="1 Acik, 0 Kapali. Dengesiz aydinlatma hissi verir.")
        self._add_entry(aug_frame, "Olasilik (0-1):", self.aug_prob_var, help_text="Bu sayfadaki efektlerin (bulaniklik, golge vb.) uygulanma ihtimali.")

        # New Augmentations
        self._add_entry(aug_frame, "Cevirme (0-3):", self.aug_flip_var, help_text="0:Kapali, 1:Yatay, 2:Dikey, 3:Her Ikisi")
        self._add_entry(aug_frame, "Kaykilma (Shear):", self.aug_shear_var, help_text="Goruntuyu yamultma miktari (orn: 0.1)")
        self._add_entry(aug_frame, "Hue (Renk):", self.aug_hue_var, help_text="Renk tonu degisimi (0-180)")
        self._add_entry(aug_frame, "Doygunluk (Sat):", self.aug_sat_var, help_text="Renk doygunlugu degisimi (0-255)")
        self._add_entry(aug_frame, "Cutout (Delik):", self.aug_cutout_var, help_text="Rastgele siyah kutu sayisi (orn: 3)")
        self._add_entry(aug_frame, "Motion Blur:", self.aug_motion_var, help_text="Hareket bulanikligi siddeti (orn: 5)")

        # 2. MODEL EGITIMI GRUBU
        self.train_frame = ttk.LabelFrame(content_frame, text="2. Model Egitimi", padding="5")
        self.train_frame.pack(fill=tk.X, pady=2)

        chk_train_frame = ttk.Frame(self.train_frame)
        chk_train_frame.pack(anchor=tk.W)
        chk_train = ttk.Checkbutton(chk_train_frame, text="Aktif Et", variable=self.chk_train_model_var, command=self._update_ui_state)
        chk_train.pack(side=tk.LEFT)
        self._add_info_icon(chk_train_frame, "Bu modulu (Model Egitimi) aktif veya pasif hale getirir.")

        self.train_params_frame = ttk.Frame(self.train_frame)
        self.train_params_frame.pack(fill=tk.X)

        # Colab Option
        colab_frame = ttk.Frame(self.train_params_frame)
        colab_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(colab_frame, text="Google Colab uzerinden Egit", variable=self.chk_colab_train_var, command=self._update_ui_state).pack(anchor=tk.W)

        colab_link_frame = ttk.Frame(colab_frame)
        colab_link_frame.pack(anchor=tk.W, padx=20)
        self.lbl_colab = ttk.Label(colab_link_frame, text="Notebook Linki:")
        self.lbl_colab.pack(side=tk.LEFT)
        self._add_info_icon(colab_link_frame, "Acilacak Colab Notebook adresi. Bos birakirsaniz ana sayfayi acar.")

        self.ent_colab = ttk.Entry(colab_frame, textvariable=self.colab_link_var)
        self.ent_colab.pack(fill=tk.X, padx=20)

        self.local_train_options = ttk.Frame(self.train_params_frame)
        self.local_train_options.pack(fill=tk.X, pady=2)

        self._add_entry(self.local_train_options, "Dataset Yolu (Drive/Local):", self.dataset_path_var, help_text="Egitim verisinin yolu. Colab kullaniyorsaniz burasi onemsizdir.")
        self._add_entry(self.local_train_options, "Epoch Sayisi:", self.epochs_var, help_text="Modelin tum veri setini kac kez gozden gecirecegi.")
        self._add_entry(self.local_train_options, "Batch Size:", self.batch_size_var, help_text="Ayni anda islenen resim sayisi. GPU bellegine gore ayarlayin.")
        self._add_entry(self.local_train_options, "Learning Rate:", self.learning_rate_var, help_text="Modelin ogrenme hizi (Adim boyutu). Standart: 0.001")
        self._add_entry(self.local_train_options, "Image Size:", self.image_size_var, help_text="Resimlerin egitimden once yeniden boyutlandirilacagi kare boyut.")

        es_frame = ttk.Frame(self.local_train_options)
        es_frame.pack(fill=tk.X, pady=1)
        patience_label_frame = ttk.Frame(es_frame)
        patience_label_frame.pack(side=tk.LEFT)
        ttk.Label(patience_label_frame, text="Early Stopping:", width=25).pack(side=tk.LEFT)
        self._add_info_icon(patience_label_frame, "Eger model gelismiyorsa egitimi otomatik durdurur.")

        ttk.Radiobutton(es_frame, text="Oto", variable=self.early_stopping_mode_var, value="Otomatik", command=self._update_ui_state).pack(side=tk.LEFT)
        ttk.Radiobutton(es_frame, text="Manuel", variable=self.early_stopping_mode_var, value="Manuel", command=self._update_ui_state).pack(side=tk.LEFT)
        ttk.Radiobutton(es_frame, text="Kapali", variable=self.early_stopping_mode_var, value="Kapali", command=self._update_ui_state).pack(side=tk.LEFT)

        self.entry_patience = ttk.Entry(es_frame, textvariable=self.early_stopping_patience_var, width=5)
        self.entry_patience.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        # 3. INFERENCE (EGRI CIKARIMI) GRUBU
        self.infer_frame = ttk.LabelFrame(content_frame, text="3. Egri Cikarimi (Inference)", padding="5")
        self.infer_frame.pack(fill=tk.X, pady=2)

        chk_infer_frame = ttk.Frame(self.infer_frame)
        chk_infer_frame.pack(anchor=tk.W)
        chk_infer = ttk.Checkbutton(chk_infer_frame, text="Aktif Et", variable=self.chk_inference_var, command=self._update_ui_state)
        chk_infer.pack(side=tk.LEFT)
        self._add_info_icon(chk_infer_frame, "Bu modulu (Egri Cikarimi) aktif veya pasif hale getirir.")

        self.infer_params_frame = ttk.Frame(self.infer_frame)
        self.infer_params_frame.pack(fill=tk.X)

        # Model Zoo - Combobox for model selection
        model_frame = ttk.Frame(self.infer_params_frame)
        model_frame.pack(fill=tk.X, pady=2)

        model_label_frame = ttk.Frame(model_frame)
        model_label_frame.pack(anchor=tk.W)
        ttk.Label(model_label_frame, text="Model (.pth):").pack(side=tk.LEFT)
        self._add_info_icon(model_label_frame, "Egitilmis model dosyasini (.pth) secin.")

        model_input_frame = ttk.Frame(model_frame)
        model_input_frame.pack(fill=tk.X)

        self.model_combo = ttk.Combobox(model_input_frame, textvariable=self.model_path_var, state='normal')
        self.model_combo.pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(model_input_frame, text="Yenile", width=8, command=self.refresh_models).pack(side=tk.RIGHT, padx=5)
        ttk.Button(model_input_frame, text="Sec...", width=8, command=self.browse_model_file).pack(side=tk.RIGHT, padx=5)

        ttk.Label(self.infer_params_frame, text="--- Girdi Secenekleri ---").pack(pady=(2,1))

        # Radyo Butonlari
        mode_frame = ttk.Frame(self.infer_params_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_frame, text="Klasor Modu (Resimler)", variable=self.inference_mode_var, value="folder", command=self._update_ui_state).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="PDF Modu", variable=self.inference_mode_var, value="pdf", command=self._update_ui_state).pack(side=tk.LEFT, padx=10)

        # Klasor Secenekleri Frame
        self.infer_folder_frame = ttk.Frame(self.infer_params_frame)
        self.infer_folder_frame.pack(fill=tk.X, pady=2)
        self._add_file_chooser(self.infer_folder_frame, "Giris Klasoru:", self.input_dir_var, is_file=False)

        # PDF Secenekleri Frame
        self.infer_pdf_frame = ttk.Frame(self.infer_params_frame)
        self.infer_pdf_frame.pack(fill=tk.X, pady=2)
        self._add_file_chooser(self.infer_pdf_frame, "PDF Dosyasi:", self.pdf_path_var, is_file=True)
        self._add_entry(self.infer_pdf_frame, "PDF Sayfalar:", self.pdf_pages_var, help_text="Ornek: '1-10' veya sadece '5'. Hepsi icin 'all' yazin.")

        ttk.Separator(self.infer_params_frame, orient='horizontal').pack(fill='x', pady=2)
        self._add_entry(self.infer_params_frame, "Threshold:", self.threshold_var, help_text="0.0 ile 1.0 arasi. Dusuk degerler daha fazla pikseli egri olarak siniflandirir.")

        self._add_file_chooser(self.infer_params_frame, "Cikis Klasoru:", self.output_dir_var, is_file=False)

        chk_clean_out_frame = ttk.Frame(self.infer_params_frame)
        chk_clean_out_frame.pack(anchor=tk.W, padx=5)
        chk_clean_out = ttk.Checkbutton(chk_clean_out_frame, text="Baslamadan Once Cikis Klasorunu Temizle", variable=self.chk_clean_output_var)
        chk_clean_out.pack(side=tk.LEFT)
        self._add_info_icon(chk_clean_out_frame, "DIKKAT: Isaretlenirse cikis klasorunun (orn: segmentation_results) icerigini tamamen siler.")

        # 4. EXCEL AKTARIMI GRUBU
        self.extract_frame = ttk.LabelFrame(content_frame, text="4. Excel Aktarimi", padding="5")
        self.extract_frame.pack(fill=tk.X, pady=2)

        chk_ext_frame = ttk.Frame(self.extract_frame)
        chk_ext_frame.pack(anchor=tk.W)
        chk_ext = ttk.Checkbutton(chk_ext_frame, text="Aktif Et", variable=self.chk_extract_data_var, command=self._update_ui_state)
        chk_ext.pack(side=tk.LEFT)
        self._add_info_icon(chk_ext_frame, "Tespit edilen egrileri dijitallestirip Excel dosyasina (.xlsx) aktarir.")

        # 5. HATA DUZELTME GRUBU
        self.correction_frame = ttk.LabelFrame(content_frame, text="5. Hata Duzeltme (Baseline)", padding="5")
        self.correction_frame.pack(fill=tk.X, pady=2)

        chk_corr_frame = ttk.Frame(self.correction_frame)
        chk_corr_frame.pack(anchor=tk.W)
        chk_corr = ttk.Checkbutton(chk_corr_frame, text="Aktif Et", variable=self.chk_error_correction_var, command=self._update_ui_state)
        chk_corr.pack(side=tk.LEFT)
        self._add_info_icon(chk_corr_frame, "Grafik verilerindeki havacilik duzeltmelerini geri alarak ham veri elde eder.")

        self.correction_params_frame = ttk.Frame(self.correction_frame)
        self.correction_params_frame.pack(fill=tk.X)
        self._add_file_chooser(self.correction_params_frame, "Düzeltme Klasörü:", self.correction_dir_var, is_file=False)

        # PRESET (KAYDET/YUKLE) BUTONLARI
        preset_frame = ttk.Frame(content_frame)
        preset_frame.pack(fill=tk.X, pady=5)

        ttk.Button(preset_frame, text="Ayarlari Kaydet", command=self.save_preset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(preset_frame, text="Ayarlari Yukle", command=self.load_preset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # 5. KONTROL BUTONLARI
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_start = ttk.Button(btn_frame, text="BASLAT", command=self.start_process, width=15)
        self.btn_start.pack(side=tk.LEFT, padx=5, expand=True)

        self.btn_stop = ttk.Button(btn_frame, text="DURDUR", command=self.stop_process, width=15, state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=5, expand=True)

        self.btn_check = ttk.Button(btn_frame, text="Ortami Kontrol Et", command=self.check_environment)
        self.btn_check.pack(side=tk.LEFT, padx=5, expand=True)

        # Ilk durum guncellemesi
        self.root.after(100, self._update_ui_state)


        # --- Sag Panel (Tab'li Yapi) ---
        right_panel = ttk.Frame(self.main_paned, padding="5")
        self.main_paned.add(right_panel, weight=1) # Sag taraf buyumeye daha meyilli olsun

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Loglar
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="Islem Loglari")

        self.log_text = scrolledtext.ScrolledText(log_tab, state='disabled', height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('info', foreground='blue')

        # Tab 2: Grafikler
        graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(graph_tab, text="Canli Grafikler")

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_iou = self.figure.add_subplot(212)

        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._reset_plots()

        # Tab 3: Onizleme / Duzenleme
        preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(preview_tab, text="Gorsel Onizleme & Duzenleme")

        preview_ctrl_frame = ttk.Frame(preview_tab)
        preview_ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(preview_ctrl_frame, text="Rastgele Egitim Verisi", command=self.show_random_training_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(preview_ctrl_frame, text="Rastgele Sonuc", command=self.show_random_inference_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(preview_ctrl_frame, text="Dosya Sec...", command=self.select_file_for_preview).pack(side=tk.LEFT, padx=5)

        ttk.Separator(preview_ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        self.btn_prev = ttk.Button(preview_ctrl_frame, text="< Onceki", command=self.prev_preview, state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next = ttk.Button(preview_ctrl_frame, text="Sonraki >", command=self.next_preview, state="disabled")
        self.btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Separator(preview_ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        self.btn_edit_mask = ttk.Button(preview_ctrl_frame, text="Maskeyi Duzenle", command=self.open_mask_editor, state="disabled")
        self.btn_edit_mask.pack(side=tk.LEFT, padx=5)

        self.preview_fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_preview = self.preview_fig.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_tab)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Tab 4: Data Statistics
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="Veri Analizi")

        stats_ctrl_frame = ttk.Frame(stats_tab)
        stats_ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(stats_ctrl_frame, text="Analizi Baslat", command=self.analyze_dataset).pack(side=tk.LEFT, padx=5)

        self.stats_text = scrolledtext.ScrolledText(stats_tab, state='disabled', height=20, font=("Courier", 10))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Baslangic Boyutu: Yari Yariya (50/50) ---
        # Pencere render edildikten sonra ayiriciyi ortaya cek
        self.root.update()
        total_width = self.root.winfo_width()
        self.main_paned.sashpos(0, total_width // 2)

    def select_file_for_preview(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")])
        if filename:
            self._display_preview(filename)

    def show_random_training_sample(self):
        dataset_path = self.dataset_path_var.get()
        images_dir = os.path.join(dataset_path, "images")
        masks_dir = os.path.join(dataset_path, "masks")
        if not os.path.exists(images_dir):
            messagebox.showerror("Hata", "Klasor yok")
            return
        files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        if not files: return

        # Dosya listesini kaydet (Navigasyon icin)
        self.preview_files = [os.path.join(images_dir, f) for f in files]

        choice_idx = random.randint(0, len(self.preview_files)-1)
        self.preview_index = choice_idx

        img_path = self.preview_files[choice_idx]
        self._display_preview(img_path, masks_dir)
        self._update_nav_buttons()

    def show_random_inference_result(self):
        output_dir = self.output_dir_var.get()
        if not os.path.exists(output_dir): return
        files = []
        for root, _, filenames in os.walk(output_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg')):
                    files.append(os.path.join(root, f))
        if not files: return

        # Dosya listesini kaydet
        self.preview_files = files

        choice_idx = random.randint(0, len(files)-1)
        self.preview_index = choice_idx
        choice = files[choice_idx]

        self._display_inference_preview(choice)
        self._update_nav_buttons()

    def prev_preview(self):
        if not self.preview_files: return
        self.preview_index = (self.preview_index - 1) % len(self.preview_files)
        self._update_preview_from_index()

    def next_preview(self):
        if not self.preview_files: return
        self.preview_index = (self.preview_index + 1) % len(self.preview_files)
        self._update_preview_from_index()

    def _update_preview_from_index(self):
        if not self.preview_files: return
        path = self.preview_files[self.preview_index]

        # Path training mi inference mi anlamaya calis
        # Basitce: icerisinde 'images' ve 'img_' varsa trainingdir (varsayim)
        if "dataset_production" in path: # Training tahmini
             dataset_path = self.dataset_path_var.get()
             masks_dir = os.path.join(dataset_path, "masks")
             self._display_preview(path, masks_dir)
        else:
             self._display_inference_preview(path)

    def _update_nav_buttons(self):
        if self.preview_files:
            self.btn_prev.config(state="normal")
            self.btn_next.config(state="normal")
        else:
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")

    def _display_inference_preview(self, choice):
        try:
            img = cv2.imread(choice)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.ax_preview.clear()
            self.ax_preview.set_title(f"Result [{self.preview_index+1}/{len(self.preview_files)}]: {os.path.basename(choice)}")
            self.ax_preview.imshow(img)
            self.ax_preview.axis('off')
            self.preview_canvas.draw()

            # Enable editing for this file itself
            self.current_preview_img = choice
            self.current_preview_mask = choice
            self.btn_edit_mask.config(state="normal")

        except Exception as e:
            messagebox.showerror("Hata", f"Onizleme hatasi: {e}")

    def _display_preview(self, img_path, mask_search_dir=None):
        try:
            img = cv2.imread(img_path)
            if img is None: return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Maske bulmaya calis
            mask_path = None
            if mask_search_dir:
                basename = os.path.basename(img_path)
                # Olasi maske isimleri
                candidates = [
                    os.path.join(mask_search_dir, basename.replace("img_", "mask_")),
                    os.path.join(mask_search_dir, basename)
                ]
                for c in candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break

            self.current_preview_img = img_path
            self.current_preview_mask = mask_path

            self.ax_preview.clear()

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Overlay
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    # Kirmizi goster
                    mask_rgb[:, :, 1] = 0
                    mask_rgb[:, :, 2] = 0

                    self.ax_preview.imshow(np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)]))
                    self.btn_edit_mask.config(state="normal")
                else:
                    self.ax_preview.imshow(img)
                    self.btn_edit_mask.config(state="disabled")
            else:
                self.ax_preview.imshow(img)
                self.btn_edit_mask.config(state="disabled")

            title = f"{os.path.basename(img_path)}"
            if self.preview_files:
                 title = f"Training [{self.preview_index+1}/{len(self.preview_files)}]: {title}"

            self.ax_preview.set_title(title)
            self.ax_preview.axis('off')
            self.preview_canvas.draw()

        except Exception as e:
            print(f"Preview error: {e}")

    def open_mask_editor(self):
        if self.current_preview_img and self.current_preview_mask:
            MaskEditor(self.root, self.current_preview_img, self.current_preview_mask)

    def _reset_plots(self):
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.epochs = []

        self.ax_loss.clear()
        self.ax_loss.set_title("Training & Validation Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        self.ax_iou.clear()
        self.ax_iou.set_title("Training & Validation IoU")
        self.ax_iou.set_xlabel("Epoch")
        self.ax_iou.set_ylabel("IoU")
        self.ax_iou.grid(True)

        self.canvas.draw()

    def _update_plots(self, epoch, t_loss, v_loss, t_iou, v_iou):
        self.epochs.append(epoch)
        self.train_losses.append(t_loss)
        self.val_losses.append(v_loss)
        self.train_ious.append(t_iou)
        self.val_ious.append(v_iou)

        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        self.ax_loss.plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
        self.ax_loss.set_title("Training & Validation Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        self.ax_iou.clear()
        self.ax_iou.plot(self.epochs, self.train_ious, 'b-', label='Train IoU')
        self.ax_iou.plot(self.epochs, self.val_ious, 'r-', label='Val IoU')
        self.ax_iou.set_title("Training & Validation IoU")
        self.ax_iou.legend()
        self.ax_iou.grid(True)

        self.canvas.draw()

    def _check_plot_queue(self):
        while not self.plot_queue.empty():
            data = self.plot_queue.get_nowait()
            self._update_plots(*data)
        self.root.after(500, self._check_plot_queue)

    def refresh_models(self):
        """Scan checkpoints folder and populate model combobox"""
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            self.model_combo['values'] = []
            messagebox.showwarning("Uyari", f"'{checkpoints_dir}' klasoru bulunamadi!")
            return

        model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        model_paths = [os.path.join(checkpoints_dir, f) for f in model_files]

        self.model_combo['values'] = model_paths
        if model_paths:
            self.model_combo.current(0)
            self.log(f"Toplam {len(model_paths)} model bulundu.", 'info')
        else:
            messagebox.showinfo("Bilgi", "Hic .pth model dosyasi bulunamadi.")

    def analyze_dataset(self):
        """Analyze dataset and display statistics"""
        dataset_path = self.dataset_path_var.get()
        images_dir = os.path.join(dataset_path, "images")
        masks_dir = os.path.join(dataset_path, "masks")

        if not os.path.exists(images_dir):
            messagebox.showerror("Hata", f"Dataset bulunamadi: {images_dir}")
            return

        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Analiz ediliyor...\n")
        self.stats_text.config(state='disabled')

        def _analyze():
            try:
                image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                total_images = len(image_files)

                pixel_means = []
                mask_coverages = []

                for img_file in image_files[:min(100, total_images)]:  # Sample first 100
                    img_path = os.path.join(images_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        pixel_means.append(np.mean(img))

                    mask_file = img_file.replace("img_", "mask_")
                    mask_path = os.path.join(masks_dir, mask_file)
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            coverage = (np.sum(mask > 127) / mask.size) * 100
                            mask_coverages.append(coverage)

                avg_pixel = np.mean(pixel_means) if pixel_means else 0
                avg_coverage = np.mean(mask_coverages) if mask_coverages else 0

                result = f"""
=== VERI SETI ANALIZ RAPORU ===

Toplam Goruntu Sayisi: {total_images}
Analiz Edilen Ornek: {min(100, total_images)}

--- Goruntu Istatistikleri ---
Ortalama Piksel Degeri: {avg_pixel:.2f} / 255
(255'e yakin = Parlak, 0'a yakin = Karanlik)

--- Maske Istatistikleri ---
Ortalama Maske Doluluk Orani: {avg_coverage:.2f}%
(Egrilerin toplam alandaki yuzdesi)

ONERILER:
"""
                if avg_coverage < 5:
                    result += "- Maske doluluk orani cok dusuk. Egriler yeteri kadar belirgin degil olabilir.\n"
                elif avg_coverage > 50:
                    result += "- Maske doluluk orani cok yuksek. Arka plan gurultusu olabilir.\n"
                else:
                    result += "- Maske doluluk orani dengeli gorunuyor.\n"

                if avg_pixel < 50:
                    result += "- Goruntuler cok karanlik. Kontrast artirilabilir.\n"
                elif avg_pixel > 200:
                    result += "- Goruntuler cok parlak. Kontrasti dengelemek iyi olabilir.\n"

                self.stats_text.config(state='normal')
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, result)
                self.stats_text.config(state='disabled')

            except Exception as e:
                self.stats_text.config(state='normal')
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, f"Hata olustu: {e}")
                self.stats_text.config(state='disabled')

        threading.Thread(target=_analyze, daemon=True).start()

    def toggle_theme(self, theme):
        """Toggle between light and dark themes with actual color changes"""
        style = ttk.Style()

        if theme == "dark":
            bg_color  = '#2b2b2b'
            fg_color  = '#ffffff'
            select_bg = '#404040'
            entry_bg  = '#3c3f41'
            text_bg   = '#1e1e1e'
            plot_bg   = '#1e1e1e'
            plot_fg   = '#ffffff'

            style.theme_use('alt')
            style.configure('TFrame',           background=bg_color)
            style.configure('TLabel',           background=bg_color, foreground=fg_color)
            style.configure('TLabelframe',      background=bg_color, foreground=fg_color)
            style.configure('TLabelframe.Label',background=bg_color, foreground=fg_color)
            style.configure('TButton',          background=select_bg, foreground=fg_color)
            style.configure('TCheckbutton',     background=bg_color, foreground=fg_color)
            style.configure('TRadiobutton',     background=bg_color, foreground=fg_color)
            style.configure('TNotebook',        background=bg_color)
            style.configure('TNotebook.Tab',    background=select_bg, foreground=fg_color)
            style.map('TNotebook.Tab', background=[('selected', entry_bg)])
            style.configure('TEntry',           fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)
            style.configure('TCombobox',        fieldbackground=entry_bg, foreground=fg_color, background=select_bg)
            # Checkbox: onay isareti (select) ve disabled durumu
            style.configure('TCheckbutton',     background=bg_color, foreground=fg_color, indicatorcolor=entry_bg)
            style.map('TCheckbutton',
                      foreground=[('disabled', '#888888')],
                      background=[('disabled', bg_color)],
                      indicatorcolor=[('selected', '#74c0fc'), ('!selected', entry_bg)])
            style.configure('TRadiobutton',     background=bg_color, foreground=fg_color)
            style.map('TRadiobutton',
                      foreground=[('disabled', '#888888')],
                      background=[('disabled', bg_color)])
            # Disabled entry/combobox -> koyu gri
            style.map('TEntry',
                      fieldbackground=[('disabled', '#333333')],
                      foreground=[('disabled', '#777777')])
            style.map('TCombobox',
                      fieldbackground=[('disabled', '#333333'), ('readonly', entry_bg)],
                      foreground=[('disabled', '#777777')])

            self.root.config(bg=bg_color)

        else:
            bg_color  = '#f0f0f0'
            fg_color  = '#000000'
            entry_bg  = '#ffffff'
            text_bg   = '#ffffff'
            select_bg = '#e0e0e0'
            plot_bg   = '#ffffff'
            plot_fg   = '#000000'

            style.theme_use('alt')
            style.configure('TFrame',           background=bg_color)
            style.configure('TLabel',           background=bg_color, foreground=fg_color)
            style.configure('TLabelframe',      background=bg_color, foreground=fg_color)
            style.configure('TLabelframe.Label',background=bg_color, foreground=fg_color)
            style.configure('TButton',          background=select_bg, foreground=fg_color)
            style.configure('TCheckbutton',     background=bg_color, foreground=fg_color, indicatorcolor=entry_bg)
            style.map('TCheckbutton',
                      foreground=[('disabled', '#888888')],
                      background=[('disabled', bg_color)],
                      indicatorcolor=[('selected', '#0078d7'), ('!selected', entry_bg)])
            style.configure('TRadiobutton',     background=bg_color, foreground=fg_color)
            style.map('TRadiobutton', foreground=[('disabled', '#888888')], background=[('disabled', bg_color)])
            style.configure('TEntry',           fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)
            style.map('TEntry', fieldbackground=[('disabled', '#e0e0e0')], foreground=[('disabled', '#888888')])
            style.configure('TCombobox',        fieldbackground=entry_bg, foreground=fg_color, background=select_bg)
            style.map('TCombobox', fieldbackground=[('disabled', '#e0e0e0'), ('readonly', entry_bg)], foreground=[('disabled', '#888888')])
            style.configure('TNotebook',        background=bg_color)
            style.configure('TNotebook.Tab',    background=select_bg, foreground=fg_color)
            style.map('TNotebook.Tab', background=[('selected', entry_bg)])

            self.root.config(bg=bg_color)

        # --- Recursive: ttk stilinin ulasamadigi tk.* widget'larini renklendirme ---
        def _apply(widget):
            cls = widget.winfo_class()
            try:
                if cls in ('Text',):
                    widget.config(bg=text_bg, fg=fg_color, insertbackground=fg_color)
                elif cls == 'Entry':
                    widget.config(bg=entry_bg, fg=fg_color, insertbackground=fg_color)
                elif cls == 'Listbox':
                    widget.config(bg=entry_bg, fg=fg_color)
                elif cls in ('Frame', 'Label'):
                    widget.config(bg=bg_color, fg=fg_color)
                elif cls == 'Canvas':
                    widget.config(bg=bg_color)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _apply(child)

        _apply(self.root)

        # --- Log ve stats metin alanlarini ozellikle guncelle ---
        for txt_widget in (self.log_text, self.stats_text):
            try:
                txt_widget.config(bg=text_bg, fg=fg_color, insertbackground=fg_color)
            except Exception:
                pass

        # --- Matplotlib grafik arka planlarini guncelle ---
        for fig, canvas in [(self.figure, self.canvas), (self.preview_fig, self.preview_canvas)]:
            try:
                fig.set_facecolor(plot_bg)
                for ax in fig.get_axes():
                    ax.set_facecolor(plot_bg)
                    ax.tick_params(colors=plot_fg)
                    ax.xaxis.label.set_color(plot_fg)
                    ax.yaxis.label.set_color(plot_fg)
                    ax.title.set_color(plot_fg)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(plot_fg)
                canvas.draw()
            except Exception:
                pass

        # --- Sol panel scrollbar rengini guncelle (tk.Scrollbar dogrudan renk destekler) ---
        if theme == "dark":
            try:
                self.left_scrollbar.config(
                    bg='#d0d0d0',
                    activebackground='#ffffff',
                    troughcolor='#3c3f41'
                )
            except Exception:
                pass
        else:
            try:
                self.left_scrollbar.config(
                    bg='#c0c0c0',
                    activebackground='#909090',
                    troughcolor='#f0f0f0'
                )
            except Exception:
                pass

        # --- Log metin tag renklerini temaya gore guncelle ---
        if theme == "dark":
            self.log_text.tag_config('error',   foreground='#ff6b6b')
            self.log_text.tag_config('success',  foreground='#69db7c')
            self.log_text.tag_config('info',     foreground='#74c0fc')
            self.log_text.tag_config('warning',  foreground='#ffd43b')
            self.log_text.config(fg='#ffffff')
        else:
            self.log_text.tag_config('error',   foreground='red')
            self.log_text.tag_config('success',  foreground='green')
            self.log_text.tag_config('info',     foreground='blue')
            self.log_text.tag_config('warning',  foreground='orange')
            self.log_text.config(fg='#000000')

        self.current_theme = theme  # Tema durumunu kaydet
        self.log(f"Tema degistirildi: {theme.capitalize()}", 'info')

    def _on_about_close(self):
        """Hakkinda penceresi kapatildiginda referansi temizle"""
        try:
            self._about_win.destroy()
        except Exception:
            pass
        self._about_win = None

    def show_about_window(self):
        """Show comprehensive About window with all GUI control explanations"""
        # Zaten aciksa one cikar, yeni pencere acma
        if hasattr(self, '_about_win') and self._about_win is not None:
            try:
                self._about_win.lift()
                self._about_win.focus_force()
                return
            except tk.TclError:
                pass  # Pencere kapanmis, yeni ac

        about_win = tk.Toplevel(self.root)
        self._about_win = about_win
        about_win.protocol("WM_DELETE_WINDOW", self._on_about_close)
        about_win.title("Hakkinda - Program ve Kontroller")
        about_win.geometry("720x700")

        # Temaya gore renkleri belirle
        is_dark = getattr(self, 'current_theme', 'light') == 'dark'
        win_bg   = "#1e1e1e" if is_dark else "#f5f5f5"
        txt_bg   = "#1e1e1e" if is_dark else "#ffffff"
        txt_fg   = "#d4d4d4" if is_dark else "#000000"

        about_win.configure(bg=win_bg)

        txt = scrolledtext.ScrolledText(
            about_win, wrap=tk.WORD, padx=16, pady=12,
            font=("Consolas", 10), bg=txt_bg, fg=txt_fg,
            insertbackground=txt_fg, relief="flat", borderwidth=0
        )
        txt.pack(fill=tk.BOTH, expand=True)

        # ── Renk etiketleri: koyu temada parlak VS Code renkleri, acik temada yuksek kontrastli renkler ──
        if is_dark:
            txt.tag_config("banner",   foreground="#00d4ff", font=("Consolas", 12, "bold"))
            txt.tag_config("subtitle", foreground="#a0a0a0", font=("Consolas", 10, "italic"))
            txt.tag_config("section",  foreground="#ffd700", font=("Consolas", 11, "bold"))
            txt.tag_config("subsec",   foreground="#4ec9b0", font=("Consolas", 10, "bold"))
            txt.tag_config("key",      foreground="#9cdcfe", font=("Consolas", 10))
            txt.tag_config("val",      foreground="#f3deb7", font=("Consolas", 10))  # daha parlak portakal-sari
            txt.tag_config("bullet",   foreground="#ffffff", font=("Consolas", 10))  # tam beyaz
            txt.tag_config("check",    foreground="#6a9955", font=("Consolas", 10, "bold"))
            txt.tag_config("new",      foreground="#f44747", font=("Consolas",  9, "bold"))
            txt.tag_config("sep",      foreground="#5c5c5c", font=("Consolas",  9))  # daha belirgin gri
        else:
            txt.tag_config("banner",   foreground="#003366", font=("Consolas", 12, "bold")) # Koyu Lacivert
            txt.tag_config("subtitle", foreground="#111111", font=("Consolas", 10, "italic", "bold")) # Tam siyah
            txt.tag_config("section",  foreground="#800000", font=("Consolas", 11, "bold")) # Bordo
            txt.tag_config("subsec",   foreground="#004d00", font=("Consolas", 10, "bold")) # Koyu Yesil
            txt.tag_config("key",      foreground="#000080", font=("Consolas", 10, "bold")) # Koyu Mavi
            txt.tag_config("val",      foreground="#000000", font=("Consolas", 10))         # Tam Siyah
            txt.tag_config("bullet",   foreground="#000000", font=("Consolas", 10, "bold")) # Tam Siyah Kalin
            txt.tag_config("check",    foreground="#006600", font=("Consolas", 10, "bold")) # Koyu Yesil
            txt.tag_config("new",      foreground="#cc0000", font=("Consolas",  9, "bold")) # Kirmizi
            txt.tag_config("sep",      foreground="#404040", font=("Consolas",  9, "bold")) # Griy
        txt.tag_config("nl",       font=("Consolas", 4))   # kucuk bosluk

        def w(text, tag="bullet"):
            txt.insert(tk.END, text, tag)

        def nl(n=1):
            txt.insert(tk.END, "\n" * n)

        def sep():
            w("  " + "─" * 62 + "\n", "sep")

        # ── BANNER ─────────────────────────────────────────────────────────
        nl()
        w("  ███████╗███████╗███╗   ██╗████████╗███████╗████████╗██╗██╗\n", "banner")
        w("  ╚══════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝╚══██╔══╝██║██║\n", "banner")
        w("    ████╗ █████╗  ██╔██╗ ██║   ██║   █████╗     ██║   ██║██║\n", "banner")
        w("       ██╗██╔══╝  ██║╚██╗██║   ██║   ██╔══╝     ██║   ██║██║\n", "banner")
        w("  ███████║███████╗██║ ╚████║   ██║   ███████╗   ██║   ██║██║\n", "banner")
        nl()
        w("  Sentetik Veri Uretimi ve Egri Analizi  │  v2.0\n", "subtitle")
        w("  Derin Ogrenme Tabanli Grafik Isleme Platformu\n", "subtitle")
        nl()
        sep()

        # ── PROGRAM HAKKINDA ───────────────────────────────────────────────
        nl()
        w("  ◆ U-NET CURVE SEGMENTATION PROJESI\n", "section")
        nl()
        w("  Bu yazilim, sentetik ucak grafikleri olusturup, derin ogrenme (U-Net)\n", "bullet")
        w("  modelini egitmektedir. Egitilen model ile gercek PDF veya goruntuler\n  uzerindeki ", "bullet")
        w("muhendislik egrileri (curve)", "key")
        w(" yuksek dogrulukla\n  otomatik olarak segmente edilebilir ve bu veriler dijitallestirilerek\n", "bullet")
        w("  Excel formatina dokulebilir.\n", "bullet")
        nl()
        w("  Uctan uca 4 dev adimda tek tikla islem sunar:\n", "bullet")
        w("  [1]", "key")
        w(" Sentetik Data Uretimi ", "bullet")
        w(" [2]", "key")
        w(" U-Net Modeli Egitimi\n", "bullet")
        w("  [3]", "key")
        w(" Akilli PDF Tarama(OCR) ", "bullet")
        w(" [4]", "key")
        w(" Sayisallastirma (Excel Ciktisi)\n", "bullet")
        nl()
        sep()

        # ── SOL PANEL ──────────────────────────────────────────────────────
        nl()
        w("  ◆ KAPSAMLI MODULLER VE ISLEVLERI\n", "section")
        nl()

        panels = [
            ("1. Fabrika Modulu (Sentetik Uretim)", [
                ("Ozellik",          "Belirtilen adette, COCO JSON standardinda coklu islemci kullanarak yapay veri (Data Augmentation) uretir."),
                ("Ozel Etkiler",     "Blur, Shadow, Noise, Flip, Shear, Cutout gibi ileri seviye yontemlerle modelin her turlu kosula karsi direncli (robust) olmasini saglar."),
            ]),
            ("2. Derin Ogrenme Modulu (U-Net Egitimi)", [
                ("Mimari",           "1-Kanal(Gri) renk okuyarak resim uzerindeki curve kisimlarini siniflandirip 0-255 arasi Binary Mask tahmini uretir."),
                ("Degerlendirme",    "Modelin asiri ezber yapmasi (overfitting) izlenir. Dice Loss ve IoU(Cakisma Skoru) skorlariyla basarililar klasorune (checkpoints) atilir."),
            ]),
            ("3. Pratik Analiz Modulu (Segmentation & PDF)", [
                ("PDF Operasyonu",   "Poppler moduluyle yuzlerce sayfalik bir pdf dosyasinin icerisindeki pikselleri grafik resimlerine boler."),
                ("OCR Robotu",       "Tesseract kullanarak tablolarin motorunu, agirligini, irtifasini anlayarak veri isimlendirmesi gerceklestirir."),
            ]),
            ("4. Sonuc ve Manuel Operasyon", [
                ("Maske Editoru",    "Yapay zekanin yaptigi tahmini begenmediyseniz cizimle silebilme/boyama olanagi."),
                ("Excel Seruveni",   "Cikarilan grafik verilerini Excel sayfalarina tablo sistemiyle okutur. Sayisallastirir."),
            ]),
        ]

        for title, items in panels:
            w(f"  ┌─ {title}\n", "subsec")
            for key, val in items:
                w(f"  │  • ", "sep")
                w(f"{key}", "key")
                w(f"  →  ", "sep")
                w(f"{val}\n", "val")
            nl()

        sep()

        # ── SAG PANEL ──────────────────────────────────────────────────────
        nl()
        w("  ◆ SAG PANEL SEKMELERI\n", "section")
        nl()

        right_panels = [
            ("Islem Loglari",         "Islem ciktilari | Kirmizi=Hata  Yesil=Basari  Mavi=Bilgi"),
            ("Canli Grafikler",       "Egitim sirasinda Loss/IoU grafiklerini anlik cizer"),
            ("Gorsel Onizleme",       "Uretilen gorseller ve model tahminleri | Maske duzenleyici"),
            ("Veri Analizi  ✦ YENI", "Piksel istatistikleri, maske doluluk orani, oneriler"),
        ]

        for name, desc in right_panels:
            w(f"  ▸ ", "sep")
            w(f"{name}\n", "subsec")
            w(f"    {desc}\n", "val")
            nl()

        sep()

        # ── OZEL OZELLIKLER ───────────────────────────────────────────────
        nl()
        w("  ◆ OZEL OZELLIKLER\n", "section")
        nl()
        features = [
            "Ayar Kaydetme / Yukleme  (JSON preset)",
            "Ilerleme Cubugu  (sol alt kose)",
            "Tooltip Yardimcilari  (imleci uzerine getirin)",
            "Model Zoo  (checkpoints klasorundeki tum modeller)",
            "Koyu / Acik Tema  (Gorunum menusu)",
        ]
        for f in features:
            w("  ✓  ", "check")
            w(f + "\n", "bullet")

        nl()
        sep()
        nl()
        w("  Sorulariniz icin:  ", "subtitle")
        w("Yardim  ›  Parametre Rehberi\n", "key")
        nl()

        txt.configure(state='disabled')


    def save_preset(self):
        """Mevcut ayarlari JSON dosyasina kaydeder"""
        data = {
            # Genel
            "num_images": self.num_images_var.get(),
            "num_workers": self.num_workers_var.get(),
            # Augmentation
            "aug_rotation": self.aug_rotation_var.get(),
            "aug_noise": self.aug_noise_var.get(),
            "aug_jpeg": self.aug_jpeg_var.get(),
            "aug_blur": self.aug_blur_var.get(),
            "aug_shadow": self.aug_shadow_var.get(),
            "aug_prob": self.aug_prob_var.get(),
            # Egitim
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "dataset_path": self.dataset_path_var.get(),
            "learning_rate": self.learning_rate_var.get(),
            "image_size": self.image_size_var.get(),
            "early_stopping_patience": self.early_stopping_patience_var.get(),
            "early_stopping_mode": self.early_stopping_mode_var.get(),
            # Inference
            "pdf_path": self.pdf_path_var.get(),
            "pdf_pages": self.pdf_pages_var.get(),
            "threshold": self.threshold_var.get(),
            "model_path": self.model_path_var.get(),
            "input_dir": self.input_dir_var.get(),
            "output_dir": self.output_dir_var.get(),
            # Checkboxlar
            "chk_gen_data": self.chk_gen_data_var.get(),
            "chk_train_model": self.chk_train_model_var.get(),
            "chk_inference": self.chk_inference_var.get(),
            "chk_extract_data": self.chk_extract_data_var.get(),
            "chk_error_correction": self.chk_error_correction_var.get(),
            "chk_clean_data": self.chk_clean_data_var.get(),
            "correction_dir": self.correction_dir_var.get()
        }

        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Basarili", f"Ayarlar kaydedildi:\n{filename}")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme hatasi: {e}")

    def load_preset(self):
        """JSON dosyasindan ayarlari yukler"""
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Degerleri yukle (Hata kontrolu basitce gecilir)
                if "num_images" in data: self.num_images_var.set(data["num_images"])
                if "num_workers" in data: self.num_workers_var.set(data["num_workers"])

                if "aug_rotation" in data: self.aug_rotation_var.set(data["aug_rotation"])
                if "aug_noise" in data: self.aug_noise_var.set(data["aug_noise"])
                if "aug_jpeg" in data: self.aug_jpeg_var.set(data["aug_jpeg"])
                if "aug_blur" in data: self.aug_blur_var.set(data["aug_blur"])
                if "aug_shadow" in data: self.aug_shadow_var.set(data["aug_shadow"])
                if "aug_prob" in data: self.aug_prob_var.set(data["aug_prob"])

                if "epochs" in data: self.epochs_var.set(data["epochs"])
                if "batch_size" in data: self.batch_size_var.set(data["batch_size"])
                if "dataset_path" in data: self.dataset_path_var.set(data["dataset_path"])
                if "learning_rate" in data: self.learning_rate_var.set(data["learning_rate"])
                if "image_size" in data: self.image_size_var.set(data["image_size"])
                if "early_stopping_patience" in data: self.early_stopping_patience_var.set(data["early_stopping_patience"])
                if "early_stopping_mode" in data: self.early_stopping_mode_var.set(data["early_stopping_mode"])

                if "pdf_path" in data: self.pdf_path_var.set(data["pdf_path"])
                if "pdf_pages" in data: self.pdf_pages_var.set(data["pdf_pages"])
                if "threshold" in data: self.threshold_var.set(data["threshold"])
                if "model_path" in data: self.model_path_var.set(data["model_path"])
                if "input_dir" in data: self.input_dir_var.set(data["input_dir"])
                if "output_dir" in data: self.output_dir_var.set(data["output_dir"])

                if "chk_gen_data" in data: self.chk_gen_data_var.set(data["chk_gen_data"])
                if "chk_train_model" in data: self.chk_train_model_var.set(data["chk_train_model"])
                if "chk_inference" in data: self.chk_inference_var.set(data["chk_inference"])
                if "chk_extract_data" in data: self.chk_extract_data_var.set(data["chk_extract_data"])
                if "chk_error_correction" in data: self.chk_error_correction_var.set(data["chk_error_correction"])
                if "chk_clean_data" in data: self.chk_clean_data_var.set(data["chk_clean_data"])
                if "correction_dir" in data: self.correction_dir_var.set(data["correction_dir"])

                self._update_ui_state()
                messagebox.showinfo("Basarili", "Ayarlar yuklendi!")

            except Exception as e:
                messagebox.showerror("Hata", f"Yukleme hatasi: {e}")

    def _add_info_icon(self, parent, text):
        """Kucuk bir bilgi ikonu ekler ve ToolTip baglar"""
        icon_lbl = ttk.Label(parent, text="ⓘ", cursor="hand2", foreground="#0078d7")
        icon_lbl.pack(side=tk.LEFT, padx=2)
        CreateToolTip(icon_lbl, text)
        return icon_lbl

    def _add_entry(self, parent, label_text, variable, help_text=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=1)

        lbl = ttk.Label(frame, text=label_text, width=25)
        lbl.pack(side=tk.LEFT)

        if help_text:
            self._add_info_icon(frame, help_text)

        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        return entry

    def _add_file_chooser(self, parent, label_text, variable, is_file=True, help_text=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        label_frame = ttk.Frame(frame)
        label_frame.pack(anchor=tk.W)

        ttk.Label(label_frame, text=label_text).pack(side=tk.LEFT)
        if help_text:
            self._add_info_icon(label_frame, help_text)

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X)

        entry = ttk.Entry(input_frame, textvariable=variable)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        if is_file:
            cmd = lambda: self._browse_file(variable)
        else:
            cmd = lambda: self._browse_dir(variable)

        btn = ttk.Button(input_frame, text="...", width=3, command=cmd)
        btn.pack(side=tk.RIGHT, padx=5)
        return btn

    def _browse_file(self, variable):
        filename = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth"), ("PDF Files", "*.pdf"), ("All Files", "*.*")])
        if filename:
            variable.set(filename)

    def _browse_dir(self, variable):
        dirname = filedialog.askdirectory()
        if dirname:
            variable.set(dirname)

    def browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Model Dosyasi Sec",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            initialdir="checkpoints" if os.path.exists("checkpoints") else "."
        )
        if filename:
            self.model_path_var.set(filename)

    def _update_ui_state(self):
        """Checkboxlarin durumuna gore icindeki widgetlari aktif/pasif yapar"""

        def set_state(parent, state):
            # Eger bu frame icinde "Aktif Et" butonu varsa, bu frame'in kendisini kapatma
            # (Cunku bazen frame'in 'disabled' olmasi icindekilerin de tiklanmasini engeller)
            has_active_chk = False
            for child in parent.winfo_children():
                if isinstance(child, ttk.Checkbutton) and "Aktif" in child.cget("text"):
                    has_active_chk = True
                    break

            for child in parent.winfo_children():
                # "Aktif Et" butonlarini ASLA kapatma
                if isinstance(child, ttk.Checkbutton) and "Aktif" in child.cget("text"):
                    child.configure(state='normal')
                    continue

                # Tum interaktif widgetlari kapsiyoruz
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Label,
                                      ttk.Radiobutton, ttk.Combobox,
                                      ttk.Spinbox, ttk.Scale)):
                    try:
                        child.configure(state=state)
                    except tk.TclError:
                        pass

                # Frame ise recursive olarak in
                if isinstance(child, (ttk.Frame, tk.Frame)):
                    set_state(child, state)

        # 1. Gen Data
        if self.chk_gen_data_var.get():
            self.gen_params_frame.pack(fill=tk.X)
            state = 'normal'
        else:
            self.gen_params_frame.pack_forget()
            state = 'disabled'
        set_state(self.gen_frame, state)

        # 2. Train
        train_active = self.chk_train_model_var.get()
        if train_active:
            self.train_params_frame.pack(fill=tk.X)
            state = 'normal'
        else:
            self.train_params_frame.pack_forget()
            state = 'disabled'
        set_state(self.train_frame, state)

        # Colab seciliyse yerel ayarlari kapat
        if train_active:
             is_colab = self.chk_colab_train_var.get()
             if is_colab:
                 set_state(self.local_train_options, 'disabled')
                 self.ent_colab.configure(state='normal')
                 self.lbl_colab.configure(state='normal')
             else:
                 set_state(self.local_train_options, 'normal')
                 self.ent_colab.configure(state='disabled')
                 self.lbl_colab.configure(state='disabled')
                 if hasattr(self, 'early_stopping_patience_var'):
                     if self.early_stopping_mode_var.get() == "Manuel":
                         self.entry_patience.configure(state='normal')
                     else:
                         self.entry_patience.configure(state='disabled')

        # 3. Inference
        infer_active = self.chk_inference_var.get()
        if infer_active:
            self.infer_params_frame.pack(fill=tk.X)
            state = 'normal'
        else:
            self.infer_params_frame.pack_forget()
            state = 'disabled'
        set_state(self.infer_frame, state)

        # Mode'a gore alt frameleri ac/kapat
        if infer_active:
             infer_mode = self.inference_mode_var.get()
             if infer_mode == 'folder':
                 set_state(self.infer_folder_frame, 'normal')
                 set_state(self.infer_pdf_frame, 'disabled')
             else:
                 set_state(self.infer_folder_frame, 'disabled')
                 set_state(self.infer_pdf_frame, 'normal')

        # 4. Extract
        if self.chk_extract_data_var.get():
            state = 'normal'
        else:
            state = 'disabled'
        set_state(self.extract_frame, state)

        # 5. Correction
        if self.chk_error_correction_var.get():
            self.correction_params_frame.pack(fill=tk.X)
            state = 'normal'
        else:
            self.correction_params_frame.pack_forget()
            state = 'disabled'
        set_state(self.correction_frame, state)
    def log(self, message, tag=None):
        self.log_queue.put((message, tag))

    def _check_queue(self):
        while not self.log_queue.empty():
            message, tag = self.log_queue.get_nowait()
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, message + "\n", tag)
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        self.root.after(100, self._check_queue)

    def run_command_in_thread(self, cmd_list, description):
        self.log(f"\n{'='*50}", 'info')
        self.log(f"BASLIYOR: {description}", 'info')
        self.log(f"Komut: {' '.join(cmd_list)}")
        self.log(f"{'='*50}\n", 'info')

        # Reset Progress
        self.progress_var.set(0)
        self.lbl_status.config(text=f"{description} calisiyor...")

        try:
            self.process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            # Regex Patterns
            prod_pattern = re.compile(r"\[PROGRESS\] (\d+)/(\d+)")
            epoch_pattern_progress = re.compile(r"Epoch (\d+)/(\d+)")
            epoch_pattern_stats = re.compile(r"Epoch (\d+)/(\d+) - Train Loss: ([\d\.]+), Val Loss: ([\d\.]+) - Train IoU: ([\d\.]+), Val IoU: ([\d\.]+)")

            for line in self.process.stdout:
                line_str = line.strip()
                self.log(line_str)

                # --- Progress Parsing ---

                # A) Production Progress
                prod_match = prod_pattern.search(line_str)
                if prod_match:
                    current = int(prod_match.group(1))
                    total = int(prod_match.group(2))
                    percent = (current / total) * 100
                    self.progress_var.set(percent)
                    self.lbl_status.config(text=f"Uretiliyor: {current}/{total} (%{percent:.1f})")

                # B) Training Progress (Epochs)
                epoch_match = epoch_pattern_progress.search(line_str)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    percent = (current_epoch / total_epochs) * 100
                    self.progress_var.set(percent)
                    self.lbl_status.config(text=f"Egitiliyor: Epoch {current_epoch}/{total_epochs} (%{percent:.1f})")

                    # C) Graph Stats
                    stats_match = epoch_pattern_stats.search(line_str)
                    if stats_match:
                        try:
                            # epoch = int(stats_match.group(1))
                            t_loss = float(stats_match.group(3))
                            v_loss = float(stats_match.group(4))
                            t_iou = float(stats_match.group(5))
                            v_iou = float(stats_match.group(6))

                            self.plot_queue.put((current_epoch, t_loss, v_loss, t_iou, v_iou))
                        except Exception as e:
                            print("Graph Parse error:", e)

            self.process.wait()

            if self.process is None: # Manually stopped
                 return False

            if self.process.returncode != 0:
                self.log(f"\n[HATA] Islem basarisiz oldu! Kod: {self.process.returncode}", 'error')
                self.lbl_status.config(text="Hata olustu!")
                return False

            self.log(f"\n[BASARILI] {description} tamamlandi.", 'success')
            self.progress_var.set(100)
            self.lbl_status.config(text=f"{description} Tamamlandi")
            return True

        except Exception as e:
            self.log(f"\n[EXCEPTION] {e}", 'error')
            self.lbl_status.config(text="Beklenmeyen Hata!")
            return False
        finally:
             self.process = None

    def check_environment(self):
        threading.Thread(target=self._run_check_env, daemon=True).start()

    def _run_check_env(self):
        self.btn_start.config(state='disabled')
        self.btn_check.config(state='disabled')

        self.run_command_in_thread(["python", "-c", "import torch; print(f'PyTorch: {torch.__version__}')"], "PyTorch Kontrolu")
        self.run_command_in_thread(["python", "-c", "import cv2; print(f'OpenCV: {cv2.__version__}')"], "OpenCV Kontrolu")

        self.btn_start.config(state='normal')
        self.btn_check.config(state='normal')

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.log("\n[KULLANICI] ISLEM DURDURULDU.", 'error')
            self.lbl_status.config(text="Kullanici Tarafindan Durduruldu")
            self.is_running = False
            self.btn_start.config(state='normal')
            self.btn_stop.config(state='disabled')

    def start_process(self):
        if self.is_running:
            return

        self.is_running = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal') # Stop active
        # Reset charts
        self._reset_plots()

        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            # 1. Sentetik Veri Uretimi
            if self.chk_gen_data_var.get():
                # Clean Data Option
                if self.chk_clean_data_var.get():
                     output_base = "dataset_production" # Hardcoded in script default, or customizable?
                     # Let's assume standard
                     if os.path.exists(output_base):
                         self.log(f"Eski veriler siliniyor: {output_base}")
                         try:
                            shutil.rmtree(output_base)
                            self.log("Temizleme Tamamlandi.")
                         except Exception as e:
                            self.log(f"Temizleme Hatasi: {e}", 'error')

                cmd = [
                    "python", "3-synthetic_production.py",
                    str(self.num_images_var.get()),
                    str(self.num_workers_var.get()),
                    "--rotation", str(self.aug_rotation_var.get()),
                    "--noise", str(self.aug_noise_var.get()),
                    "--jpeg", str(self.aug_jpeg_var.get()),
                    "--blur", str(self.aug_blur_var.get()),
                    "--shadow", str(self.aug_shadow_var.get()),
                    "--prob", str(self.aug_prob_var.get()),
                    "--flip", str(self.aug_flip_var.get()),
                    "--shear", str(self.aug_shear_var.get()),
                    "--hue", str(self.aug_hue_var.get()),
                    "--sat", str(self.aug_sat_var.get()),
                    "--cutout", str(self.aug_cutout_var.get()),
                    "--motion", str(self.aug_motion_var.get())
                ]
                if not self.run_command_in_thread(cmd, "Sentetik Veri Uretimi"):
                    raise Exception("Veri uretim hatasi/durduruldu")

            if not self.is_running: return # Check stop

            # 2. Egitim
            # 2. Egitim
            # 2. Egitim
            if self.chk_train_model_var.get():
                if self.chk_colab_train_var.get():
                    import webbrowser

                    # 1. Notebook'u Oku ve Kopyala (Ana Thread'de calistirilmali)
                    self.root.after(0, self._copy_notebook_content)

                    # 2. Tarayiciyi Ac
                    url = self.colab_link_var.get()
                    if not url: url = "https://colab.research.google.com/"

                    self.log(f"↪ [COLAB] Colab Notebook tarayicida aciliyor: {url}")
                    webbrowser.open(url)

                    if self.chk_inference_var.get():
                        self.log("UYARI: Colab egitimi senkron degildir. Inference adimi yerel model bekleyebilir.", 'warning')
                        # Yine de devam etsin mi? Hayir, model yoksa hata verecek.
                else:
                    es_mode = self.early_stopping_mode_var.get()
                    if es_mode == "Otomatik":
                        p_val = "10"
                    elif es_mode == "Kapali":
                        p_val = "0"
                    else:
                        p_val = str(self.early_stopping_patience_var.get())

                    cmd = [
                        "python", "train_unet.py",
                        "--dataset", self.dataset_path_var.get(),
                        "--epochs", str(self.epochs_var.get()),
                        "--batch_size", str(self.batch_size_var.get()),
                        "--learning_rate", str(self.learning_rate_var.get()),
                        "--image_size", str(self.image_size_var.get()),
                        "--patience", p_val
                    ]
                    if not self.run_command_in_thread(cmd, "Model Egitimi"):
                        raise Exception("Egitim hatasi/durduruldu")

            if not self.is_running: return

            # 3. Inference
            if self.chk_inference_var.get():
                model_path = self.model_path_var.get()
                input_dir = self.input_dir_var.get()
                pdf_path = self.pdf_path_var.get()
                output_dir = self.output_dir_var.get()

                # Cikti Klasorunu Temizle
                if self.chk_clean_output_var.get():
                    if os.path.exists(output_dir):
                        self.log(f"Bilgi: Cikti klasoru temizleniyor... ({output_dir})")
                        try:
                            shutil.rmtree(output_dir)
                            # Klasoru hemen geri olusturmaya gerek yok, 5-segment_curves yapar
                            # Ama eger excel export hemen pesinden gelirse ve inference fail ederse?
                            # Sorun degil.
                        except Exception as e:
                            self.log(f"UYARI: Klasor temizlenemedi: {e}", 'warning')

                if not os.path.exists(model_path):
                    self.log(f"HATA: Model dosyasi bulunamadi: {model_path}", 'error')
                    raise Exception("Model yok")

                cmd = [
                    "python", "-u", "5-segment_curves.py",
                    "--model", model_path,
                    "--output_dir", output_dir,
                    "--threshold", str(self.threshold_var.get())
                ]

                # Mode'a gore islem yap
                infer_mode = self.inference_mode_var.get()

                if infer_mode == 'pdf':
                    if pdf_path and os.path.exists(pdf_path):
                        cmd.extend(["--pdf", pdf_path])
                        cmd.extend(["--pages", self.pdf_pages_var.get()])
                        self.log(f"Bilgi: PDF modu secildi ({pdf_path})")
                    else:
                        self.log(f"HATA: Gecerli bir PDF dosyasi belirtmelisiniz!", 'error')
                        raise Exception("PDF yok")

                elif infer_mode == 'folder':
                    if input_dir and os.path.exists(input_dir):
                        cmd.extend(["--input_dir", input_dir])
                        self.log(f"Bilgi: Klasor modu secildi ({input_dir})")
                    else:
                        self.log(f"HATA: Gecerli bir Giris Klasoru belirtmelisiniz!", 'error')
                        raise Exception("Klasor yok")
                else:
                    self.log(f"HATA: Gecersiz inference modu: {infer_mode}", 'error')
                    raise Exception("Mod gecersiz")

                if not self.run_command_in_thread(cmd, "Egri Cikarimi (Inference)"):
                    raise Exception("Inference hatasi/durduruldu")

            if not self.is_running: return

            # 4. Veri Cikarimi (Excel)
            if self.chk_extract_data_var.get():
                input_dir = self.input_dir_var.get()
                output_dir = self.output_dir_var.get()
                excel_path = os.path.join(output_dir, "curve_data.xlsx")

                # Inference moduna gore orijinal resimlerin yerini belirle
                infer_mode = self.inference_mode_var.get()
                if infer_mode == 'pdf':
                    # PDF modunda resimler output_dir_extracted klasorune cikariliyor
                    original_dir = f"{output_dir}_extracted"
                else:
                    # Klasor modunda direkt input_dir
                    original_dir = input_dir

                cmd = [
                    "python", "-u", "extract_curve_data.py",
                    "--segmentation_dir", output_dir,
                    "--original_dir", original_dir,
                    "--output", excel_path
                ]

                self.run_command_in_thread(cmd, "Veri Cikarimi (Excel)")

            if not self.is_running: return

            # 5. Hata Duzeltme (Error Correction)
            if self.chk_error_correction_var.get():
                output_dir = self.correction_dir_var.get()
                found_any = False
                for suffix in ["_One_Engine", "_Two_Engine", ""]:
                    excel_input = os.path.join(output_dir, f"curve_data{suffix}.xlsx")
                    if os.path.exists(excel_input):
                        found_any = True
                        excel_output = os.path.join(output_dir, f"curve_data{suffix}_HAM_Baseline.xlsx")
                        cmd = [
                            "python", "6-correction_tool.py",
                            "-i", excel_input,
                            "-o", excel_output
                        ]
                        run_name = f"Hata Duzeltme ({suffix.strip('_').replace('_', ' ') or 'Genel'})"
                        self.run_command_in_thread(cmd, run_name)

                if not found_any:
                    self.log(f"UYARI: Girdi Excel dosyalari bulunamadi: {output_dir}/curve_data*.xlsx", 'warning')

            self.log("\n[TUM ISLEMLER TAMAMLANDI]", 'success')
            messagebox.showinfo("Basarili", "Tum islemler basariyla tamamlandi!")

        except Exception as e:
            if self.is_running: # If manual stop, don't show error box
                 self.log(f"\n[ISLEM DURDURULDU] {e}", 'error')
                 messagebox.showerror("Hata", f"Islem sirasinda hata olustu:\n{e}")

        finally:
            self.is_running = False
            self.root.after(0, lambda: self.btn_start.config(state='normal'))
            self.root.after(0, lambda: self.btn_stop.config(state='disabled'))

    def _copy_notebook_content(self):
        """Ana thread uzerinde clipboard islemi yapar"""
        try:
            import json
            # Absolute path kullan
            notebook_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_unet_colab.ipynb")

            if os.path.exists(notebook_file):
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    nb_content = json.load(f)

                code_cells = []
                # Header ekle
                code_cells.append(f"# --- OTOMATIK KOPYALANAN KOD ---\n# Kaynak: {os.path.basename(notebook_file)}\n\n")

                for cell in nb_content.get('cells', []):
                    if cell['cell_type'] == 'code':
                        source = "".join(cell['source'])
                        code_cells.append(f"# --- HUCRE ---\n{source}\n\n")
                    elif cell['cell_type'] == 'markdown':
                        source = "".join(cell['source'])
                        code_cells.append("\n".join([f"# {line}" for line in source.splitlines()]) + "\n\n")

                full_code = "".join(code_cells)
                self.root.clipboard_clear()
                self.root.clipboard_append(full_code)
                self.root.update()
                self.log(f"📋 [BASARILI] Kodlar panoya kopyalandi! (Ctrl+V ile yapistirin)")
            else:
                self.log(f"UYARI: '{notebook_file}' bulunamadi.", 'warning')
        except Exception as e:
            self.log(f"HATA: Kopyalama hatasi: {e}", 'error')


if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticDataGUI(root)
    root.mainloop()
