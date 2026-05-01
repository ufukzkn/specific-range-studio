# synthetic_production.py
"""
Uretim veri setesi ureteci ucak semalari icin.
5000+ bagini getirilen sentetik semalar yaratir.
Tek bir COCO JSON aciklama dosyasi - multiprocessing ile OPTIME edilmis.
"""

###############################################################################
# ENCODING FIX (Windows Console icin UTF-8 destegi)
###############################################################################
import sys
import io
# Windows konsolunda Turkce karakter sorunu cozumu
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

###############################################################################
# IMPORTS (Ice Aktarmalar)
###############################################################################
# 1. Standart Kutuphaneler:
# - os, shutil, pathlib: Dosya ve klasor yonetimi (olusturma, silme, yol islemleri)
# - json: Veri seti etiketlerini (annotations) kaydetmek icin
# - random, math: Rastgelelik (cesitlilik) ve matematiksel hesaplamalar icin
# - concurrent.futures, functools: COKLU ISLEM (Multiprocessing) icin kritik oneme sahiptir.
#   Kodun tum CPU cekirdeklerini kullanarak hizli calismasini saglar.
import os
import json
import math
import random
import time
import shutil
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 2. Ucuncu Taraf Kutuphaneler (Goruntu Isleme):
# - numpy: Goruntuleri sayisal matrisler olarak islemek icin temel kutuphane
# - cv2 (OpenCV): Goruntu okuma, yazma, dondurme ve yeniden boyutlandirma islemleri
# - PIL: Bazi ozel efektler ve goruntu iyilestirmeleri icin
import numpy as np
import cv2
from PIL import Image, ImageEnhance

# 3. Yerel Moduller (synthetic_main):
# Bu modul, projenin "Motoru" gibidir. sadece bir "Yonetici"dir.
# Asil grafik cizme isini buradaki fonksiyonlar yapar.
#
# - ChartConfig: Grafigin "Recetesi" (min/max degerler, egri sayisi vb.)
# - CurveData: Egri verilerini tutan yapi
# - draw_chart_matplotlib: Matplotlib kullanarak grafigi CIZEN asil ressam fonksiyon.
# - add_scan_artifacts: Temiz grafigi bozarak (gurultu, dondurme) "tarayicidan cikmis" gibi yapan fonksiyon.
# - colorize_curves_from_data: Egitim icin hedef (maske) goruntulerini boyayan fonksiyon.
from synthetic_main import (
    ChartConfig, CurveData, draw_chart_matplotlib,
    colorize_curves_from_data, generate_coco_annotation,
    add_scan_artifacts
)



class ProductionDatasetGenerator:
    """
    Preprocessing ile buyuk olcekli uretim veri seti olustur.

    Bu sinif bir 'FABRIKA MUDURU' gibi calisir. Asagidaki islemleri yonetir:
    1. Konfigurasyon Tasarimi: Her grafik icin rastgele bir plan olusturur.
    2. Uretim Hatti: Grafikleri cizer ve bozar (eskitir).
    3. Depolama: Goruntuleri ve etiketleri klasorlere yerlestirir.
    4. Kayit Tutma: COCO formatinda detayli bir envanter (json) tutar.

    Attributes:
        output_base (str): Fabrikanin cikti verecegi ana klasor yolu.
        coco_dataset (dict): Tum uretim verilerinin tutuldugu ana kayit defteri.
    """

    def __init__(self, output_base: str = "dataset_production", aug_config: Dict = None):
        """
        Baslatici metod (Kurulum Asamasi).

        Args:
            output_base (str): Veri setinin kaydedilecegi ana dizin.
            aug_config (dict): Veri artirma parametreleri.
        """
        self.output_base = output_base
        self.images_dir = os.path.join(output_base, "images")
        self.masks_dir = os.path.join(output_base, "masks")

        # Augmentation varsayilanlari
        self.aug_config = aug_config if aug_config else {
            "rotation_angle": 3.0,
            "noise_level": 15.0,
            "jpeg_quality": 70,
            "blur_level": 0,
            "shadow_enabled": 0,
            "probability": 0.7,
            "flip_mode": 0,        # 0: Kapali, 1: Yatay, 2: Dikey, 3: Her ikisi
            "shear_level": 0.0,    # 0.0 - 0.2 arasi ideal
            "hsv_jitter": [0, 0, 0], # [Hue, Sat, Val] degisim miktarlari (0-255)
            "cutout_count": 0,     # 0: Kapali, >0: Delik sayisi
            "motion_blur": 0       # 0: Kapali, >0: Kernel boyutu
        }

        # Dizinleri olustur (yoksa yaratir)
        for d in [self.images_dir, self.masks_dir]:
            os.makedirs(d, exist_ok=True)

        # COCO veri seti yapisini baslat
        self.coco_dataset = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "curve"}]
        }
        self.annotation_id = 1
        self.image_counter = 0

    def apply_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """
        Goruntuye uretim preprocessing'i uygula (Veri Artirma / Augmentation).
        """
        prob = self.aug_config.get("probability", 0.7)
        max_angle = self.aug_config.get("rotation_angle", 3.0)
        max_noise = self.aug_config.get("noise_level", 15.0)
        min_jpeg = self.aug_config.get("jpeg_quality", 70)
        blur_lvl = self.aug_config.get("blur_level", 0)
        shadow_en = self.aug_config.get("shadow_enabled", 0)

        # 1. Rastgele Dondurme (Rotation):
        if random.random() < prob and max_angle > 0:
            angle = random.uniform(-max_angle, max_angle)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 2. Yeniden Boyutlandirma (Resize):
        target_size = random.choice([1024, 1080, 1152, 1280])
        h, w = img.shape[:2]
        aspect = w / h

        if aspect > 1:
            new_w = target_size
            new_h = int(target_size / aspect)
        else:
            new_h = target_size
            new_w = int(target_size * aspect)

        new_w = (new_w // 2) * 2
        new_h = (new_h // 2) * 2

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        max_dim = max(new_w, new_h)
        pad_w = (max_dim - new_w) // 2
        pad_h = (max_dim - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 3. Gri Tonlama (Grayscale):
        if random.random() < prob:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # 4. Parlaklik ve Kontrast:
        if random.random() < prob:
            brightness_factor = random.uniform(0.90, 1.10)
            contrast_factor = random.uniform(0.85, 1.15)

            img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_factor*10)
            img = np.clip(img, 0, 255).astype(np.uint8)

        # 5. Gurultu Ekleme (Noise):
        if random.random() < prob and max_noise > 0:
            noise = np.random.normal(0, random.uniform(5, max_noise), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. JPEG Sikistirma (Compression Artifacts):
        if random.random() < prob and min_jpeg < 100:
            quality = random.randint(min_jpeg, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)

        # 7. Bulaniklik (Blur): Kamera odaksizligi veya hizli tarama
        if random.random() < prob and blur_lvl > 0:
            k = random.choice([x for x in range(3, blur_lvl + 2, 2)]) # Tek sayi uret (3, 5, 7...)
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 9. Flip (Cevirme):
        flip_mode = self.aug_config.get("flip_mode", 0)
        if flip_mode > 0 and random.random() < prob:
            if flip_mode == 1: # Horizontal
                img = cv2.flip(img, 1)
            elif flip_mode == 2: # Vertical
                img = cv2.flip(img, 0)
            elif flip_mode == 3: # Both
                img = cv2.flip(img, -1)

        # 10. Shear (Kaykilma):
        shear_val = self.aug_config.get("shear_level", 0.0)
        if shear_val > 0 and random.random() < prob:
            h, w = img.shape[:2]
            shear_x = random.uniform(-shear_val, shear_val)
            shear_y = random.uniform(-shear_val, shear_val)
            M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 11. HSV Jitter (Renk/Doygunluk):
        hsv_vals = self.aug_config.get("hsv_jitter", [0, 0, 0])
        if any(v > 0 for v in hsv_vals) and random.random() < prob:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            h_adj = random.uniform(-hsv_vals[0], hsv_vals[0])
            s_adj = random.uniform(1.0 - hsv_vals[1]/255, 1.0 + hsv_vals[1]/255)
            v_adj = random.uniform(1.0 - hsv_vals[2]/255, 1.0 + hsv_vals[2]/255)

            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_adj) % 180
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_adj, 0, 255)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_adj, 0, 255)
            img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 12. Cutout (Rastgele Kesikler):
        cut_count = self.aug_config.get("cutout_count", 0)
        if cut_count > 0 and random.random() < prob:
            h, w = img.shape[:2]
            for _ in range(random.randint(1, cut_count)):
                size = random.randint(20, 100)
                y1 = random.randint(0, h - size)
                x1 = random.randint(0, w - size)
                cv2.rectangle(img, (x1, y1), (x1 + size, y1 + size), (0, 0, 0), -1)

        # 13. Motion Blur (Hareket Bulanikligi):
        m_blur = self.aug_config.get("motion_blur", 0)
        if m_blur > 0 and random.random() < prob:
            size = random.choice([x for x in range(3, m_blur + 2, 2)])
            kernel_v = np.zeros((size, size))
            kernel_v[:, int((size - 1)/2)] = np.ones(size)
            kernel_v /= size
            img = cv2.filter2D(img, -1, kernel_v)

        return img

    @staticmethod
    def _generate_sample_worker(output_base: str, aug_config: Dict, sample_id: int) -> Tuple[Dict, List]:
        """
        Multiprocessing icin statik isci fonksiyonu.
        """
        try:
            from synthetic_main import ChartConfig, draw_chart_matplotlib, add_scan_artifacts

            # Augmentation konfigurasyonu ile baslat
            gen = ProductionDatasetGenerator(output_base, aug_config=aug_config)

            # Uretim (Generate): Grafigi ciz ve isle
            full_img, mask_img, curves, config = gen.generate_sample(sample_id, seed=None)

            h, w = full_img.shape[:2]

            # Kayit (Save)
            img_filename = f"img_{sample_id:05d}.png"
            mask_filename = f"mask_{sample_id:05d}.png"

            img_path = os.path.join(gen.images_dir, img_filename)
            mask_path = os.path.join(gen.masks_dir, mask_filename)

            cv2.imwrite(img_path, full_img)
            cv2.imwrite(mask_path, mask_img)

            # Sonuc Paketleme
            image_info = {
                "id": sample_id,
                "file_name": img_filename,
                "mask_file": mask_filename,
                "width": w,
                "height": h,
                "x_min": float(config.x_min),
                "x_max": float(config.x_max),
                "y_min": float(config.y_min),
                "y_max": float(config.y_max)
            }

            annotations = []

            # Anotasyonları doÄrudan mask objesinden (mask_img) çÄkarmali
            if len(mask_img.shape) == 3 and mask_img.shape[2] == 3:
                gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_mask = mask_img
            _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3: continue  # En az 3 nokta (bir çokgen için)

                # Çokgen noktalarını [x1, y1, x2, y2, ...] formatına düzleştir.
                segmentation = contour.flatten().tolist()

                x, y, bw, bh = cv2.boundingRect(contour)
                bbox_area = float(cv2.contourArea(contour))

                if bw > 0 and bh > 0:
                    annotation = {
                        "image_id": sample_id,
                        "category_id": 1,  # 1 = curve sınıfı
                        "area": bbox_area,
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "iscrowd": 0,
                        "segmentation": [segmentation]
                    }
                    annotations.append(annotation)

            return image_info, annotations

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def generate_dataset(self, n_samples: int = 5000, num_workers: int = 4):
        """
        Multiprocessing ile tam veri seti olustur.
        """
        print("[INFO] Generating {:,} charts with {} workers...".format(n_samples, num_workers))
        print(f"[INFO] Augmentation: {self.aug_config}")
        print("[INFO] Output: {}".format(self.output_base))
        print("=" * 60)

        start_time = time.time()

        # Config'i worker'a gecir
        worker_func = partial(self._generate_sample_worker, self.output_base, self.aug_config)

        successful_samples = 0
        failed_samples = 0

        coco_images = []
        coco_annotations = []
        annotation_id = 1

        def handle_result(idx, result):
            nonlocal annotation_id, successful_samples, failed_samples
            if result is not None:
                image_info, annotations = result
                coco_images.append(image_info)
                for ann in annotations:
                    ann['id'] = annotation_id
                    coco_annotations.append(ann)
                    annotation_id += 1
                successful_samples += 1
            else:
                failed_samples += 1

            if (idx + 1) % max(1, n_samples // 10) == 0 or (idx + 1) == n_samples:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta_secs = int((n_samples - idx - 1) / rate) if rate > 0 else 0
                print(f"[PROGRESS] {idx+1}/{n_samples} | E.T.A: {eta_secs}s")

        if num_workers <= 1:
            print("[INFO] Running sequential generation (no multiprocessing).")
            for idx in range(n_samples):
                try:
                    handle_result(idx, worker_func(idx))
                except Exception as e:
                    failed_samples += 1
                    print(f"[ERROR] Sample failed: {str(e)[:100]}")
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(worker_func, i): i
                    for i in range(n_samples)
                }

                for idx, future in enumerate(as_completed(futures)):
                    try:
                        handle_result(idx, future.result(timeout=60))
                    except Exception as e:
                        failed_samples += 1
                        print(f"[ERROR] Sample failed: {str(e)[:100]}")

        self.coco_dataset["images"] = coco_images
        self.coco_dataset["annotations"] = coco_annotations

        coco_path = os.path.join(self.output_base, "annotations.json")
        with open(coco_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_dataset, f, indent=2)

        print("[DONE] Üretim tamamlandı ({:.1f}s)".format(time.time() - start_time))

        # Fiziksel olarak böl (splits/train, splits/valid, splits/test)
        self.create_splits(self.coco_dataset)


    def advanced_random_config(self) -> ChartConfig:
        """
        Farkli grafik yapilandirmalari olustur.

        Bu metod:
        - Rastgele bir grafik stili secer (aircraft_range, exponential, vs.)
        - Secilen stile uygun eksen araliklari, egri sayisi ve tiplerini belirler
        - Drag (surukleme) indekslerini guvenli bir sekilde secer.
        """

        # 1. Grafik Stili Secimi (Style Selection):
        # Veri setinde cesitlilik saglamak icin 5 farkli stil tanimlanmistir.
        # Her stilin kendine has karakteristigi (tepecikli, yukselen, dalgali vb.) vardir.
        chart_styles = ['aircraft_range', 'exponential', 'power_law', 'multi_peak', 'declining']
        style = random.choice(chart_styles)

        # 2. Parametrelerin Belirlenmesi:
        # Secilen stile gore mantikli eksen araliklari ve egri tipleri atanir.

        # Tip A: Ucak Menzil Grafikleri (Aircraft Range)
        if style == 'aircraft_range':
            x_ranges = [(0.20, 0.95), (0.30, 1.00), (0.40, 1.10), (0.50, 1.20)]
            y_ranges = [(0.02, 0.15), (0.04, 0.18), (0.05, 0.20)]
            n_curves = random.randint(6, 12)
            curve_types = ['peaked', 'peaked_oval'] * 3 + ['rising', 'falling']

        # Tip B: Ustel Grafikler (Exponential)
        elif style == 'exponential':
            x_ranges = [(0.10, 1.50), (0.00, 1.20)]
            y_ranges = [(0.01, 0.25), (0.05, 0.30)]
            n_curves = random.randint(4, 9)
            curve_types = ['rising'] * 5 + ['falling'] * 2

        # Tip C: Guc Yasasi Grafikleri (Power Law)
        elif style == 'power_law':
            x_ranges = [(0.30, 1.20), (0.20, 1.00)]
            y_ranges = [(0.10, 0.50), (0.05, 0.45)]
            n_curves = random.randint(5, 10)
            curve_types = ['peaked', 'rising']

        # Tip D: Cok Tepecikli Grafikler (Multi-peak)
        elif style == 'multi_peak':
            x_ranges = [(0.20, 1.30), (0.00, 1.40)]
            y_ranges = [(0.05, 0.35), (0.10, 0.40)]
            n_curves = random.randint(4, 8)
            curve_types = ['wavy'] * 3 + ['peaked'] * 2

        # Tip E: Azalan Grafikler (Declining)
        else:  # declining
            x_ranges = [(0.30, 1.20), (0.10, 1.00)]
            y_ranges = [(0.05, 0.25), (0.10, 0.30)]
            n_curves = random.randint(5, 10)
            curve_types = ['falling'] * 4 + ['peaked']

        x_min, x_max = random.choice(x_ranges)
        y_min, y_max = random.choice(y_ranges)
        curve_type = random.choice(curve_types) if curve_types else 'peaked'

        # 3. Drag (Surukleme) Indekslerinin Secimi:
        # Havacilik grafiklerinde her egri bir "Drag Index" degerine karsilik gelir.
        # Bu bolum, grafikteki egri sayisi kadar indeksi guvenli bir sekilde secer.
        all_drag_indices = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300]
        max_drag_options = [100, 125, 150, 200, 250, 300]
        max_drag = random.choice(max_drag_options)

        available_drag = [d for d in all_drag_indices if d <= max_drag]

        # En az n_curves kadar indeksimiz oldugundan emin oluyoruz
        if len(available_drag) < n_curves:
            available_drag = all_drag_indices  # Yeterli degilse hepsini kullan

        # Guvenli sekilde ornekleme yapiyoruz
        sample_size = min(n_curves, len(available_drag))
        if sample_size > 0:
            selected_drag = sorted(random.sample(available_drag, sample_size))
            selected_drag.reverse() # Genellikle dusuk drag ustte olur
        else:
            selected_drag = [0]

        # 4. Sonuc (Return):
        # Tum kararlar ChartConfig nesnesi icinde paketlenip dondurulur.
        # Ayrica, izgara, oklar, bilgi kutucuklari gibi elemanlar rastgele acilir/kapanir.
        return ChartConfig(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            n_curves=n_curves,
            curve_type=curve_type,
            curve_lw=random.uniform(0.4, 0.7),
            drag_indices=selected_drag,
            add_grid=random.random() < 0.90,       # Izgara (%90)
            add_arrows=random.random() < 0.80,     # Oklar (%80)
            add_envelope_optimum=random.random() < 0.65, # Optimum zarfi (%65)
            add_envelope_endurance=random.random() < 0.35,
            add_vmax_line=random.random() < 0.20,
            add_text_boxes=random.random() < 0.75, # Bilgi kutulari (%75)
            add_drag_labels=random.random() < 0.50,
            # NEW: Diversity Features
            add_secondary_curves=random.random() < 0.30, # %30 carpet plot
            x_scale=random.choice(['linear', 'linear', 'log']) if x_min > 0 else 'linear',
            y_scale=random.choice(['linear', 'linear', 'log']) if y_min > 0 else 'linear',
            marker_style=random.choice([None, None, None, 'o', 'x', '.', 'd']),
            background_type=random.choice(['plain', 'plain', 'grid', 'grid', 'noisy_paper'])
        )

        prob = self.aug_config.get("probability", 0.7)
        max_angle = self.aug_config.get("rotation_angle", 3.0)
        max_noise = self.aug_config.get("noise_level", 15.0)
        min_jpeg = self.aug_config.get("jpeg_quality", 70)
        blur_lvl = self.aug_config.get("blur_level", 0)
        shadow_en = self.aug_config.get("shadow_enabled", 0)

        # 1. Rastgele Dondurme (Rotation):
        # Gercek hayatta kagitlar tarayiciya tam duz (0 derece) yerlestirilmez.
        # Bu kod, gorseli belirlenen acilara kadar rastgele dondurerek bu egriligi simule eder.
        if random.random() < prob and max_angle > 0:
            angle = random.uniform(-max_angle, max_angle)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 2. Yeniden Boyutlandirma (Resize):
        # Yapay zeka modelleri genellikle kare (ornegin 1024x1024) goruntulerle calisir.
        # Bu kisim, goruntunun en-boy oranini bozmadan belirlenen hedef boyuta kucultur.
        target_size = random.choice([1024, 1080, 1152, 1280])
        h, w = img.shape[:2]
        aspect = w / h

        if aspect > 1:  # enindené daha genis
            new_w = target_size
            new_h = int(target_size / aspect)
        else:  # yeninden daha uzun
            new_h = target_size
            new_w = int(target_size * aspect)

        # Boyutlarin cift oldugundan emin ol (model uyumlulugu icin)
        new_w = (new_w // 2) * 2
        new_h = (new_h // 2) * 2

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Kareye Tamamlama (Padding):
        # Kalan bosluklari (kenarlari) beyaz renkle doldurarak tam kare haline getirir.
        max_dim = max(new_w, new_h)
        pad_w = (max_dim - new_w) // 2
        pad_h = (max_dim - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 3. Gri Tonlama (Grayscale):
        # Bazi gercek grafikler renkli degil, siyah-beyazdir veya fotokopidir.
        if random.random() < prob:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # 4. Parlaklik ve Kontrast:
        # Tarama kalitesini taklit eder (soluk veya cok koyu taramalar).
        if random.random() < prob:
            brightness_factor = random.uniform(0.90, 1.10)
            contrast_factor = random.uniform(0.85, 1.15)

            img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_factor*10)
            img = np.clip(img, 0, 255).astype(np.uint8)

        # 5. Gurultu Ekleme (Noise):
        # Kagit dokusu, toz veya tarayici sensorundeki karincalanmalari simule eder.
        if random.random() < prob and max_noise > 0:
            noise = np.random.normal(0, random.uniform(5, max_noise), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. JPEG Sikistirma (Compression Artifacts):
        # Internetteki dusuk kaliteli, bloklu/bulanik resimleri taklit eder.
        if random.random() < prob and min_jpeg < 100:
            quality = random.randint(min_jpeg, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)

        # 7. Bulaniklik (Blur): Kamera odaksizligi veya hizli tarama
        if random.random() < prob and blur_lvl > 0:
            k = random.choice([x for x in range(3, blur_lvl + 2, 2)])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 8. Golgelenme (Shadow/Vignette): Cep telefonu cekimi dengesiz aydinlatma
        if shadow_en == 1 and random.random() < prob:
            h_img, w_img = img.shape[:2]
            c_x = random.randint(int(w_img * 0.2), int(w_img * 0.8))
            c_y = random.randint(int(h_img * 0.2), int(h_img * 0.8))
            Y, X = np.ogrid[:h_img, :w_img]
            dist_from_center = np.sqrt((X - c_x)**2 + (Y - c_y)**2)
            max_dist = np.sqrt((w_img/2)**2 + (h_img/2)**2)
            mask = 1.0 - (dist_from_center / max_dist) * random.uniform(0.4, 0.8)
            mask = np.clip(mask, 0, 1)
            mask = np.dstack([mask]*3) if len(img.shape) == 3 else mask
            img = (img * mask).astype(np.uint8)

        return img

    def generate_sample(self, sample_id: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray, List[CurveData], ChartConfig]:
        """
        Tum preprocessing'le tek bir ornek olustur.

        Bu metod, bir egitim orneginin "Dogumundan - Paketlenmesine" kadar olan surecini yonetir.

        Asamalar:
        1. Planlama: Rastgele bir grafik konfigurasyonu (recete) olusturulur.
        2. Cizim (Rendering): Matplotlib ile yuksek cozunurluklu "temiz" grafik cizilir.
        3. Eskitme (Artifacts): Kagit bozulmalari ve tarayici hatalari eklenir.
        4. Isleme (Preprocessing): Dondurme, boyutlandirma ve gurultu ekleme yapilir.
        5. Hedefleme (Ground Truth): Egitim icin gerekli maske ve renkli hedef goruntuleri hazirlanir.
        """

        # 1. Tohumlama (Seeding):
        # Tekrarlanabilirlik icindir. Ayni seed verilirse, her zaman ayni grafigi uretir.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 2. Planlama (Configuration):
        # Rastgele parametrelerle bir grafik taslagi olusturulur.
        config = self.advanced_random_config()

        # Konfigurasyon Dogrulama:
        # Bazen rastgele secimler uyumsuz olabilir, burada duzeltilir.
        if not config.drag_indices or len(config.drag_indices) == 0:
            config.drag_indices = [0]

        if config.n_curves != len(config.drag_indices):
            config.n_curves = len(config.drag_indices)

        # 3. Cizim (Rendering):
        # Temel grafik, hedef boyuttan DAHA YUKSEK cozunurlukte (1200px) cizilir.
        # Neden? Cunku goruntuyu sonradan kucultmek (downsampling), dusuk cozunurlukte cizip buyutmekten daha kalitelidir (anti-aliasing saglar).
        base_size = 1200
        full_img, mask, curves_data = draw_chart_matplotlib(config, W=base_size, H=int(base_size*0.75))

        # Hata Kontrolu:
        # Eger cizim basarisiz olur ve hic egri verisi donmezse hata firlatilir.
        if not curves_data or len(curves_data) == 0:
            raise ValueError(f"Config icin egri olusturulmadi: {config}")

        # 4. Eskitme (Scan Artifacts):
        # Temiz dijital cizime, taranmis kagit hissi verilir (hafif bozulmalar).
        full_img = add_scan_artifacts(full_img, strength=random.uniform(0.8, 1.5), background_style=config.background_type)

        # 5. Isleme (Preprocessing):
        # Veri artirma (augmentation) uygulanir: Dondurme, gurultu, JPEG sikistirma vb.
        # Bu adimda goruntu hedef boyuta (ornegin 1024px) getirilir.
        full_img_processed = self.apply_preprocessing(full_img)

        # 6. Maske Isleme (Mask Resizing):
        # Orijinal maske de yeni goruntu boyutuna (processed_img boyutuna) uyarlanmalidir.
        # DIKKAT: interpolation kullanilmaz veya Nearest Neighbor kullanilir, cunku maske
        # sadece 0 ve 1 (siyah ve beyaz) olmalidir. Gri tonlar istenmez.
        h, w = full_img_processed.shape[:2]
        mask_processed = cv2.resize(mask, (w, h))

        # 7. Renkli Hedef (Color Target):
        # Modelin egitimi veya gorsel dogrulama icin, egrilerin farkli renklerle boyandigi
        # temiz bir "yer gercegi" (ground truth) goruntusu olusturulur.
        # Siyah arka plan uzerinde sadece renkli cizgiler bulunur.
        target_colored = colorize_curves_from_data(
            curves_data, config, w, h,
            show_axes=False, black_background=True
        )

        return full_img_processed, mask_processed, curves_data, config

    def create_splits(self, full_coco: Dict = None):
        """
        Görselleri ve maskeleri fiziksel olarak Train, Valid ve Test klasörlerine taşı (direkt output_base altında).
        Oranlar: %70 Train, %15 Valid, %15 Test
        Ayrıca her klasöre ilgili COCO anotasyonlarını (_annotations.coco) ekle.
        """
        print("\n[SPLIT] Veri seti fiziksel olarak bölünüyor (Doğrudan klasör yapısı)...")

        if full_coco is None:
            coco_path = os.path.join(self.output_base, "annotations.json")
            if not os.path.exists(coco_path):
                print("[HATA] Ana annotations.json bulunamadı!")
                return
            with open(coco_path, 'r', encoding='utf-8') as f:
                full_coco = json.load(f)

        all_images = full_coco.get("images", [])
        total = len(all_images)

        if total == 0:
            print("[UYARI] Hiç görüntü bulunamadı!")
            return

        # Karıştır
        random.seed(42)
        indices = list(range(total))
        random.shuffle(indices)

        # Bölme indeksleri (70/15/15)
        train_cut = int(total * 0.70)
        valid_cut = int(total * 0.85)

        splits_map = {
            "train": indices[:train_cut],
            "valid": indices[train_cut:valid_cut],
            "test": indices[valid_cut:]
        }

        for split_name, split_indices in splits_map.items():
            if not split_indices: continue

            # Klasörleri oluştur (Örn: dataset_production/train)
            split_dir = os.path.join(self.output_base, split_name)
            split_img_dir = split_dir
            split_mask_dir = os.path.join(split_dir, "masks")
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(split_mask_dir, exist_ok=True)

            # Bu split için veri toplama
            subset_coco = {
                "images": [],
                "annotations": [],
                "categories": full_coco.get("categories", [])
            }

            selected_img_ids = set()
            for idx in split_indices:
                img_info = all_images[idx]
                img_name = img_info["file_name"]

                # Dosya yolları
                src_img = os.path.join(self.images_dir, img_name)
                dst_img = os.path.join(split_img_dir, img_name)

                mask_name = img_name.replace("img_", "mask_")
                src_mask = os.path.join(self.masks_dir, mask_name)
                dst_mask = os.path.join(split_mask_dir, mask_name)

                # Dağıtma (Taşıma) İşlemi
                if os.path.exists(src_img):
                    shutil.move(src_img, dst_img)
                if os.path.exists(src_mask):
                    shutil.move(src_mask, dst_mask)

                subset_coco["images"].append(img_info)
                selected_img_ids.add(img_info["id"])

            # Anotasyonları filtrele
            for ann in full_coco.get("annotations", []):
                if ann["image_id"] in selected_img_ids:
                    subset_coco["annotations"].append(ann)

            # _annotations.coco.json olarak kaydet
            split_coco_path = os.path.join(split_dir, "_annotations.coco.json")
            with open(split_coco_path, 'w', encoding='utf-8') as f:
                json.dump(subset_coco, f, indent=2)

            print(f"  -> {split_name}: {len(split_indices)} grafik ve .coco dosyası ({self.output_base}/{split_name}/ altında)")

        # Temizlik
        coco_path = os.path.join(self.output_base, "annotations.json")
        try:
            if os.path.exists(self.images_dir) and not os.listdir(self.images_dir):
                os.rmdir(self.images_dir)
            if os.path.exists(self.masks_dir) and not os.listdir(self.masks_dir):
                os.rmdir(self.masks_dir)
            if os.path.exists(coco_path):
                os.remove(coco_path)
        except Exception as e:
            print(f"[UYARI] Temizlik hatası: {e}")

        print("\n[BAŞARILI] Dataset 'splits/' klasörü altında organize edildi.")

        print("\n[BAŞARILI] Dataset Roboflow yapısına dönüştürüldü.")

def upload_to_drive(dataset_path="dataset_production"):
    """
    Dataset klasörünü Google Drive'a yüklemek için rclone kullanır.

    Önce rclone kurulmalı ve Google Drive remote yapılandırılmalı.
    """
    if not os.path.exists(dataset_path):
        messagebox.showerror("Hata", f"Dataset klasörü bulunamadı: {dataset_path}")
        return False

    # rclone kurulu mu kontrol et
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        msg = ("rclone kurulu değil!\\n\\n"
               "Google Drive'a yüklemek için rclone gerekli.\\n"
               "İndirmek için: https://rclone.org/downloads/\\n\\n"
               "Alternatif: Manuel olarak dataset_production klasörünü Drive'a sürükle-bırak yapın.")
        messagebox.showwarning("rclone Bulunamadı", msg)
        return False

    # Drive remote var mı kontrol et
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    remotes = result.stdout.strip().split('\\n')

    if not any('drive' in r.lower() for r in remotes):
        msg = ("Google Drive remote yapılandırılmamış!\\n\\n"
               "Önce rclone'u yapılandırın:\\n"
               "  1. Komut satırında: rclone config\\n"
               "  2. 'n' (new remote)\\n"
               "  3. Name: gdrive\\n"
               "  4. Storage: Google Drive seçin\\n"
               "  5. Tarayıcıda Google hesabınıza giriş yapın")
        messagebox.showwarning("Drive Yapılandırılmamış", msg)
        return False

    # Upload yap
    drive_remote = [r for r in remotes if 'drive' in r.lower()][0].strip(':')

    response = messagebox.askyesno(
        "Drive'a Yükle",
        f"Dataset '{dataset_path}' Google Drive'a yüklenecek.\\n\\n"
        f"Remote: {drive_remote}\\n"
        f"Hedef: {drive_remote}:dataset_production\\n\\n"
        "Devam edilsin mi?"
    )

    if not response:
        return False

    try:
        # Progress window
        progress_win = tk.Toplevel()
        progress_win.title("Yükleniyor...")
        progress_win.geometry("400x100")
        tk.Label(progress_win, text="Dataset Google Drive'a yükleniyor...", font=('Arial', 10)).pack(pady=10)
        progress_label = tk.Label(progress_win, text="Lütfen bekleyin...", font=('Arial', 9))
        progress_label.pack(pady=5)
        progress_win.update()

        # rclone copy komutu
        cmd = f'rclone copy "{dataset_path}" "{drive_remote}:dataset_production" --progress'
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Output oku
        for line in process.stdout:
            print(line.strip())
            if "Transferred:" in line:
                progress_label.config(text=line.strip()[:60])
                progress_win.update()

        process.wait()
        progress_win.destroy()

        if process.returncode == 0:
            messagebox.showinfo("Başarılı", f"Dataset Google Drive'a yüklendi!\\n\\nKonum: {drive_remote}:dataset_production")
            return True
        else:
            messagebox.showerror("Hata", "Yükleme başarısız oldu.")
            return False

    except Exception as e:
        messagebox.showerror("Hata", f"Yükleme hatası: {str(e)}")
        return False

def simple_zip_upload(dataset_path="dataset_production"):
    """
    Basit yöntem: Dataset'i ZIP'le, kullanıcının manuel yüklemesi için kaydet
    """
    if not os.path.exists(dataset_path):
        messagebox.showerror("Hata", f"Dataset klasörü bulunamadı: {dataset_path}")
        return False

    # ZIP hedef yolu sor
    zip_path = filedialog.asksaveasfilename(
        title="ZIP Dosyasını Kaydet",
        defaultextension=".zip",
        filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        initialfile="dataset_production.zip"
    )

    if not zip_path:
        return False

    try:
        # ZIP oluştur
        import zipfile

        progress_win = tk.Toplevel()
        progress_win.title("ZIP Oluşturuluyor...")
        progress_win.geometry("350x80")
        tk.Label(progress_win, text="Dataset ZIP'leniyor...", font=('Arial', 10)).pack(pady=10)
        progress_label = tk.Label(progress_win, text="", font=('Arial', 9))
        progress_label.pack(pady=5)
        progress_win.update()

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            total_files = sum([len(files) for _, _, files in os.walk(dataset_path)])
            count = 0

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(dataset_path))
                    zipf.write(file_path, arcname)
                    count += 1
                    if count % 10 == 0:
                        progress_label.config(text=f"{count}/{total_files} dosya")
                        progress_win.update()

        progress_win.destroy()

        msg = (f"Dataset ZIP dosyası oluşturuldu!\\n\\n"
               f"Konum: {zip_path}\\n\\n"
               "Şimdi bu dosyayı Google Drive'a manuel olarak yükleyebilirsiniz:\\n"
               "1. drive.google.com'a gidin\\n"
               "2. ZIP dosyasını sürükle-bırak yapın\\n"
               "3. Colab'da ZIP'i açın")

        messagebox.showinfo("Başarılı", msg)

        # ZIP konumunu aç
        if messagebox.askyesno("Klasörü Aç", "ZIP dosyasının bulunduğu klasörü açmak ister misiniz?"):
            os.startfile(os.path.dirname(zip_path))

        return True

    except Exception as e:
        messagebox.showerror("Hata", f"ZIP oluşturma hatası: {str(e)}")
        return False

def test_direct_generation():
    """
    Direct test - bypass multiprocessing completely
    """
    import cv2
    import traceback

    print("Testing direct generation (no multiprocessing)...")
    gen = ProductionDatasetGenerator("test_direct", aug_config={"probability": 0.7})

    print("Calling generate_sample...")
    try:
        full_img, mask_img, curves, config = gen.generate_sample(0, seed=42)
        print(f"✅ Generated: shape={full_img.shape}")

        os.makedirs("test_direct/images", exist_ok=True)
        os.makedirs("test_direct/masks", exist_ok=True)
        cv2.imwrite("test_direct/images/test_0.png", full_img)
        cv2.imwrite("test_direct/masks/test_0.png", mask_img)
        print("✅ Saved successfully")

        print(f"Curves: {len(curves)}")
        print(f"Config: x_range=[{config.x_min}, {config.x_max}]")

    except Exception as e:
        print(f"❌ Failed: {e}")
        traceback.print_exc()

def test_worker_function():
    """
    Quick test to see if worker function works
    """
    import traceback

    print("Creating generator...")
    gen = ProductionDatasetGenerator("test_output")

    print("Testing worker function...")
    try:
        result = gen._generate_sample_worker("test_output", {"probability": 0.7}, 0)
        if result:
            print(f"✅ Worker returned: {type(result)}")
            img_info, annotations = result
            print(f"   Images: {img_info.get('id')}")
            print(f"   Annotations: {len(annotations)}")
        else:
            print("❌ Worker returned None")
    except Exception as e:
        print(f"❌ Worker failed: {e}")
        traceback.print_exc()

def test_splits_function():
    print("Testing create_splits manually...")
    dataset_path = "dataset_production"

    annotations_path = os.path.join(dataset_path, "annotations.json")
    if not os.path.exists(annotations_path):
        print(f"[ERROR] {annotations_path} does not exist. Cannot test splits.")
        return

    with open(annotations_path, 'r') as f:
        coco_dataset = json.load(f)

    all_images = coco_dataset["images"]
    total = len(all_images)

    print(f"Total images: {total}")

    if total == 0:
        print("[WARNING] No images found!")
    else:
        indices = list(range(total))
        random.seed(42)
        random.shuffle(indices)

        train_cut = int(total * 0.70)
        val_cut = int(total * 0.85)

        train_indices = indices[:train_cut]
        val_indices = indices[train_cut:val_cut]
        test_indices = indices[val_cut:]

        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        def create_subset(subset_indices, name):
            subset_data = {
                "images": [],
                "annotations": [],
                "categories": coco_dataset["categories"]
            }

            selected_img_ids = set()
            for idx in subset_indices:
                img_data = all_images[idx]
                subset_data["images"].append(img_data)
                selected_img_ids.add(img_data["id"])

            for ann in coco_dataset["annotations"]:
                if ann["image_id"] in selected_img_ids:
                    subset_data["annotations"].append(ann)

            out_path = os.path.join(dataset_path, f"{name}.json")
            with open(out_path, 'w') as f:
                json.dump(subset_data, f, indent=2)
            print(f"Created: {out_path} ({len(subset_data['images'])} images, {len(subset_data['annotations'])} annotations)")

        create_subset(train_indices, "train")
        create_subset(val_indices, "val")
        create_subset(test_indices, "test")

        print("\\n✅ Split files created successfully!")

if __name__ == "__main__":
    import sys
    import argparse

    # -------------------------------------------------------------------------
    # ANA CALISMA BLOGU (MAIN EXECUTION)
    # -------------------------------------------------------------------------
    # Windows'ta multiprocessing kullanirken bu blok ZORUNLUDUR.
    # Eger bu kontrol olmazsa, her yeni acilan process (isci), kodu bastan calistirip
    # sonsuz donguye (recursive spawn bomb) girer ve bilgisayari kilitler.

    # Argparse ile parametreleri al
    parser = argparse.ArgumentParser(description="Sentetik Veri Uretimi")
    parser.add_argument("n_samples", type=int, nargs='?', default=5000, help="Uretilecek gorsel sayisi")
    parser.add_argument("num_workers", type=int, nargs='?', default=6, help="Worker (is parcacigi) sayisi")

    # Test flags
    parser.add_argument("--test-direct", action="store_true", help="Run direct generation test without multiprocessing")
    parser.add_argument("--test-worker", action="store_true", help="Run worker generation test")
    parser.add_argument("--test-splits", action="store_true", help="Run dataset splitting test")

    # Augmentation parametreleri
    parser.add_argument("--rotation", type=float, default=3.0, help="Maksimum dondurme acisi (+- derece)")
    parser.add_argument("--noise", type=float, default=15.0, help="Maksimum gurultu seviyesi")
    parser.add_argument("--jpeg", type=int, default=70, help="Minimum JPEG kalitesi (0-100)")
    parser.add_argument("--prob", type=float, default=0.7, help="Efektlerin uygulanma olasiligi (0.0 - 1.0)")
    parser.add_argument("--blur", type=int, default=0, help="Maksimum bulaniklik kerneli (orn. 3, 5, 7. 0 = kapali)")
    parser.add_argument("--shadow", type=int, choices=[0, 1], default=0, help="Golge/Aydinlatma sorunu (1=Acik, 0=Kapali)")
    parser.add_argument("--flip", type=int, choices=[0, 1, 2, 3], default=0, help="Cevirme (0: off, 1: H, 2: V, 3: Both)")
    parser.add_argument("--shear", type=float, default=0.0, help="Kaykilma miktari (0.0-0.3)")
    parser.add_argument("--hue", type=int, default=0, help="Hue jitter (0-180)")
    parser.add_argument("--sat", type=int, default=0, help="Saturation jitter (0-255)")
    parser.add_argument("--cutout", type=int, default=0, help="Cutout delik sayisi")
    parser.add_argument("--motion", type=int, default=0, help="Motion blur kernel boyutu")

    args = parser.parse_args()

    if args.test_direct:
        test_direct_generation()
        sys.exit(0)

    if args.test_worker:
        test_worker_function()
        sys.exit(0)

    if args.test_splits:
        test_splits_function()
        sys.exit(0)

    n_samples = args.n_samples
    num_workers = args.num_workers

    # Augmentation konfigurasyonu
    aug_config = {
        "rotation_angle": args.rotation,
        "noise_level": args.noise,
        "jpeg_quality": args.jpeg,
        "probability": args.prob,
        "blur_level": args.blur,
        "shadow_enabled": args.shadow,
        "flip_mode": args.flip,
        "shear_level": args.shear,
        "hsv_jitter": [args.hue, args.sat, 0],
        "cutout_count": args.cutout,
        "motion_blur": args.motion
    }

    print("\n" + "=" * 70)
    print("SENTETIK UCAK GRAFIKLERI VERI SISTEMI URETECI - URETIM")
    print("=" * 70)
    print("Konfigurasyon:")
    print(f"   * Toplam ornekler: {n_samples:,}")
    print(f"   * Isciler: {num_workers}")
    print(f"   * Augmentation: {aug_config}")
    print("   * Cikis: dataset_production/")

    # Zaman Tahmini
    print("\nZaman tahmini...")
    if n_samples <= 100:
        est_time = "< 1 dk"
    elif n_samples <= 500:
        est_time = "2-5 dk"
    elif n_samples <= 2000:
        est_time = "5-10 dk"
    else:
        est_time = "15-30 dk"
    print(f"   * Tahmini zaman: {est_time}")
    print("=" * 70 + "\n")

    # Veri uretimini baslat - aug_config ile
    generator = ProductionDatasetGenerator(output_base="dataset_production", aug_config=aug_config)

    # Ana uretim fonksiyonunu cagir
    generator.generate_dataset(n_samples=n_samples, num_workers=num_workers)

    # Bitis Raporu
    print("\n" + "=" * 70)
    print("VERI SISTEMI URETIMI TAMAMLANDI!")
    print("=" * 70)
    print("Sonraki adimlar:")
    print("   1. 'dataset_production/' klasorunu kontrol et")
    print("   2. Roboflow'a yukle veya model egitimine basla")
    print("   3. On isleme yapilandir (zaten uygulandi):")
    print("      * Otomatik dondurme (+-3 derece)")
    print("      * Yeniden boyutlandirma (fit modu)")
    print("      * Gri tonlama (Grayscale)")
    print("      * Gurultu + JPEG bozulmalari")
    print("      * Bulaniklik (Blur): " + ("Acik" if aug_config['blur_level']>0 else "Kapali"))
    print("      * Golge (Shadow): " + ("Acik" if aug_config['shadow_enabled']==1 else "Kapali"))
    print("=" * 70 + "\n")
