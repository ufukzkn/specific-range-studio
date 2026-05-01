"""
Curve Segmentation - Model Inference & Visualization
====================================================
Bu script, egitilmis bir Derin Ogrenme modelini kullanarak grafiklerdeki egrileri (curves) segmente eder.

NASIL KULLANILIR? (TUTORIAL)
------------------------------
Asagidaki komutlari terminale (cmd veya PowerShell) yazarak calistirabilirsiniz.

1. KLASORDEKI TUM GORUNTULERI ISLEME:
   Eger elinizde bir klasor dolusu resim varsa ve hepsini islemek istiyorsaniz:

   python kontrol.py --model checkpoints/best_model.pth --input_dir "C:/Resimlerim/Test" --output_dir "Sonuclar"

   - --model       : Egitilmis model dosyanizin yolu (Zorunlu)
   - --input_dir   : Icinde grafik resimlerinin oldugu klasor (Zorunlu)
   - --output_dir  : Sonuclarin kaydedilecegi klasor (Otomatik olusturulur)

2. PDF DOSYASINDAN GRAFIK CIKARMA VE ISLEME:
   Eger bir PDF dosyasindaki belirli sayfalardaki grafikleri hem cikarip hem islemek istiyorsaniz:

   python kontrol.py --model checkpoints/best_model.pth --pdf "C:/Belgelerim/kitap.pdf" --pages 10-50 --output_dir "PDF_Sonuclari"

   - --pdf         : PDF dosyasinin tam yolu
   - --pages       : Islenecek sayfa araligi (Orn: "10-50" veya "5" tek sayfa)

3. SADECE PDF'DEN GRAFIK CIKARMA (SEGMENTASYON YAPMA):
   Modeli calistirmadan sadece grafikleri ayiklamak isterseniz:

   python kontrol.py --model checkpoints/best_model.pth --pdf "C:/Belgelerim/kitap.pdf" --pages 10-50 --output_dir "PDF_Ciktilari" --extract_only

   - --extract_only: Sadece grafikleri cikarir, segmentasyon yapmaz.

4. EGRILERIN VERISINI EXCEL'E AKTARMA (DATA DIGITIZATION):
   Segmentasyon sonucunda olusan maskelerden egri koordinatlarini cekmek isterseniz:

   python kontrol.py --model checkpoints/best_model.pth --input_dir "Girdiler" --output_dir "Ciktilar" --extract_curve_data

   - --extract_curve_data : Egri verilerini sayisal olarak cikarir.
   - --curve_data_output  : Excel dosyasinin adi (Varsayilan: curve_data.xlsx)

NOT: Dosya yollarinda bosluk varsa mutlaka cift tirnak (") kullanin. Orn: "D:/My Documents/Test"
"""

import os
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] PDF destegi yuklu degil. 'pip install pdf2image pillow' calistirin")

try:
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("[WARNING] OCR destegi yuklu degil. 'pip install pytesseract' calistirin")

try:
    import pandas as pd
    PANDAS_SUPPORT = True
except ImportError:
    PANDAS_SUPPORT = False
    print("[WARNING] Pandas yuklu degil. 'pip install pandas openpyxl' calistirin")

# ============================================================================
# U-NET MODEL MIMARISI (Standart PyTorch Implementasyonu)
# ============================================================================
# Train_unet.py dosyasina bagimliligi kaldirmak icin sinif buraya eklendi.

import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2 Islemi

    Her bir U-Net basamaginda iki kere konvolusyon uygulanir.
    Bu sinif bu tekrarlayan islemi paketler.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder (Asagi Indirgeme) Blogu
    Yapilan Islem: MaxPool2d (Boyut Kucultme) + DoubleConv (Ozellik Cikarma)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder (Yukari Genisletme) Blogu
    Yapilan Islem: Upsampling (Buyutme) + Skip Connection Birlestirme + DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Bilinear interpolation ile buyutme (Daha hizli)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Boyut uyusmazligi varsa padding ile duzelt
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # Skip connection birlestirme
        return self.conv(x)


class OutConv(nn.Module):
    """Cikis Katmani: Ozellikleri sonuca donusturur"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Tam U-Net Modeli Mimarisi.
    Encoder-Decoder yapisi ve atlama baglantilari (skip connections) ile calisir.
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- ENCODER ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # --- DECODER ---
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

UNET_AVAILABLE = True

# Eger extract_curve_data modulu varsa import et
try:
    from extract_curve_data import CurveExtractor, ExcelExporter, parse_filename_metadata
    CURVE_EXTRACTION_AVAILABLE = True
except ImportError:
    CURVE_EXTRACTION_AVAILABLE = False


# ============================================================================
# RESNET ENCODER-DECODER MODEL (Demo_Model.pth icin)
# ============================================================================

class ResNetEncoderDecoder(nn.Module):
    """ResNet encoder + Custom decoder mimarisi"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ResNet50 encoder kisimlarini olustur
        self.encoder = nn.Sequential(
            # conv1 (7x7 -> 64)
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # layer1 (64 -> 256)
            nn.Sequential(
                ResidualBlock(64, 64, stride=1),
                ResidualBlock(64, 64, stride=1),
                ResidualBlock(64, 256, stride=1),
            ),

            # layer2 (256 -> 512)
            nn.Sequential(
                ResidualBlock(256, 128, stride=2),
                ResidualBlock(128, 128, stride=1),
                ResidualBlock(128, 128, stride=1),
                ResidualBlock(128, 512, stride=1),
            ),

            # layer3 (512 -> 1024)
            nn.Sequential(
                ResidualBlock(512, 256, stride=2),
                ResidualBlock(256, 256, stride=1),
                *[ResidualBlock(256, 256, stride=1) for _ in range(4)],
                ResidualBlock(256, 1024, stride=1),
            ),

            # layer4 (1024 -> 2048)
            nn.Sequential(
                ResidualBlock(1024, 512, stride=2),
                ResidualBlock(512, 512, stride=1),
                ResidualBlock(512, 2048, stride=1),
            ),
        )

        # Custom decoder
        self.decoder = nn.Sequential(
            # 2048 -> 1024 (sisir + conv)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # 1024 -> 512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 512 -> 256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256 -> 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 -> out_channels
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out


class ResidualBlock(nn.Module):
    """ResNet residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


# ============================================================================
# PDF GRAFIK CIKARICI (Grafik_Crop_Tag.py Algoritmasi)
# ============================================================================

class PDFGraphicExtractor:
    """PDF dosyasindan grafikleri cikarp, OCR ile bilgi tac eden sinif"""

    def __init__(self, poppler_path: Optional[str] = None, tesseract_path: Optional[str] = None):
        """
        Args:
            poppler_path: Poppler-utils path (Windows icin)
            tesseract_path: Tesseract OCR path
        """
        if not PDF_SUPPORT:
            raise ImportError("PDF support gerekli: pip install pdf2image pillow")

        self.poppler_path = poppler_path
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    @staticmethod
    def crop_smart_area_v9(pil_image):
        """V9 Smart Crop - Grafiklerin cevresindeki ana alani bul"""
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        grid_rect = None
        img_h, img_w = img_cv.shape[:2]

        if contours is not None:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > (img_h * img_w * 0.1) and area < (img_h * img_w * 0.95):
                    if area > max_area:
                        max_area = area
                        grid_rect = (x, y, w, h)

        if grid_rect:
            gx, gy, gw, gh = grid_rect
            crop_x = max(0, gx - 160)
            crop_y = max(0, gy - 30)
            crop_w = min(img_w, (gx + gw + 400))
            crop_h = min(img_h, (gy + gh + 350))
            return pil_image.crop((crop_x, crop_y, crop_w, crop_h))
        else:
            return pil_image

    @staticmethod
    def preprocess_for_ocr(pil_image):
        """OCR icin goruntuyu hazirla"""
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        img_cv = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, img_thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(img_thresh)

    @staticmethod
    def clean_number(text):
        """Metinden sadece numaralari cikar"""
        return re.sub(r'\D', '', text)

    @staticmethod
    def extract_chart_info_v9(processed_image):
        """
        OCR'dan bilgi cikar: Motor Tipi, Irtifa, Agirlik
        Format: "{engine}-{altitude}-{weight}lb.png"
        """
        if not OCR_SUPPORT:
            print("[WARNING] OCR disabled - pytesseract yuklu degil")
            return None

        try:
            text = pytesseract.image_to_string(processed_image, config='--psm 6')
        except Exception as e:
            print(f"[WARNING] OCR basarisiz: {e}")
            return None

        text = text.replace('\n', ' ').strip().lower()

        # 1. Motor Tipi
        if "single" in text:
            engine_code = "1"
        else:
            engine_code = "2"

        altitude = None
        weight = None

        # 2. Irtifa & Agirlik
        # A) SEA LEVEL KONTROLU
        sea_level_pattern = r'sea\s*[l1i]ev[el1ij]|s\.l\.'

        if re.search(sea_level_pattern, text):
            altitude = "0"

            # Agirlik (Sea Level)
            weight_sl_pattern = r'-\s*(\d[\d,\.\s]*)\s*(?:pounds|lbs|lb)'
            w_match = re.search(weight_sl_pattern, text)

            if w_match:
                weight = PDFGraphicExtractor.clean_number(w_match.group(1))
            else:
                w_fallback = re.search(r'(\d[\d,\.\s]*)\s*(?:pounds|lbs|lb)', text)
                if w_fallback:
                    weight = PDFGraphicExtractor.clean_number(w_fallback.group(1))
        else:
            # B) Normal FEET Kontrolu
            alt_match = re.search(r'(\d[\d,\.\s]*)\s*(?:feet|ft)', text)
            if alt_match:
                altitude = PDFGraphicExtractor.clean_number(alt_match.group(1))

                # Agirlik (Normal)
                weight_pattern = r'feet.*?-\s*(\d[\d,\.\s]*)\s*(?:pounds|lbs|lb)'
                w_match = re.search(weight_pattern, text)
                if w_match:
                    weight = PDFGraphicExtractor.clean_number(w_match.group(1))
                else:
                    w_fallback = re.search(r'feet.*?(\d[\d,\.\s]*)\s*(?:pounds|lbs|lb)', text)
                    if w_fallback:
                        weight = PDFGraphicExtractor.clean_number(w_fallback.group(1))
            else:
                return None

        if not weight:
            return None

        return f"{engine_code}-{altitude}-{weight}lb.png"

    def extract_charts_from_pdf(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        output_dir: str = "extracted_charts"
    ) -> list:
        """
        PDF'den belirtilen sayfalar arasindaki grafikleri cikar

        Args:
            pdf_path: PDF dosyasi yolu
            start_page: Baslangic sayfasi (1-indexed)
            end_page: Bitis sayfasi (inclusive)
            output_dir: Grafiklerin kaydedilecegi klasor

        Returns:
            Kaydedilen dosyalarin listesi
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF bulunamadi: {pdf_path}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[PDF Processing] {pdf_path}")
        print(f"[Pages] {start_page} - {end_page}")
        print(f"[Output] {output_dir}\n")

        # PDF'yi oku
        try:
            images = convert_from_path(
                pdf_path,
                first_page=start_page,
                last_page=end_page,
                dpi=300,
                poppler_path=self.poppler_path
            )
        except Exception as e:
            print(f"[ERROR] PDF okuma basarisiz: {e}")
            return []

        saved_files = []
        count_ok = 0
        count_manual = 0

        print(f"[Processing] {len(images)} sayfa isleniyor...")

        for i, img_full in enumerate(tqdm(images, desc="Extracting charts")):
            current_page_num = start_page + i

            try:
                # 1. Crop
                final_cropped_image = self.crop_smart_area_v9(img_full)

                # 2. OCR Hazirligi
                w, h = final_cropped_image.size
                bottom_area = final_cropped_image.crop((0, h * 0.75, w, h))
                processed_bottom = self.preprocess_for_ocr(bottom_area)

                # 3. Bilgi Cikarma
                filename = self.extract_chart_info_v9(processed_bottom)

                if filename:
                    save_path = os.path.join(output_dir, filename)

                    # Dosya zaten var ise, suffix ekle
                    if os.path.exists(save_path):
                        filename = filename.replace('.png', '_2.png')
                        save_path = os.path.join(output_dir, filename)

                    final_cropped_image.save(save_path, "PNG")
                    saved_files.append(save_path)
                    print(f"  [OK] Sayfa {current_page_num} → {filename}")
                    count_ok += 1
                else:
                    # OCR basarisiz - manuel dosya olustur
                    filename = f"MANUEL_Sayfa_{current_page_num}.png"
                    save_path = os.path.join(output_dir, filename)
                    final_cropped_image.save(save_path, "PNG")
                    saved_files.append(save_path)
                    print(f"  [WARN] Sayfa {current_page_num} → {filename} (OCR basarisiz)")
                    count_manual += 1

            except Exception as e:
                print(f"  [ERROR] Sayfa {current_page_num}: {e}")

        print(f"\n{'='*70}")
        print(f"[OK] Cikarma Tamamlandi!")
        print(f"  - Basarili: {count_ok}")
        print(f"  - Manuel: {count_manual}")
        print(f"  - Toplam: {len(saved_files)}")
        print(f"  - Klasor: {output_dir}")
        print(f"{'='*70}\n")

        return saved_files


class CurveSegmentationInference:
    """U-Net veya ResNet ile curve segmentation yapan inference sinifi"""
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Egitilmis model dosyasi
            device: torch device (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Checkpoint'i yukle ve mimarisini algila
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"[OK] Checkpoint yuklendi: {model_path}")
        print(f"[OK] Device: {self.device}")

        # Checkpoint keys'i kontrol et - UNet mi ResNet mi?
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Keys'ten mimarisini algila
        keys = list(state_dict.keys())

        # UNet yapisini kontrol et (inc.double_conv...)
        if any('inc.double_conv' in k for k in keys):
            print("[*] UNet mimarisi algilandi...")
            if UNET_AVAILABLE:
                self.model = UNet(n_channels=1, n_classes=1)
            else:
                raise RuntimeError("UNet mimarisine ihtiyac var ama train_unet.py yuklu degil")

        # ResNet / EfficientNet yapisini kontrol et (encoder.layer veya encoder.blocks...)
        elif any('encoder.' in k or 'decoder.block' in k for k in keys):
            if any('segmentation_head' in k for k in keys):
                print("[*] segmentation_models_pytorch (smp) mimarisi algilandi...")
                try:
                    import segmentation_models_pytorch as smp
                    # Once efficientnet-b3 (yeni model) dene, basarisizsa resnet34 (eski model) dene
                    encoder_candidates = ['efficientnet-b3', 'resnet34']
                    loaded = False
                    for enc in encoder_candidates:
                        try:
                            candidate_model = smp.Unet(
                                encoder_name=enc,
                                encoder_weights=None,
                                in_channels=3,
                                classes=1
                            )
                            # Uyumluluğu kontrol et (kaba dogrulama: anahtar sayisi)
                            candidate_keys = set(candidate_model.state_dict().keys())
                            state_keys = set(state_dict.keys())
                            overlap = len(candidate_keys & state_keys) / max(len(state_keys), 1)
                            if overlap > 0.8:  # %80'den fazla anahtar eslesiyor ise dogru mimari
                                self.model = candidate_model
                                self._is_smp = True
                                print(f"[OK] Encoder algilandi: {enc} (uyum: {overlap:.1%})")
                                loaded = True
                                break
                        except Exception:
                            continue
                    if not loaded:
                        print("[WARNING] Encoder algilanamadi, resnet34 olarak yukleniyor...")
                        self.model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
                        self._is_smp = True
                except ImportError:
                    print("[WARNING] segmentation_models_pytorch yuklu degil!")
                    self.model = ResNetEncoderDecoder(in_channels=1, out_channels=1)
                    self._is_smp = False
            else:
                print("[*] ResNet encoder-decoder mimarisi algilandi...")
                self.model = ResNetEncoderDecoder(in_channels=1, out_channels=1)
                self._is_smp = False

        else:
            print(f"[WARNING] Model mimarisi bilinmiyor. Keys ornegi: {keys[:5]}")
            print("[*] ResNet encoder-decoder olarak denenecek...")
            self.model = ResNetEncoderDecoder(in_channels=1, out_channels=1)
            self._is_smp = False

        # State dict'i yukle
        try:
            self.model.load_state_dict(state_dict)
            print("[OK] Model weights basariyla yuklendi")
        except RuntimeError as e:
            print(f"[WARNING] State dict loading hatasi: {e}")
            print("[*] Partial loading deneniyor...")
            # Uyusmayan katmanlari goz ardi et
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"[OK] Partial weights yuklendi ({len(pretrained_dict)}/{len(state_dict)} katman)")

        self.model = self.model.to(self.device)
        self.model.eval()

    def segment(
        self,
        image_path: str,
        input_size: Tuple[int, int] = (640, 640),
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bir goruntuyu segmente et

        Args:
            image_path: Grafik goruntu dosyasi
            input_size: Model input boyutu (H, W)
            threshold: Binary mask esigi (0-1)

        Returns:
            (original_image, segmentation_mask, confidence_map)
        """
        # Goruntuyu renkli yukle (Training asamasinda RGB kullanildigi icin onemli)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Goruntu yuklenemedi: {image_path}")

        original_shape = image_bgr.shape[:2]
        original_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # Geri dondurmek icin grayscale

        # Resize icin RGB'ye donustur (model input boyutuna)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, input_size)

        if getattr(self, '_is_smp', False):
            # Model 3 kanalli egitilmis (smp resnet34 in_channels=3)
            image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image_tensor = (image_tensor.to(self.device) - mean) / std
        else:
            # SMP degilse tek kanalli model olabilir, grayscale'e cevirip resize edelim
            gray_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            image_tensor = torch.from_numpy(gray_resized).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            confidence_map = torch.sigmoid(logits)  # Probability map
            confidence_map = confidence_map.squeeze().cpu().numpy()

        # Binary segmentation
        segmentation_mask = (confidence_map > threshold).astype(np.uint8) * 255

        # Orijinal boyuta resize et
        segmentation_mask = cv2.resize(
            segmentation_mask,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Cizgilerin kopuk olmasini engellemek ve ince ayar yapmak icin Morfolojik islemler
        try:
            from skimage.morphology import skeletonize
            # 1. Once birbirine cok yakin olan/kopuk cizgileri birlestirmek icin Closing (Dilate + Erode) uygula
            # NOT: Birden fazla egrinin birlesmesini onlemek icin kernel boyutu kucultuldu.
            kernel_close = np.ones((3, 3), np.uint8)
            segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel_close)

            # 2. Birlestirilmis kalin maskeyi iskeletlestir (tam ortadan tek piksel cizgi bul)
            binary_mask = (segmentation_mask > 127)
            skeleton = skeletonize(binary_mask)
            segmentation_mask = (skeleton * 255).astype(np.uint8)

            # 3. İskeleti gorunur kilmak/kalinlastirmak icin kucuk bir dilation uygula
            # NOT: Egrilerin birbirine dokunmamasi icin iterasyon sayisi 2'den 1'e dusuruldu.
            kernel_dilate = np.ones((3, 3), np.uint8)
            segmentation_mask = cv2.dilate(segmentation_mask, kernel_dilate, iterations=1)
        except ImportError:
            print("[WARNING] skimage yuklu degil, iskeletizasyon (thinning) yapilamadi. Lutfen 'pip install scikit-image' yapin.")

        confidence_map = cv2.resize(
            confidence_map,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        return original_image, segmentation_mask, confidence_map

    def visualize(
        self,
        image_path: str,
        threshold: float = 0.5,
        save_path: str = None
    ):
        """
        Segmentation sonuclarini gorsellestir

        Args:
            image_path: Grafik dosyasi
            threshold: Binary mask esigi
            save_path: Kaydedilecek dosya yolu
        """
        original, mask, confidence = self.segment(image_path, threshold=threshold)

        # Figur olustur
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Curve Segmentation Results\n{Path(image_path).name}', fontsize=16)

        # 1. Orijinal goruntu
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 2. Confidence map (heatmap)
        im = axes[0, 1].imshow(confidence, cmap='hot')
        axes[0, 1].set_title('Confidence Map')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])

        # 3. Binary mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title(f'Binary Segmentation (threshold={threshold})')
        axes[0, 2].axis('off')

        # 4. Overlay (original + mask)
        overlay = original.copy().astype(float)
        overlay[mask > 127] = overlay[mask > 127] * 0.7 + 255 * 0.3  # Hafif mavi overlay
        axes[1, 0].imshow(overlay.astype(np.uint8), cmap='gray')
        axes[1, 0].set_title('Original + Segmentation Overlay')
        axes[1, 0].axis('off')

        # 5. Renkli mask (curves=kirmizi)
        mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_colored[mask > 127] = [0, 0, 255]  # Kirmizi
        axes[1, 1].imshow(mask_colored)
        axes[1, 1].set_title('Curves (Red)')
        axes[1, 1].axis('off')

        # 6. Combined (original + colored mask)
        combined = original.copy()
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        combined[mask > 127] = [0, 0, 200]  # Curve bolgeleri koyu kirmizi
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title('Original + Curves Overlay')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Gorsellestirme kaydedildi: {save_path}")

        plt.show()

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        threshold: float = 0.5,
        save_confidence: bool = True
    ):
        """
        Klasordeki tum goruntuleri isle ve sonuclari alt klasörlere dagit.
        """
        output_path = Path(output_dir)

        # Alt klasorleri olustur
        seg_dir = output_path / "segmentation"
        overlay_dir = output_path / "overlay"

        for d in [seg_dir, overlay_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Goruntu dosyalarini bul
        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not image_files:
            print(f"[ERROR] {input_dir} klasorunde goruntu bulunamadi")
            return

        print(f"\n[Processing] {len(image_files)} goruntu isleniyor...\n")

        # Her goruntu icin
        for image_file in tqdm(image_files, desc="Segmenting"):
            image_path = os.path.join(input_dir, image_file)

            try:
                original, mask, confidence = self.segment(image_path, threshold=threshold)

                # Dosya ismini hazirla (orijinal ismi koru)
                base_name = image_file

                # Segmentation mask
                cv2.imwrite(str(seg_dir / base_name), mask)

                # Overlay (Renkli highlight: Orijinal resim üzerine kirmizi maske)
                overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                # Maskenin oldugu yerleri parlak kirmizi yap
                overlay[mask > 127] = [0, 0, 255]

                # Hafif seffaflik isterseniz addWeighted kullanilabilir:
                # colored_mask = np.zeros_like(overlay)
                # colored_mask[mask > 127] = [0, 0, 255]
                # overlay = cv2.addWeighted(overlay, 0.8, colored_mask, 0.5, 0)

                cv2.imwrite(str(overlay_dir / base_name), overlay)

            except Exception as e:
                print(f"[ERROR] {image_file} islenirken hata: {e}")

        print(f"\n[OK] Tum sonuclar kaydedildi: {output_dir}")
        print(f"    - {output_dir}/segmentation/ (binary mask)")
        print(f"    - {output_dir}/overlay/ (original + segmentation highlight)")


def main():
    """
    Ana calisma blogu.
    Komut satiri argumanlarini (CLI arguments) isler ve ilgili modu calistirir.
    """
    parser = argparse.ArgumentParser(
        description='Curve Segmentation Inference (Toplu Islem)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  # Klasordeki tum goruntuleri isle
  python kontrol.py --model checkpoints/best_model.pth --input_dir test_images/ --output_dir results/

  # PDF'den grafikleri cikar ve segmente et
  python kontrol.py --model checkpoints/best_model.pth --pdf file.pdf --pages 10-50 --output_dir results/
        """
    )

    # Zorunlu Parametreler
    parser.add_argument('--model', type=str, required=True,
                        help='Egitilmis Model Dosyasi (.pth)')

    # Mod Secenekleri (Biri zorunlu olmali)
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Girdi Klasoru (Icindeki tum resimler islenir)')
    parser.add_argument('--pdf', type=str, default=None,
                        help='PDF Dosyasi (Grafikleri ayiklamak icin)')

    # Opsiyonel Parametreler
    parser.add_argument('--pages', type=str, default=None,
                        help='PDF Sayfa Araligi (Orn: 10-50)')
    parser.add_argument('--output_dir', type=str, default='segmentation_results',
                        help='Sonuclarin kaydedilecegi klasor')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary Maske Esigi (0.0 - 1.0 arasi, Varsayilan: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Calisma Birimi: "cuda" veya "cpu"')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Model Giris Boyutu (Varsayilan: 256)')
    parser.add_argument('--no_confidence', action='store_true',
                        help='Guven haritalarini (Heatmap) kaydetme (Disk tasarrufu)')

    # PDF Ayarlari (Windows)
    parser.add_argument('--poppler_path', type=str, default=None,
                        help='Poppler bin yolu (Windows icin PDF destegi)')
    parser.add_argument('--tesseract_path', type=str, default=None,
                        help='Tesseract OCR bin yolu')

    # Ekstra Modlar
    parser.add_argument('--extract_only', action='store_true',
                        help='Sadece grafikleri PDF\'den cikar, segmentasyon yapma')
    parser.add_argument('--extract_curve_data', action='store_true',
                        help='Segmentasyon sonrasi egri verilerini (sayisal) Excel\'e aktar')
    parser.add_argument('--curve_data_output', type=str, default='curve_data.xlsx',
                        help='Excel dosyasi adi')

    args = parser.parse_args()

    # --- Validasyon Kontrolleri ---
    if not os.path.exists(args.model):
        print(f"[HATA] Model dosyasi bulunamadi: {args.model}")
        return

    # Kullanici ne yapmak istiyor? (Klasor mu PDF mi?)
    mode_count = sum([bool(args.input_dir), bool(args.pdf)])
    if mode_count == 0:
        print("[HATA] Lutfen bir kaynak belirtin: --input_dir veya --pdf")
        return
    if mode_count > 1:
        print("[HATA] Sadece BIR kaynak secebilirsiniz: --input_dir veya --pdf")
        return

    # Inference Motorunu Baslat
    # Model sadece bir kere yuklenir, tum resimler icin kullanilir.
    # input_size parametresini Tuple formatina cevir (256, 256)
    try:
        inferencer = CurveSegmentationInference(args.model, device=args.device)
    except Exception as e:
        print(f"[KRITIK HATA] Model yuklenirken sorun olustu: {e}")
        return

    print(f"\n{'='*70}")
    print(f"CURVE SEGMENTATION (TOPLU ISLEM MODU)")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------
    # SENARYO 1: PDF MODU
    # -----------------------------------------------------------
    if args.pdf:
        if not PDF_SUPPORT:
            print("[HATA] PDF destegi yuklu degil!")
            print("Lutfen kurun: pip install pdf2image pillow pytesseract")
            return

        if not args.pages:
            print("[HATA] Lutfen sayfa araligini belirtin: --pages 10-50")
            return

        try:
            start_page, end_page = map(int, args.pages.split('-'))
        except ValueError:
            print("[HATA] Sayfa formati hatali! Dogru format: --pages 10-50")
            return

        # Ekstraktor (Ayiklayici) Hazirla
        extractor = PDFGraphicExtractor(
            poppler_path=args.poppler_path,
            tesseract_path=args.tesseract_path
        )

        # Grafikleri Once Gecici Bir Klasore Cikar
        extracted_dir = f"{args.output_dir}_extracted"
        extracted_files = extractor.extract_charts_from_pdf(
            args.pdf,
            start_page,
            end_page,
            output_dir=extracted_dir
        )

        # Eger kullanici "Sadece Cikar" dediyse burada bitir
        if args.extract_only:
            print(f"\n[✓] Sadece cikarma modu devrede - Segmentasyon yapilmadi.")
            print(f"[✓] Grafikleri buradan bulabilirsiniz: {extracted_dir}/")
            return

        # Cikarilan grafikleri segmente et
        if extracted_files:
            print(f"\n[Isleniyor] {len(extracted_files)} adet grafik segmente ediliyor...")
            inferencer.process_batch(
                extracted_dir,
                args.output_dir,
                threshold=args.threshold,
                save_confidence=not args.no_confidence
            )

    # -----------------------------------------------------------
    # SENARYO 2: KLASOR MODU (Batch)
    # -----------------------------------------------------------
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[HATA] Klasor bulunamadi: {args.input_dir}")
            return

        inferencer.process_batch(
            args.input_dir,
            args.output_dir,
            threshold=args.threshold,
            save_confidence=not args.no_confidence
        )

    print(f"\n{'='*70}")
    print("[BASARILI] Islem tamamlandi!")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------
    # OPSIYONEL: CURVE DATA EXTRACTION (Veri Sayisallastirma)
    # -----------------------------------------------------------
    if args.extract_curve_data:
        if not CURVE_EXTRACTION_AVAILABLE:
            print("[UYARI] Curve extraction modulu (extract_curve_data.py) bulunamadi.")
            return

        if not PANDAS_SUPPORT:
            print("[UYARI] Excel kaydi icin Pandas gerekli: pip install pandas openpyxl")
            return

        print(f"\n{'='*70}")
        print("EGRI VERISI CIKARIMI (DATA DIGITIZATION)")
        print(f"{'='*70}\n")

        try:
            # Data Extractor Baslat
            extractor = CurveExtractor(tesseract_path=args.tesseract_path)
            all_data = []

            # Olusturulan maskeleri bul
            mask_files = sorted([
                f for f in os.listdir(args.output_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_extracted' not in f
            ])

            # Eğer segmentation klasörü varsa oradan devam et
            seg_path = Path(args.output_dir) / "segmentation"
            if seg_path.exists():
                mask_files = sorted([f.name for f in seg_path.iterdir() if f.suffix.lower() == '.png'])
                mask_dir = str(seg_path)
            else:
                mask_dir = args.output_dir

            if not mask_files:
                print(f"[UYARI] Islenecek maske bulunamadi: {mask_dir}")
                return

            print(f"[Isleniyor] {len(mask_files)} adet curve'in verisi sayisallastiriliyor...\n")

            # Orijinal grafiklerin konumunu belirle (OCR icin gerekli olabilir)
            if args.pdf:
                original_graphics_dir = f"{args.output_dir}_extracted"
            else:
                original_graphics_dir = args.input_dir

            overlay_dir = Path(args.output_dir) / "overlay"

            for mask_file in tqdm(mask_files, desc="Veri Cikarma"):
                meta = parse_filename_metadata(mask_file)
                if not meta: continue

                mask_path = os.path.join(mask_dir, mask_file)
                base_name = Path(mask_file).stem

                # Orijinal dosya yolunu bulmaya calis
                original_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    possible_path = os.path.join(original_graphics_dir, base_name + ext)
                    if os.path.exists(possible_path):
                        original_path = possible_path
                        break

                if not original_path:
                    continue

                overlay_path = str(overlay_dir / mask_file) if overlay_dir.exists() else None

                try:
                    # Egriyi veriye cevir
                    curves = extractor.extract(
                        mask_path=mask_path,
                        original_path=original_path,
                        overlay_img_path=overlay_path
                    )

                    if curves:
                        all_data.append({
                            'altitude': meta['altitude'],
                            'weight': meta['weight'],
                            'engine': meta['engine'],
                            'curves': curves
                        })
                except Exception as e:
                    print(f"[Hata] {mask_file}: {e}")

            if all_data:
                print(f"\n[Kaydediliyor] Excel dosyasi olusturuluyor: {args.curve_data_output}")
                ExcelExporter.export(
                    all_data,
                    args.curve_data_output
                )
                print(f"[✓] Veriler basariyla kaydedildi: {args.curve_data_output}")
            else:
                print("[UYARI] Hic curve verisi cikarilamadi.")

        except Exception as e:
            print(f"[KRITIK HATA] Veri cikarma sureci basarisiz: {e}")
            import traceback
            traceback.print_exc()

# Programin Ana Giris Noktasi
if __name__ == '__main__':
    main()
