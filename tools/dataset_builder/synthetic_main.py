# synthetic_v5.py
"""
Sentez ucak grafikleri uretici - V5
Gercekci oklar icin matplotlib kullanir (orijinal kod gibi).
Ince egriler, uygun ok stilleri.
"""

# IMPORTS (Kutuphaneler)
# -----------------------------------------------------------------------------
import io                       # Bellekte dosya islemleri (resimleri diske kaydetmeden islemek icin)
import math                     # Matematiksel islemler (sinus, kosinus vb.)
import random                   # Rastgelelik (egri sekilleri, gurultu vb.)
import numpy as np              # Sayisal matris islemleri (goruntuler matris olarak tutulur)
from PIL import Image, ImageEnhance, ImageFilter # Goruntu isleme (parlaklik, kontrast)
import matplotlib.pyplot as plt # Grafik cizimi icin ana kutuphane (Plotting)
import matplotlib.patches as mpatches # Sekiller cizmek icin (oklar vb.)
from typing import Tuple, List, Optional, Dict # Tip ipuclari (kodun okunabilirligi icin)
from dataclasses import dataclass # Sinif tanimlarini kisaltmak icin
import cv2                      # OpenCV: Bilgisayarli goru ve goruntu isleme kutuphanesi


@dataclass
class ChartConfig:
    """
    Grafik olusturmak icin 'Recete' (Konfigurasyon).
    Bu sinif, cizilecek grafigin tum ozelliklerini tutar.
    """
    # 1. Eksen Araliklari (Axis Ranges):
    # Grafigin X ve Y eksenlerinin minimum ve maksimum degerleri.
    # Bu degerler, gercek ucak grafiklerinden (Performance Charts) alinmistir.
    x_min: float = 0.30
    x_max: float = 1.00
    y_min: float = 0.04
    y_max: float = 0.15

    # 2. Egri Ayarlari (Curve Settings):
    n_curves: int = 8           # Toplam egri sayisi
    curve_type: str = 'peaked'  # Egri sekli ('peaked': tepecikli, 'rising': yukselen, vb.)
    curve_lw: float = 0.6       # Cizgi kalinligi (float). Gercekci olmasi icin ince tutulur.

    # 3. Surukleme Indeksleri (Drag Indices):
    # Havacilikta her egri bir "Drag Index" degerine karsilik gelir.
    # Genellikle alttaki egri en yuksek, ustteki egri en dusuk dirence (drag) sahiptir.
    # Mevcut indeksler: [0, 25, 50, ... 300]
    drag_indices: List[int] = None

    # 4. Gorsel Elemanlar (Visual Elements):
    # Grafigi zenginlestirmek icin rastgele acilip kapanabilen ozellikler.
    add_grid: bool = True               # Arka plan izgarasi
    add_arrows: bool = True             # Egrileri gosteren oklar
    add_envelope_optimum: bool = True   # "Optimum Cruise" cizgisi
    add_envelope_endurance: bool = False # "Maximum Endurance" cizgisi
    add_vmax_line: bool = False         # "Vmax" hiz siniri cizgisi
    add_text_boxes: bool = True         # Bilgi kutucuklari (Legend vb.)
    add_fuel_labels: bool = True        # Yakit etiketleri
    add_drag_labels: bool = True        # Surukleme indeksi etiketleri

    # 5. Diversity & Realism (New):
    add_secondary_curves: bool = False  # Carpet plot (intersecting curves)
    x_scale: str = 'linear'           # 'linear' veya 'log'
    y_scale: str = 'linear'           # 'linear' veya 'log'
    marker_style: Optional[str] = None # 'o', 'x', '.', 'd' vb.
    background_type: str = 'plain'    # 'plain', 'grid', 'noisy_paper'


@dataclass(frozen=True)
class AxisDetectionResult:
    """
    Eksen Algilama Sonucu.
    Goruntu uzerindeki X ve Y eksenlerinin piksel koordinatlarini tutar.
    Bu, grafigi "okumak" (dijitallestirmek) icin gereklidir.
    """
    origin_px: tuple[float, float]       # Orijin noktasi (0,0) pikseli
    xref_px: tuple[float, float]         # X ekseninin bitis noktasi
    yref_px: tuple[float, float]         # Y ekseninin bitis noktasi
    x_axis_line: tuple[int, int, int, int] # X ekseni cizgisi (x1, y1, x2, y2)
    y_axis_line: tuple[int, int, int, int] # Y ekseni cizgisi (x1, y1, x2, y2)


@dataclass
class CurveData:
    """
    Tek bir egrinin verisi.
    Cizim yapildiktan sonra, o egrinin matematiksel verilerini saklar.
    """
    x: np.ndarray       # X koordinatlari (Numpy dizisi)
    y: np.ndarray       # Y koordinatlari (Numpy dizisi)
    drag_index: int     # Bu egrinin temsil ettigi Drag Index degeri


def _len2(x1: int, y1: int, x2: int, y2: int) -> float:
    """Iki nokta arasindaki mesafenin karesini hesapla (karekok almaktan hizlidir)."""
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    return dx * dx + dy * dy


def _clamp(v: int, lo: int, hi: int) -> int:
    """Bir sayiyi belirli bir aralikta (lo-hi) sinirla."""
    return lo if v < lo else hi if v > hi else v


def _pick_x_axis(segs: list[tuple[int, int, int, int]], w: int, h: int, curve_bbox: Optional[tuple[int,int,int,int]] = None) -> Optional[tuple[int, int, int, int]]:
    """
    Bulunan cizgiler arasindan X ekseni olmaya en uygun adayi secer.
    Kriterler:
    1. Yeterince uzun olmali.
    2. Yatay olmali (acisi < 15 veya > 165 derece).
    3. Egrilerin altinda yer almali.
    """
    candidates = []
    min_length = w * 0.25 # En az resim genisliginin ceyregi kadar

    # Egri bbox'i varsa, X ekseni egri tabaninda veya altinda olmali
    curve_bottom = curve_bbox[3] if curve_bbox else h * 0.5

    for x1, y1, x2, y2 in segs:
        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        if length < min_length:
            continue

        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 15 or angle > 165: # Yataylik kontrolu
            y_mid = (y1 + y2) / 2
            # X ekseni egri tabaninda veya altinda olmali
            if y_mid >= curve_bottom - 20:  # Kucuk tolerans (20px)
                candidates.append((y_mid, length, (x1, y1, x2, y2)))

    if not candidates:
        return None

    # En iyi adayi sec: En asagida olan (Y degeri en buyuk) ve en uzun olan
    # (Matplotlib koordinat sisteminde Y asagi dogru artar)
    candidates.sort(key=lambda x: (x[0], -x[1]))
    best = candidates[0][2]
    x1, y1, x2, y2 = best

    # Soldan saga dogru duzenle
    if x1 > x2:
        return (x2, y2, x1, y1)
    return best


def _pick_y_axis(segs: list[tuple[int, int, int, int]], w: int, h: int, curve_bbox: Optional[tuple[int,int,int,int]] = None) -> Optional[tuple[int, int, int, int]]:
    """
    Bulunan cizgiler arasindan Y ekseni olmaya en uygun adayi secer.
    Kriterler:
    1. Yeterince uzun olmali.
    2. Dikey olmali (acisi 75-105 derece arasi).
    3. Egrilerin solunda yer almali.
    """
    candidates = []
    min_length = h * 0.25

    # Egri bbox'i varsa, Y ekseni egri sol kenarinda veya solunda olmali
    curve_left = curve_bbox[0] if curve_bbox else w * 0.5

    for x1, y1, x2, y2 in segs:
        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        if length < min_length:
            continue

        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 75 < angle < 105: # Dikeylik kontrolu
            x_mid = (x1 + x2) / 2
            # Y ekseni egri sol kenarinda veya solunda olmali
            if x_mid <= curve_left + 20:  # Kucuk tolerans
                candidates.append((x_mid, length, (x1, y1, x2, y2)))

    if not candidates:
        return None

    # Egri soluna en yakin olani sec (ama yine de solunda)
    candidates.sort(key=lambda x: (-x[0], -x[1]))  # En sagdaki (egrilere en yakin) once
    best = candidates[0][2]
    x1, y1, x2, y2 = best

    if y1 < y2:
        return (x2, y2, x1, y1)
    return best


def _extend_axes_to_intersection(x_axis, y_axis):
    """
    X ve Y eksen cizgilerini kesistikleri noktaya (Orijin) kadar uzatir.
    Boylece tam bir "L" sekli elde edilir.
    """
    if x_axis is None or y_axis is None:
        return x_axis, y_axis

    x1, y1, x2, y2 = x_axis
    x3, y3, x4, y4 = y_axis

    # Egimleri hesapla (y = mx + c)
    x_dx = x2 - x1
    x_dy = y2 - y1
    y_dx = x4 - x3
    y_dy = y4 - y3

    # Sifira bolum hatasindan kacin (Dikey cizgiler icin egim sonsuzdur)
    if x_dx == 0:
        x_slope = float('inf')
    else:
        x_slope = x_dy / x_dx

    if y_dx == 0:
        y_slope = float('inf')
    else:
        y_slope = y_dy / y_dx

    # Dogrular paralelse kesismezler
    if x_slope == y_slope:
        return x_axis, y_axis

    # Analitik geometri ile kesisim (intersection) noktasini bul
    if x_slope == float('inf'):
        intersection_x = x1
        y_intercept = y3 - y_slope * x3
        intersection_y = y_slope * intersection_x + y_intercept
    elif y_slope == float('inf'):
        intersection_x = x3
        x_intercept = y1 - x_slope * x1
        intersection_y = x_slope * intersection_x + x_intercept
    else:
        x_intercept = y1 - x_slope * x1
        y_intercept = y3 - y_slope * x3
        intersection_x = (y_intercept - x_intercept) / (x_slope - y_slope)
        intersection_y = x_slope * intersection_x + x_intercept

    ix, iy = int(round(intersection_x)), int(round(intersection_y))

    # X eksenini uzat: Mevcut uc noktalardan hangisi kesisime daha uzaksa onu koru, digerini kesisim yap
    if (x1 - ix)**2 + (y1 - iy)**2 > (x2 - ix)**2 + (y2 - iy)**2:
        new_x_axis = (ix, iy, x1, y1)
    else:
        new_x_axis = (ix, iy, x2, y2)

    # Y eksenini uzat
    if (x3 - ix)**2 + (y3 - iy)**2 > (x4 - ix)**2 + (y4 - iy)**2:
        new_y_axis = (ix, iy, x3, y3)
    else:
        new_y_axis = (ix, iy, x4, y4)

    return new_x_axis, new_y_axis


def compute_axes_from_config(config: 'ChartConfig', W: int, H: int) -> AxisDetectionResult:
    """
    Eksen piksel koordinatlarini dogrudan yapilandirmadan hesapla.
    Tam eksen araliklarini bildigimiz sentetik veriler icin.
    Bu, "Ground Truth" (Gercek Veri) eksen bilgisidir.
    """
    # Matplotlib'de eksenler (0,0) ile (1,1) arasinda normalize edilmis koordinatlardadir.
    # Ancak biz piksel koordinatlarina (0..W, 0..H) cevirmeliyiz.

    # 1. Veri Koordinatlarini Piksel Koordinatlarina Donusturme Yardimcisi:
    # Goruntu isleme kutuphanelerinde (OpenCV) Y ekseni asagi dogru artar (0 en ust).
    # Matplotlib ve verilerde Y yukari dogru artar. Bu yuzden Y'yi ters ceviriyoruz (1.0 - ...).
    def data_to_px(dx, dy):
        # Normalize edilmis (0-1) degerleri piksel boyutlariyla carp
        px = int((dx - config.x_min) / (config.x_max - config.x_min) * W)
        py = int((1.0 - (dy - config.y_min) / (config.y_max - config.y_min)) * H)
        return _clamp(px, 0, W-1), _clamp(py, 0, H-1)

    # 2. Kritik Noktalari Hesapla:

    # Orijin (Grafik alaninin sol alt kosesi)
    ox, oy = data_to_px(config.x_min, config.y_min)

    # X ekseninin en sag ucu (Right End)
    x2_px, y2_px = data_to_px(config.x_max, config.y_min)

    # Y ekseninin en ust ucu (Top End)
    x4_px, y4_px = data_to_px(config.x_min, config.y_max)

    # X ekseninin baslangici (Orijin ile ayni olmali)
    x1_px, y1_px = ox, oy
    # Y ekseninin baslangici
    x3_px, y3_px = ox, oy

    return AxisDetectionResult(
        origin_px=(float(ox), float(oy)),
        xref_px=(float(x2_px), float(y2_px)),  # Sag uc
        yref_px=(float(x4_px), float(y4_px)),  # Ust uc
        x_axis_line=(x1_px, y1_px, x2_px, y2_px),
        y_axis_line=(x3_px, y3_px, x4_px, y4_px),
    )


def detect_axes_from_rgb(image_rgb: np.ndarray, curve_mask: Optional[np.ndarray] = None) -> Optional[AxisDetectionResult]:
    """RGB goruntusunden X/Y eksenlerini algila, rehberlik icin egri maski kullan."""
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None

    h, w = gray.shape

    # Saglanmissa masktan egri sinirlayici kutusunu al
    curve_bbox = None
    if curve_mask is not None:
        if len(curve_mask.shape) == 3:
            mask_gray = cv2.cvtColor(curve_mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = curve_mask
        ys, xs = np.where(mask_gray > 127)
        if len(xs) > 0:
            curve_bbox = (xs.min(), ys.min(), xs.max(), ys.max())

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Goruntu boyutuna orantili minimum cizgi uzunlugu
    min_line_len = max(50, min(w, h) // 4)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=15,
    )

    if lines is None:
        return None

    segs = [tuple(map(int, l[0])) for l in lines]
    x_axis = _pick_x_axis(segs, w, h, curve_bbox)
    y_axis = _pick_y_axis(segs, w, h, curve_bbox)

    if x_axis is None or y_axis is None:
        return None

    # Eksenleri kesisime kadar uzat
    x_axis, y_axis = _extend_axes_to_intersection(x_axis, y_axis)

    # Orijin kesisim noktasinda (her iki uzatilmis eksenin ilk noktasi)
    ox, oy = x_axis[0], x_axis[1]

    # Referans noktalari diger uclarda
    xref_x, xref_y = x_axis[2], x_axis[3]
    yref_x, yref_y = y_axis[2], y_axis[3]

    # Goruntu sinirlarina sikistir
    ox = _clamp(ox, 0, w - 1)
    oy = _clamp(oy, 0, h - 1)
    xref_x = _clamp(xref_x, 0, w - 1)
    xref_y = _clamp(xref_y, 0, h - 1)
    yref_x = _clamp(yref_x, 0, w - 1)
    yref_y = _clamp(yref_y, 0, h - 1)

    return AxisDetectionResult(
        origin_px=(float(ox), float(oy)),
        xref_px=(float(xref_x), float(xref_y)),
        yref_px=(float(yref_x), float(yref_y)),
        x_axis_line=(ox, oy, xref_x, xref_y),
        y_axis_line=(ox, oy, yref_x, yref_y),
    )


def generate_curve_shape(
    x: np.ndarray,
    curve_type: str,
    curve_index: int,
    total_curves: int
) -> np.ndarray:
    """
    Matematiksel formullerle farkli egri sekilleri olustur.

    Args:
        x: X ekseni degerleri (numpy dizisi)
        curve_type: Egri tipi ('peaked', 'rising', 'falling', 'wavy'...)
        curve_index: Bu kacinci egri? (Yukseklik/Altitude etkisi icin)
        total_curves: Toplam egri sayisi
    """
    # Egrinin goreceli yuksekligi (0.0 = en alt, 1.0 = en ust)
    # STANDART DEGIL: Egriler arasindaki mesafeye rastgelelik ekle (siradan uzaklas)
    alt = (curve_index + random.uniform(-0.15, 0.15)) / max(total_curves - 1, 1)
    alt = np.clip(alt, 0.0, 1.0)

    # X degerlerini 0 ile 1 arasina normalize et (Formullerde kolaylik icin)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # === HER EGRI ICIN BAGIMSIZ ASIRI RASTGELELIK ===
    # Bu degiskenler onceden butun grafik icin ortak seciliyordu, simdi her egri (cizgi)
    # kendi basina ayri bir egim karakterine sahip olacak, boylece paralel gitmeyecekler.

    if curve_type == 'peaked':
        # "Tepecikli" Egriler (Orn: Range vs Mach)
        # Tepe noktasi konumunu cok daha genis bir alana yay (0.15'ten 0.85'e kadar her yerde olabilir)
        peak_pos = random.uniform(0.15, 0.85)
        start_y = 0.10 + random.uniform(-0.06, 0.06)
        peak_y = 0.40 + alt * 0.45 + random.uniform(-0.10, 0.15) # Tepe yuksekligi varyasyonu (Daha ekstrem)
        end_y = 0.15 + alt * 0.25 + random.uniform(-0.08, 0.08)

        # Egim siddetleri: Cok dik (exponential) veya cok yatay (logarithmic) cikis/inisler
        rise_exp = random.uniform(0.3, 4.0)
        fall_exp = random.uniform(0.3, 3.5)

        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                # Yukselis kismi
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (1 - (1 - progress) ** rise_exp)
            else:
                # Dusus kismi
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = peak_y - (peak_y - end_y) * (progress ** fall_exp)

    elif curve_type == 'peaked_oval':
        # "Oval Tepecikli" - Sinuzoidal gecisli
        peak_pos = random.uniform(0.15, 0.85) # Daha asimetrik peakler
        start_y = 0.10 + random.uniform(-0.06, 0.06)
        peak_y = 0.45 + alt * 0.45 + random.uniform(-0.15, 0.15)
        end_y = 0.15 + alt * 0.30 + random.uniform(-0.08, 0.08)

        # Sinus egrilerinin sikismasini/yayilmasini inanilmaz seviyelere cikar
        rise_exp = random.uniform(0.5, 3.0)
        fall_exp = random.uniform(0.5, 3.0)

        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (math.sin(progress * math.pi / 2) ** rise_exp)
            else:
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = end_y + (peak_y - end_y) * (math.cos(progress * math.pi / 2) ** fall_exp)

    elif curve_type == 'wavy':
        # "Dalgali" Egriler (Multi-peak)
        freq = random.uniform(0.3, 5.0) # Dalga sikligini asiri aralikli yap
        phase = random.uniform(0, 2 * np.pi)

        # Ana dalga sinyali ve rastgele harmonikler
        wave = 0.5 + random.uniform(0.1, 0.45) * np.sin(2 * np.pi * (x_norm * freq + phase))
        wave += random.uniform(0.01, 0.3) * np.sin(4 * np.pi * (x_norm * (freq * random.uniform(0.3, 2.0)) + phase))

        # Tumsek (Hump) eklentisi (Sayisini da artirabiliriz)
        if random.random() > 0.2: # %80 ihtimalle tumsek at
            hump_center = random.uniform(0.1, 0.9)
            hump_width = random.uniform(0.05, 0.5)
            hump_height = random.uniform(0.1, 0.4) * (1 if random.random() > 0.4 else -1) # Bazen cukur
            hump = hump_height * np.exp(-((x_norm - hump_center) / hump_width) ** 2)
            wave += hump

        # Altitute (Yukseklik) offseti ekle
        y = wave * 0.5 + (alt * 0.5)
        y = np.clip(y, 0.04, 0.96)

    elif curve_type == 'rising':
        # "Yukselen" Egriler (Logaritmik, Lineer, Exponential)
        start_y = 0.05 + alt * 0.20 + random.uniform(-0.10, 0.10)
        end_y = 0.40 + alt * 0.55 + random.uniform(-0.25, 0.25)

        # Egim cesitliligi (Curvature)
        # Kesisen ve birbirine batan cizgiler icin en onemli degisken:
        curvature = random.uniform(0.2, 5.0)
        y = start_y + (end_y - start_y) * (x_norm ** curvature)

    elif curve_type == 'falling':
        # "Alcalan" Egriler
        start_y = 0.50 + alt * 0.50 + random.uniform(-0.25, 0.25)
        end_y = 0.05 + alt * 0.20 + random.uniform(-0.10, 0.10)

        # Dusus hizinin vahsi varyasyonu
        curvature = random.uniform(0.2, 5.0)
        y = start_y - (start_y - end_y) * (x_norm ** curvature)

    else:  # mixed
        # Karisik mod: Rastgele bir tip sec ve recursive cagir
        return generate_curve_shape(x, random.choice(['peaked', 'peaked_oval', 'rising', 'falling', 'wavy']),
                                   curve_index, total_curves)

    return y


def fig_to_array(fig, dpi=150, tight=True) -> np.ndarray:
    """Matplotlib seklini numpy dizisine donustur."""
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white', edgecolor='none')
    else:
        # Fixed size for mask consistency
        fig.savefig(buf, format='png', dpi=dpi,
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def draw_chart_matplotlib(
    config: ChartConfig,
    W: int = 800,
    H: int = 600
) -> Tuple[np.ndarray, np.ndarray, List[CurveData]]:
    """
    Matplotlib kutuphanesi ile "temiz" bir grafik ciz.

    Bu fonksiyon, konfigurasyondaki (ChartConfig) receteyi alip,
    yuksek kaliteli bir grafik goruntusu (RGB) ve segmentasyon maskesi uretir.

    Args:
        config: Grafik ayarlari
        W, H: Cikti goruntu boyutlari (piksel)

    Doner:
        full_img: Cizilen grafik (RGB Numpy dizisi)
        mask: Sadece egrilerin oldugu siyah-beyaz maske
        curves_data: Cizilen egrilerin matematiksel verileri
    """
    # Inc basina piksel (DPI) hesabi. Matplotlib inc ile calisir.
    # Varsayilan DPI=100 varsayarsak:
    fig_w, fig_h = W / 100, H / 100

    # 1. Egri Verilerinin Hazirlanmasi:
    # Grafigi cizmeden once, her egrinin X ve Y koordinatlarini hesapliyoruz.
    x = np.linspace(config.x_min + 0.02, config.x_max - 0.02, 400)
    curves_data = []

    for i in range(config.n_curves):
        # generate_curve_shape ile 0-1 arasinda normalize edilmis sekli al
        y_norm = generate_curve_shape(x, config.curve_type, i, config.n_curves)

        # Normalize sekli gercek Y araligina (y_min, y_max) olcekle
        y = config.y_min + y_norm * (config.y_max - config.y_min)
        y = np.clip(y, config.y_min + 0.001, config.y_max - 0.001)

        # Bu egriye ait Drag Index'i belirle
        drag_idx = config.drag_indices[i] if config.drag_indices else i * 25
        curves_data.append(CurveData(x=x.copy(), y=y, drag_index=drag_idx))

    # 2. Grafik Cizimi (Plotting):
    # Matplotlib figurunu baslat
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    # Eksen Olcegi (Scale):
    ax1.set_xscale(config.x_scale)
    ax1.set_yscale(config.y_scale)

    ax1.set_xlim(config.x_min, config.x_max)
    ax1.set_ylim(config.y_min, config.y_max)

    # Izgara (Grid) Cizimi:
    if config.add_grid:
        x_range = config.x_max - config.x_min
        y_range = config.y_max - config.y_min

        # Izgara sikligini araliga gore otomatik ayarla
        x_major = 0.1 if x_range > 0.5 else 0.05
        y_major = 0.01 if y_range < 0.08 else 0.02

        ax1.set_xticks(np.arange(config.x_min, config.x_max + 0.001, x_major))
        ax1.set_xticks(np.arange(config.x_min, config.x_max + 0.001, x_major/2), minor=True)
        ax1.set_yticks(np.arange(config.y_min, config.y_max + 0.001, y_major))
        ax1.set_yticks(np.arange(config.y_min, config.y_max + 0.001, y_major/2), minor=True)

        # Ana ve ara izgaralar farkli kalinlikta
        ax1.grid(True, which='major', linewidth=0.8, alpha=0.5, color='black')
        ax1.grid(True, which='minor', linewidth=0.4, alpha=0.3, color='black')

    # Eksen Cizgileri (Spines/Axes Lines):
    # Cerceveyi kalinlastir, boylece goruntu isleme ile eksenleri bulmak kolaylasir.
    ax1.axhline(y=config.y_min, color='black', linewidth=2.0, zorder=10) # Alt X
    ax1.axvline(x=config.x_min, color='black', linewidth=2.0, zorder=10) # Sol Y

    # Eksen Centikleri (Ticks):
    # Iceriye dogru bakan centikler
    ax1.tick_params(axis='both', which='major', length=6, width=1.5, direction='in')
    ax1.tick_params(axis='both', which='minor', length=3, width=1.0, direction='in')

    # 4. Spines (Cerceve Cizgileri):
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Eksen Etiketleri (Labels):
    ax1.set_xlabel('MACH NUMBER', fontsize=10, fontweight='bold')
    # "Specific Range" = Yakit Verimliligi (Birim yakitla gidilen mesafe)
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', fontsize=8)

    # 5. Egrilerin Cizimi:
    # Daha once hesapladigimiz (1. adim) egrileri siyah (k) ve ince cizgilerle ciziyoruz.
    for curve in curves_data:
        ax1.plot(curve.x, curve.y, 'k-', linewidth=config.curve_lw,
                 marker=config.marker_style, markevery=random.randint(20, 50), markersize=3)

    # C) SECONDARY CURVES (Carpet Plot Effect)
    # Bu, grafigi kesen dikeyimsi egri ailesidir.
    if config.add_secondary_curves:
        # Ana egrilerin orta noktalarini ve uclarini kullanarak dikey gecisler olustur
        for x_idx in [int(len(x)*0.1), int(len(x)*0.3), int(len(x)*0.5), int(len(x)*0.7), int(len(x)*0.9)]:
            cx_vals = [c.x[x_idx] for c in curves_data]
            cy_vals = [c.y[x_idx] for c in curves_data]
            ax1.plot(cx_vals, cy_vals, 'k-', linewidth=config.curve_lw * 0.7, alpha=0.8)

    # 6. Zarf Egrileri (Envelopes):
    # Bu cizgiler, ucagin en verimli oldugu noktalari birlestirir.

    # A) OPTIMUM CRUISE (En iyi seyir hizi)
    if config.add_envelope_optimum:
        if config.curve_type == 'peaked':
            # Tepe noktalarini birlestir
            envelope_pts = [(curve.x[np.argmax(curve.y)], curve.y.max()) for curve in curves_data]
        else:
            # Orta noktalari birlestir (Alternatif)
            envelope_pts = [(curve.x[int(len(curve.x)*0.5)], curve.y[int(len(curve.y)*0.5)]) for curve in curves_data]

        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2) # Zarf cizgisi biraz daha kalin

        # Etiket ekle
        ax1.text(ex[0] - 0.03, ey[-1] + (config.y_max - config.y_min) * 0.02,
                'OPTIMUM\nCRUISE', fontsize=8, ha='right', va='bottom')

    # B) MAXIMUM ENDURANCE (Havada en uzun kalma suresi)
    if config.add_envelope_endurance:
        # Genellikle daha dusuk hizlarda olur (sol tarafta)
        envelope_pts = [(curve.x[int(len(curve.x)*0.2)], curve.y[int(len(curve.y)*0.2)]) for curve in curves_data]
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)

        ax1.text(ex[-1] - 0.02, ey[0] - (config.y_max - config.y_min) * 0.02,
                'MAXIMUM\nENDURANCE', fontsize=8, ha='right', va='top')

    # 7. Oklar ve Aciklamalar (Arrows Logic):
    # Bu kisim grafikteki oklari ve veri etiketlerini rastgele yerlestirir.
    if config.add_arrows:
        for idx, curve in enumerate(reversed(curves_data)):
            cx, cy = curve.x, curve.y

            # Rastgele ok konumunu sec: uc nokta VEYA egrinin ortasi
            if random.random() < 0.5:
                # A) Uc Noktaya Ok:
                # Egrinin en sagina eklenir, kuyrugu daha sagda olur.
                arrow_idx = -1
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]

                dx = random.uniform(0.04, 0.08)
                dy = random.uniform(-0.005, 0.005)
                x_tail = x_head + dx
                y_tail = y_head + dy
            else:
                # B) Orta Bolgeye Ok:
                # Egrinin ortasinda bir yere carprazlama gelir.
                mid_start = len(cx) // 4
                mid_end = 3 * len(cx) // 4
                arrow_idx = random.randint(mid_start, mid_end)
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]

                # Rastgele aci ve mesafe
                angle = random.uniform(20, 70)  # derece
                dist = random.uniform(0.05, 0.10)

                # Sag-ust veya Sag-alt capraz
                if random.random() < 0.5:
                    dx = dist * math.cos(math.radians(angle))
                    dy = dist * math.sin(math.radians(angle))
                else:
                    dx = dist * math.cos(math.radians(-angle))
                    dy = dist * math.sin(math.radians(-angle))
                x_tail = x_head + dx
                y_tail = y_head + dy

            # Ok Cizgisi (Kuyruktan Basa):
            ax1.plot([x_tail, x_head], [y_tail, y_head], color="black", linewidth=0.6)

            # Ok Basini (Arrowhead) Rastgelestir:
            # Bazen ici bos (open), bazen dolu (filled) ucgen.
            if random.random() < 0.4:
                arrow_style = random.choice(["-|>", "->"])
                fill_style = "none"  # Ici bos
            else:
                arrow_style = random.choice(["-|>", "-|>", "->"])
                fill_style = "black" # Ici dolu

            # Annotation (Ok Basi Cizimi):
            ax1.annotate(
                "",
                xy=(x_head, y_head),      # Hedef nokta (Ok ucu)
                xytext=(x_tail, y_tail),  # Kaynak nokta (Kuyruk)
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    lw=random.uniform(0.7, 1.1),
                    color="black",
                    fc=fill_style,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=random.uniform(12, 18), # Ok basi buyuklugu
                ),
            )

            # Ekstra Cizgiler (Decoration Dashes):
            # Grafiklerde bazen oklarin yaninda kesikli referans cizgileri olur.
            if random.random() < 0.95: # Olasilik artirildi (0.85 -> 0.95)
                dash_len = random.uniform(0.06, 0.14)
                dash_angle = math.radians(random.choice([35, 45, 55, 65, 75]))
                dash_dx = dash_len * math.cos(dash_angle)
                dash_dy = dash_len * math.sin(dash_angle)

                # Rastgele bir konuma yerlestir (grafigin ortalarinda)
                base_x = config.x_min + (config.x_max - config.x_min) * random.uniform(0.25, 0.75)
                base_y = config.y_min + (config.y_max - config.y_min) * random.uniform(0.25, 0.75)

                dash_start = (base_x, base_y)
                dash_end = (dash_start[0] + dash_dx, dash_start[1] + dash_dy)

                ax1.plot([dash_start[0], dash_end[0]], [dash_start[1], dash_end[1]],
                    color="black", linewidth=0.6, linestyle=(0, (14, 8))) # Seyrek kesikli cizgi

            # Drag Index Etiketi (Rakam):
            # Oku veya cizgiyi takip eden rakam (Orn: "0", "50", "100")
            label_x = x_tail + random.uniform(0.06, 0.10)
            ax1.text(label_x, y_tail + random.uniform(-0.002, 0.002),
                    str(curve.drag_index), fontsize=8, va='center', ha='left')

    # 8. Ekstra Suslemeler (Decorations):
    # Ana oklarin disinda, grafigin bos kalan yerlerine rastgele kesikli cizgiler ekler.
    # KULLANICI ISTEGI: Kesikli cizgilerin sayisi artirildi.
    if random.random() < 0.95:  # Olasilik artirildi (0.7 -> 0.95)
        n_extra_dashes = random.randint(8, 15)  # Sayi artirildi (2-6 -> 8-15)
        for _ in range(n_extra_dashes):
            # Rastgele merkezi konum
            cx = config.x_min + (config.x_max - config.x_min) * random.uniform(0.1, 0.9)
            cy = config.y_min + (config.y_max - config.y_min) * random.uniform(0.1, 0.9)
            dash_len = random.uniform(0.04, 0.10)
            dash_angle = math.radians(random.choice([35, 45, 55, 65, 75]))
            dash_dx = dash_len * math.cos(dash_angle)
            dash_dy = dash_len * math.sin(dash_angle)
            ax1.plot([cx, cx + dash_dx], [cy, cy + dash_dy],
                    color="black", linewidth=0.5, linestyle=(0, (12, 7)))

    # 9. Metin Kutulari (Text Boxes):
    if config.add_text_boxes:
        # TOPLAM YAKIT AKISI kutusu (Sag ust kose)
        ax1.text(
            config.x_max - 0.05, config.y_max - 0.005,
            'TOTAL FUEL FLOW—\nPOUNDS PER HOUR',
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
        )

        # Legend kutusu (Sol ust kose)
        legend_x = config.x_min + (config.x_max - config.x_min) * 0.15
        legend_y = config.y_max - (config.y_max - config.y_min) * 0.1

        ax1.text(
            legend_x, legend_y,
            '◄─ CRUISE    DASH ─►\n      AOA          AOA\n(USED FOR INTERFERENCE\n DRAG DETERMINATION)',
            fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
        )

    # 10. Surukleme Indeksi Listesi (Drag Labels):
    # Grafigin sag tarafinda listelenen referans rakamlar.
    if config.add_drag_labels:
        labels = ['0.00', '25.00', '50.00', '75.00', '100.00', '125.00', '150.00']
        base_x = config.x_min + (config.x_max - config.x_min) * 0.65
        base_y = config.y_min + (config.y_max - config.y_min) * 0.15

        for i, lbl in enumerate(labels[:random.randint(4, 7)]):
            ax1.text(base_x + random.uniform(-0.02, 0.02),
                    base_y + i * (config.y_max - config.y_min) * 0.05,
                    lbl, fontsize=7, alpha=0.9)

    # 11. Vmax Cizgisi (Hiz Siniri):
    if config.add_vmax_line:
        vmax_pts = [(curve.x[int(len(curve.x)*0.85)], curve.y[int(len(curve.y)*0.85)]) for curve in curves_data]
        vmax_pts.sort(key=lambda p: p[1])
        vx, vy = zip(*vmax_pts)
        ax1.plot(vx, vy, 'k--', linewidth=0.8) # Kesikli cizgi
        ax1.text(vx[-1], vy[-1] + 0.003, r'$V_{max}$(MIL)', fontsize=7)

    # Goruntuyu Numpy Dizisine Cevir (RGB)
    full_img = fig_to_array(fig1, dpi=150, tight=True)
    full_img = cv2.resize(full_img, (W, H))

    # ========== MASKE OLUSTURMA (Sadece Egriler) ==========
    # Ayni koordinat sisteminde yeni bir figur aciyoruz.
    # Ancak arka plan SIYAH, cizgiler BEYAZ olacak.
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.set_xlim(config.x_min, config.x_max)
    ax2.set_ylim(config.y_min, config.y_max)
    ax2.set_position([0, 0, 1, 1])  # Tum alani kapla
    ax2.axis('off') # Eksenleri gizle
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')

    # Maske uzerinde daha iyi algilama icin egrileri biraz daha kalin ciz
    for curve in curves_data:
        ax2.plot(curve.x, curve.y, 'w-', linewidth=config.curve_lw + 0.6)

    mask_img = fig_to_array(fig2, dpi=150, tight=False)
    mask_img = cv2.resize(mask_img, (W, H))

    # Goruntuyu ikili (binary) maskeye cevir (0 veya 255)
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)

    return full_img, mask, curves_data


def random_config() -> ChartConfig:
    """
    Rastgele grafik yapilandirmasi olustur.

    Gercek dunya verilerine dayali araliklar ve dagilimlar kullanir.
    Amac, egitim veri setinde cesitliligi (diversity) maksimize etmektir.
    """
    x_ranges = [
        (0.30, 0.95), (0.30, 1.00), (0.40, 1.10), (0.50, 1.20),
        (0.50, 1.30), (0.50, 1.40), (0.60, 1.40)
    ]
    y_ranges = [
        (0.04, 0.15), (0.05, 0.15), (0.06, 0.17), (0.07, 0.18),
        (0.08, 0.19), (0.08, 0.20), (0.05, 0.14)
    ]

    x_min, x_max = random.choice(x_ranges)
    y_min, y_max = random.choice(y_ranges)

    # Agirlikli egri turu secimi (Gercek verilerdeki sikliga gore)
    # peaked_oval: %28, peaked: %26, rising: %16, falling: %14, wavy: %10, mixed: %6
    curve_types = ['peaked_oval'] * 28 + ['peaked'] * 26 + ['rising'] * 16 + ['falling'] * 14 + ['wavy'] * 10 + ['mixed'] * 6
    curve_type = random.choice(curve_types)

    # Egri turune gore farkli egri sayilari
    if curve_type == 'wavy':
        n_curves = random.randint(3, 6)
    elif curve_type in ['falling', 'mixed']:
        n_curves = random.randint(4, 7)
    elif curve_type == 'rising':
        n_curves = random.randint(5, 8)
    else:  # peaked, peaked_oval
        n_curves = random.randint(6, 12)

    # Mevcut surukleme indeksleri
    all_drag_indices = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300]

    # Bu grafik icin rastgele maksimum surukleme indeksi araligi
    max_drag_options = [100, 125, 150, 200, 250, 300]
    max_drag = random.choice(max_drag_options)

    # 0'dan max_drag'a kadar surukleme indekslerini sec
    available_drag_indices = [d for d in all_drag_indices if d <= max_drag]

    # Rastgele n_curves surukleme indekslerini sec
    selected_drag_indices = sorted(random.sample(available_drag_indices, min(n_curves, len(available_drag_indices))))

    # Yeterli degilse, sahip olduklarimizla doldur
    while len(selected_drag_indices) < n_curves:
        selected_drag_indices.append(selected_drag_indices[-1] + 25)

    # Tersine cevir boylece ust egri (en yuksek Y) en dusuk surukleme indeksine (0) sahip olsun
    selected_drag_indices.reverse()
    selected_drag_indices = selected_drag_indices[:n_curves]

    return ChartConfig(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        n_curves=n_curves,
        curve_type=curve_type,
        curve_lw=random.uniform(0.3, 0.6),  # Random thickness (0.3=very thin, 0.6=normal)
        drag_indices=selected_drag_indices,
        add_grid=random.random() < 0.95,
        add_arrows=random.random() < 0.85,
        add_envelope_optimum=random.random() < 0.70,
        add_envelope_endurance=random.random() < 0.35,
        add_vmax_line=random.random() < 0.25,
        add_text_boxes=random.random() < 0.75,
        add_fuel_labels=random.random() < 0.80,
        add_drag_labels=random.random() < 0.55,
    )


def add_scan_artifacts(img: np.ndarray, strength: float = 1.0, background_style: str = 'plain') -> np.ndarray:
    """
    Sentetik grafige "tarama/fotokopi" kusurlari (artifacts) ekler.
    """
    h, w = img.shape[:2]

    # 0. Background Styles (New):
    if background_style == 'grid':
        # Mavi veya yesil kareli kagit efekti
        grid_color = random.choice([(200, 220, 255), (210, 255, 210)]) # Acik mavi veya acik yesil
        spacing = random.randint(20, 40)
        overlay = img.copy()
        for x in range(0, w, spacing):
            cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, spacing):
            cv2.line(overlay, (0, y), (w, y), grid_color, 1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    pil_img = Image.fromarray(img)

    # 1. Hafif Dondurme (Rotation):
    angle = random.uniform(-1.5, 1.5) * strength
    pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)

    # 2. Renk/Parlaklik Bozulmalari:
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.85, 1.15))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.80, 1.30))

    # xerox effect (High contrast / thresholding)
    if random.random() < 0.3 * strength:
        arr = np.array(pil_img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Adaptive thresholding for xerox look
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        # Blend it back a bit to avoid total loss of detail
        pil_img = Image.fromarray(cv2.cvtColor(cv2.addWeighted(gray, 0.4, binary, 0.6, 0), cv2.COLOR_GRAY2RGB))

    # 3. Piksel Gurultusu (Noise):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise_level = 0.015 * strength
    if random.random() < 0.2: # Salt and pepper
        s_p = np.random.choice([0, 1, 0.5], size=arr.shape[:2], p=[0.01, 0.01, 0.98])
        for i in range(3): arr[:,:,i] *= s_p

    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 1)

    # 4. JPEG Sikistirma Bozulmalari:
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(
        buf, format='JPEG', quality=random.randint(40, 85)
    )
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def make_sample(W: int = 512, H: int = 512, seed: int = None,
                add_artifacts: bool = True) -> Tuple[np.ndarray, np.ndarray, List[CurveData]]:
    """
    Tek bir egitim ornegi olusturur (Goruntu + Hedef Maske).

    Adimlar:
    1. config = random_config() ile rastgele bir grafik tasarla.
    2. draw_chart_matplotlib() ile bu grafigi yuksek kalitede ciz.
    3. add_artifacts is True ise, tarama/fotokopi gurultusu ekle.
    4. colorize_curves_from_data() ile modelin bulmasi gereken "Temiz" hedef goruntuyu olustur.

    Doner:
        input_img: Modele girecek gurultulu RGB (veya BGR) goruntu.
        target_img: Modelin uretmesi beklenen temiz, renkli egriler.
        curves_data: Egrilerin matematiksel verisi (Koordinatlar, Drag Index vb.)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. & 2. Tasarla ve Ciz
    config = random_config()
    full_img, mask, curves_data = draw_chart_matplotlib(config, W, H)

    # 3. Gurultu Ekle (Data Augmentation)
    if add_artifacts:
        full_img = add_scan_artifacts(full_img)

    # 4. Hedef Goruntuyu (Ground Truth) Olustur
    # Arka plan siyah, egriler farkli renklerde.
    target_colored = colorize_curves_from_data(
        curves_data, config, W, H,
        show_axes=False,
        black_background=True
    )

    # OpenCV (cv2) BGR formati kullanir, Matplotlib RGB. Donusum yap.
    full_img_bgr = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)

    return full_img_bgr, target_colored, curves_data


def colorize_curves_from_data(
    curves_data: List[CurveData],
    config: ChartConfig,
    W: int,
    H: int,
    show_axes: bool = True,
    axis_result: Optional[AxisDetectionResult] = None,
    full_img_rgb: Optional[np.ndarray] = None,
    black_background: bool = False
) -> np.ndarray:
    """
    Egri verilerinden (CurveData) yola cikarak "Temiz" bir goruntu olusturur.

    Amac:
    - Model egitimi icin "Ground Truth" (Hedef) goruntusu uretmek (Siyah arka plan, renkli cizgiler).
    - Veya gorsellestirme/debug amacli, bulunan egrileri orijinal uzerine cizmek.

    Her egriye farkli bir renk (Hue) atanir, boylece kesisim noktalarinda bile ayirt edilebilirler.
    """
    fig_w, fig_h = W / 100, H / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')

    bg_color = 'black' if black_background else 'white'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # X ve Y eksenlerini ciz
    if show_axes:
        # X ekseni (alt)
        ax.axhline(y=config.y_min, color='black', linewidth=1.5, zorder=1)
        # Y ekseni (sol)
        ax.axvline(x=config.x_min, color='black', linewidth=1.5, zorder=1)

        # Eksen etiketleri
        ax.text(config.x_max, config.y_min - (config.y_max - config.y_min) * 0.05,
               f'X: {config.x_min:.2f} - {config.x_max:.2f}',
               fontsize=9, ha='right', va='top', color='black')
        ax.text(config.x_min - (config.x_max - config.x_min) * 0.02, config.y_max,
               f'Y: {config.y_min:.2f} - {config.y_max:.2f}',
               fontsize=9, ha='right', va='top', color='black', rotation=90)

    n_curves = len(curves_data)
    for i, curve in enumerate(curves_data):
        hue = int(180 * i / max(n_curves, 1))
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        ax.plot(curve.x, curve.y, color=np.array(rgb_color) / 255.0, linewidth=config.curve_lw + 0.3, zorder=2)

    colored = fig_to_array(fig, dpi=150, tight=False)
    colored = cv2.resize(colored, (W, H))
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    # Istenirse tam goruntuden eksenleri otomatik olarak algila
    if show_axes and axis_result is None and full_img_rgb is not None:
        axis_result = detect_axes_from_rgb(full_img_rgb)

    # Algilanan eksenleri renkli ciktiya yerlestir (piksel alani)
    if show_axes and axis_result is not None:
        x1, y1, x2, y2 = axis_result.x_axis_line
        x3, y3, x4, y4 = axis_result.y_axis_line
        cv2.line(colored_bgr, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.line(colored_bgr, (x3, y3), (x4, y4), (0, 0, 0), 2)

        ox, oy = map(int, axis_result.origin_px)
        cv2.circle(colored_bgr, (ox, oy), 3, (0, 0, 0), -1)

        # Eksen uc noktalarini etiketle
        cv2.putText(colored_bgr, "X", (x2 + 4, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(colored_bgr, "Y", (x4 + 4, y4 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return colored_bgr


def generate_coco_annotation(
    curves_data: List[CurveData],
    config: ChartConfig,
    image_id: int,
    image_filename: str,
    W: int,
    H: int
) -> Dict:
    """
    Egriler icin COCO (Common Objects in Context) formatinda anotasyon olusturur.

    Nesne Algilama (Mask R-CNN vb.) modelleri icin standart format.
    Her egri bir "nesne" (instance) olarak etiketlenir.

    Doner:
        Dict: Goruntu bilgisi ve segmentasyon verilerini iceren sozluk.
    """
    import json

    def data_to_px(dx, dy):
        """Veri koordinatlarini piksel koordinatlarina donustur."""
        px = int((dx - config.x_min) / (config.x_max - config.x_min) * W)
        py = int((1.0 - (dy - config.y_min) / (config.y_max - config.y_min)) * H)
        return _clamp(px, 0, W-1), _clamp(py, 0, H-1)

    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "width": W,
        "height": H,
        "x_min": float(config.x_min),
        "x_max": float(config.x_max),
        "y_min": float(config.y_min),
        "y_max": float(config.y_max)
    }

    annotations = []

    for ann_id, curve in enumerate(curves_data):
        # Egri verilerini piksel koordinatlarina donustur
        curve_px = np.array([data_to_px(x_val, y_val) for x_val, y_val in zip(curve.x, curve.y)])

        # Sinirlayici kutuyu hesapla
        if len(curve_px) > 0:
            xs = curve_px[:, 0]
            ys = curve_px[:, 1]
            x_min_px = int(xs.min())
            y_min_px = int(ys.min())
            x_max_px = int(xs.max())
            y_max_px = int(ys.max())

            bbox_width = x_max_px - x_min_px
            bbox_height = y_max_px - y_min_px
            bbox_area = bbox_width * bbox_height

            if bbox_width > 0 and bbox_height > 0:
                # Egri noktalarindan segmentasyon poligonu olustur
                segmentation = curve_px.tolist()

                annotation = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # Tum egriler ayni kategoriye sahip
                    "area": float(bbox_area),
                    "bbox": [float(x_min_px), float(y_min_px), float(bbox_width), float(bbox_height)],
                    "iscrowd": 0,
                    "drag_index": int(curve.drag_index),
                    "segmentation": [segmentation]
                }
                annotations.append(annotation)

    return {
        "image": image_info,
        "annotations": annotations
    }


if __name__ == "__main__":
    import os
    import json

    print("SURUKLEME INDEKSI destegi ile V5 ornekleri olusturuluyor...")
    print("=" * 60)
    print("Bu modul dogrudan calistirildiginda test ornekleri uretir.")
    print("Gercek veri seti uretimi icin 'kontrol.py' icindeki ProductionDatasetGenerator kullanilir.")

    # Cikis dizini olustur
    output_dir = "v5_dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Farkli egri turlerini test et
    print("\n▶ Surukleme indeksleri ile farkli egri turlerini test et:")
    for ctype in ['peaked', 'rising', 'falling']:
        # Surukleme indeksleri ile konfigurasyon olustur
        all_drag_indices = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300]
        max_drag = 150 if ctype == 'peaked' else 200

        selected_drag = sorted([d for d in all_drag_indices if d <= max_drag])
        selected_drag.reverse()  # Ust egri = 0, alt = maks
        config = ChartConfig(
            x_min=0.35, x_max=1.15,
            y_min=0.06, y_max=0.16,
            n_curves=min(8, len(selected_drag)),
            curve_type=ctype,
            curve_lw=random.uniform(0.3, 0.6),
            drag_indices=selected_drag[:min(8, len(selected_drag))],
            add_envelope_optimum=True,
            add_envelope_endurance=(ctype == 'peaked'),
            add_vmax_line=(ctype == 'falling')
        )

        full, mask, curves = draw_chart_matplotlib(config, W=800, H=600)

        # SIYAH arka planda renkli egriler (egitim hedefi)
        colored_target = colorize_curves_from_data(curves, config, W=800, H=600, show_axes=False, black_background=True)

        # Goruntuleri kaydet
        cv2.imwrite(os.path.join(output_dir, f'v5_{ctype}.png'), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'v5_{ctype}_target.png'), colored_target)

        # COCO aciklamasi olustur ve kaydet
        coco_data = generate_coco_annotation(curves, config, 1, f'v5_{ctype}.png', 800, 600)
        with open(os.path.join(output_dir, f'v5_{ctype}_annotation.json'), 'w') as f:
            json.dump(coco_data, f, indent=2)

        # Egri bilgisini yazdir
        print(f"\n  {ctype.upper()}:")
        print(f"     Egriler: {len(curves)}")
        for i, curve in enumerate(curves):
            print(f"       Egri {i}: surukleme_indeksi = {curve.drag_index}")

    # COCO aciklamalari ile rastgele ornekler olustur
    print("\n▶ Aciklamalarla rastgele ornekler olusturuluyor:")
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "curve"}]
    }

    annotation_id = 1

    for i in range(10):
        input_img, target_img, curves = make_sample(W=512, H=512, seed=i, add_artifacts=True)

        # Konfigurasyonu al (tohumla yeniden olusturulmasi gerekir)
        random.seed(i)
        np.random.seed(i)
        config = random_config()

        # Goruntuleri kaydet
        input_path = os.path.join(output_dir, f'v5_random_{i}_input.png')
        target_path = os.path.join(output_dir, f'v5_random_{i}_target.png')

        cv2.imwrite(input_path, input_img)
        cv2.imwrite(target_path, target_img)

        # COCO aciklamasi olustur
        coco_data = generate_coco_annotation(
            curves, config, i+1,
            os.path.basename(input_path),
            512, 512
        )

        # Veri setine ekle
        coco_dataset["images"].append(coco_data["image"])

        for ann in coco_data["annotations"]:
            ann["id"] = annotation_id
            coco_dataset["annotations"].append(ann)
            annotation_id += 1

        print(f"  [OK] v5_random_{i}: {len(curves)} egriler "
              f"(surukleme_indeksleri: {[c.drag_index for c in curves]})")

    # Tam COCO veri setini kaydet
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f, indent=2)

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print(f"Cikis dizini: {output_dir}/")
    print("\nOlusturulan dosyalar:")
    print("   • Giris goruntuleri: v5_random_*_input.png")
    print("   • Hedef goruntuler: v5_random_*_target.png")
    print("   • COCO aciklamalari: annotations.json")
    print("\nHer egrinin:")
    print("   • surukleme_indeksi: 0, 25, 50, 75, 100, 125, 150, 200, 250, 300")
    print("   • Sinirlayici kutu koordinatlari")
    print("   • Segmentasyon poligonu")
