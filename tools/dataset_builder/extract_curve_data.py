"""
Curve Data Extraction - From Segmentation Masks to Excel
========================================================

Uses Overlay Red Channel as Source of Truth for Exact User-Seen Center-Alignment.
"""

import os
import cv2
import math
import re
import sys
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Fix Turkish character issues on Windows terminal
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ============================================================================
# CONSTANTS
# ============================================================================

DRAG_INDEX_SEQUENCE = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400]
N_SAMPLE_POINTS = 40

# Speed of sound constants
SEA_LEVEL_SOS   = 661.48
SEA_LEVEL_TEMP  = 288.15
LAPSE_RATE      = 0.0019812
TROPOPAUSE_ALT  = 36089
STRATO_TEMP     = 216.65

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_fuel_flow(altitude: float, mach: float, specific_range: float) -> Optional[float]:
    if not all(v is not None and v != 0 for v in [altitude, mach, specific_range]):
        return None
    temp = (SEA_LEVEL_TEMP - LAPSE_RATE * altitude) if altitude <= TROPOPAUSE_ALT else STRATO_TEMP
    sos  = SEA_LEVEL_SOS * math.sqrt(temp / SEA_LEVEL_TEMP)
    tas  = mach * sos
    return round(tas / specific_range, 2)

def parse_filename_metadata(filename: str) -> Optional[Dict]:
    stem = Path(filename).stem
    m = re.match(r'^(\d+)-(\d+)-(\d+)lb$', stem, re.IGNORECASE)
    if m:
        return {'engine': int(m.group(1)), 'altitude': int(m.group(2)), 'weight': int(m.group(3))}
    nums = re.findall(r'\d+', stem)
    if len(nums) >= 3:
        return {'engine': int(nums[0]), 'altitude': int(nums[1]), 'weight': int(nums[2])}
    return None

def arc_length_sample(points: np.ndarray, n: int = 40) -> np.ndarray:
    if len(points) < 2:
        return np.tile(points[0], (n, 1)) if len(points) == 1 else np.zeros((n, 2))

    # Sort points primarily by X to ensure ordering for standard curves
    points = points[np.argsort(points[:, 0])]

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_dist[-1]
    if total == 0:
        return np.tile(points[0], (n, 1))
    sample_dists = np.linspace(0, total, n)
    sampled = []
    for d in sample_dists:
        idx = np.searchsorted(cum_dist, d, side='right') - 1
        idx = np.clip(idx, 0, len(points) - 2)
        seg_len = seg_lengths[idx]
        t = (d - cum_dist[idx]) / seg_len if seg_len > 0 else 0.0
        sampled.append(points[idx] + t * diffs[idx])
    return np.array(sampled)

# ============================================================================
# AXIS DETECTOR (OCR)
# ============================================================================

class AxisDetector:
    DEFAULT_AXIS = {'x_min': 0.20, 'x_max': 0.90, 'y_min': 0.01, 'y_max': 0.12}

    def __init__(self, tesseract_path: Optional[str] = None, allow_fallback: bool = True):
        self.allow_fallback = allow_fallback
        if not OCR_AVAILABLE:
            return

        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            for path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def find_grid_bounds(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]
        best, max_area = None, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if img_area * 0.1 < area < img_area * 0.95 and area > max_area:
                max_area, best = area, (x, y, w, h)
        return best

    @staticmethod
    def clean_number(text: str) -> Optional[float]:
        text = text.upper().replace('O', '0').replace('I', '1').replace('L', '1').replace(',', '.')
        clean = re.sub(r'[^\d.]', '', text)
        if not clean or clean == '.': return None
        if clean.count('.') > 1:
            parts = clean.split('.'); clean = parts[0] + '.' + parts[1]
        if clean.startswith('.'): clean = '0' + clean
        if clean.endswith('.'): clean = clean[:-1]
        try: return float(clean)
        except: return None

    @staticmethod
    def scan_strip(img_roi, config: str) -> List[Dict]:
        candidates = []
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                val = AxisDetector.clean_number(text)
                if val is not None:
                    cx = (data['left'][i] + data['width'][i]/2)/2
                    cy = (data['top'][i] + data['height'][i]/2)/2
                    candidates.append({'val': val, 'x': cx, 'y': cy})
        except: pass
        return candidates

    def get_axis_extremes(self, img_path: str) -> Tuple[float, float, float, float]:
        try:
            img = cv2.imread(img_path)
            if img is None: return self._fallback()
            h, w = img.shape[:2]
            rect = self.find_grid_bounds(img)
            if not rect: return self._fallback()

            gx, gy, gw, gh = rect
            cfg = r'--psm 6 -c tessedit_char_whitelist=0123456789.'

            # Y Axis
            roi_y = img[max(0, gy-60):gy+gh+5, max(0, gx-120):gx]
            y_cands = self.scan_strip(roi_y, cfg)
            valid_y = []
            for c in y_cands:
                val = c['val']/100.0 if 1.0 < c['val'] <= 60 else c['val']
                if 0.005 < val <= 0.6: valid_y.append({'val': val, 'gy': max(0, gy-60)+c['y']})

            y_min, y_max = None, None
            if valid_y:
                valid_y.sort(key=lambda k: k['gy'])
                y_max, y_min = valid_y[0]['val'], valid_y[-1]['val']
            if y_min is not None and y_max is not None and y_max < y_min: y_min, y_max = y_max, y_min
            if y_min is None and y_max is not None: y_min = 0.0

            # X Axis
            roi_x = img[gy+gh:min(h, gy+gh+80), max(0, gx-50):min(w, gx+gw+100)]
            x_cands = self.scan_strip(roi_x, cfg)
            valid_x = sorted([c for c in x_cands if 0.05 < c['val'] <= 4.0], key=lambda k: k['x'])
            if valid_x:
                x_min, x_max = valid_x[0]['val'], valid_x[-1]['val']
                if len(valid_x)>1 and x_max > 3.0 and valid_x[-2]['val'] < 3.0: x_max = valid_x[-2]['val']
            else: x_min, x_max = None, None

            if any(v is None for v in [x_min, x_max, y_min, y_max]): return self._fallback()
            return x_min, x_max, y_min, y_max
        except: return self._fallback()

    def _fallback(self, path=None):
        d = self.DEFAULT_AXIS
        return d['x_min'], d['x_max'], d['y_min'], d['y_max']

# ============================================================================
# CURVE EXTRACTION
# ============================================================================

class CurveExtractor:
    def __init__(self, x_min=0.2, x_max=0.9, y_min=0.01, y_max=0.12, use_ocr=True, tesseract_path=None, min_area=30):
        self.n_points = N_SAMPLE_POINTS
        self.min_area = min_area
        self.use_ocr = use_ocr
        self.axis_detector = AxisDetector(tesseract_path, allow_fallback=True)
        self.axis_detector.DEFAULT_AXIS = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    def extract(self, mask_path: str, original_path: str = None, debug_img_path: str = None, overlay_img_path: str = None) -> List[Dict]:
        """
        Extraction from Source of Truth:
        1. If overlay_img_path exists, use its RED channel as the master mask.
           This ensures points are ONLY on what the user sees in red.
        2. Otherwise, use mask_path (binary mask).
        """

        # Load visual truth if available, otherwise load binary mask
        source_path = overlay_img_path if (overlay_img_path and os.path.exists(overlay_img_path)) else mask_path
        img_raw = cv2.imread(source_path)
        if img_raw is None:
            # Fallback to mask if overlay failed to load
            img_raw = cv2.imread(mask_path)
            if img_raw is None: return []

        h_img, w_img = img_raw.shape[:2]

        # Determine the mask from the chosen source
        if len(img_raw.shape) == 3 and img_raw.shape[2] == 3:
            hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
            # If we are using an overlay, target RED. If we are using a 3ch binary mask, target WHITE.
            if np.max(hsv[:,:,1]) > 50: # Colored image detected
                lower1 = np.array([0, 100, 100]); upper1 = np.array([10, 255, 255])
                lower2 = np.array([160, 100, 100]); upper2 = np.array([180, 255, 255])
                mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
            else:
                gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY)

        x0, x1, y0, y1 = self.axis_detector.get_axis_extremes(original_path) if self.use_ocr and original_path else self.axis_detector._fallback()

        # Skeletonization for Precise Center Alignment
        if SKIMAGE_AVAILABLE:
            bool_mask = mask > 127
            skeleton = skeletonize(bool_mask)
            mask = (skeleton * 255).astype(np.uint8)

        # Connected Components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= (self.min_area / 4 if SKIMAGE_AVAILABLE else self.min_area):
                comps.append({'id': i, 'cy': centroids[i][1], 'area': area})

        # Prepare canvas for updated visualization (we draw dots ON the current overlay)
        debug_canvas = None
        viz_path = overlay_img_path if overlay_img_path else debug_img_path
        if viz_path and os.path.exists(viz_path):
            debug_canvas = cv2.imread(viz_path)
        elif original_path and os.path.exists(original_path):
            debug_canvas = cv2.imread(original_path)

        if debug_canvas is not None:
             cv2.putText(debug_canvas, f"X:[{x0:.2g}-{x1:.2g}] Y:[{y0:.3g}-{y1:.3g}]", (50, 50), 0, 0.7, (0, 0, 255), 2)

        if not comps:
            if debug_canvas is not None and viz_path: cv2.imwrite(viz_path, debug_canvas)
            return []

        comps.sort(key=lambda c: c['cy'])
        assigned_drag = DRAG_INDEX_SEQUENCE[:len(comps)]

        results = []
        for i, comp in enumerate(comps):
            ys, xs = np.where(labels == comp['id'])
            pts_pixels = np.column_stack((xs, ys))
            sampled = arc_length_sample(pts_pixels, n=self.n_points)

            drag_val = assigned_drag[i]
            pts = []
            # High contrast dots: Cyan or Yellow
            color = (0, 255, 255) if i % 2 == 0 else (255, 255, 0)

            for px, py in sampled:
                mach = x0 + (px/w_img)*(x1-x0)
                sr   = y1 - (py/h_img)*(y1-y0)
                pts.append((round(mach, 5), round(sr, 6)))
                if debug_canvas is not None:
                    cv2.circle(debug_canvas, (int(px), int(py)), 3, color, -1)
                    cv2.circle(debug_canvas, (int(px), int(py)), 4, (0, 0, 0), 1)

            if debug_canvas is not None and len(sampled) > 0:
                lbl = f"D:{drag_val}"
                lx, ly = int(sampled[0][0]), int(sampled[0][1])
                cv2.putText(debug_canvas, lbl, (lx+5, ly-5), 0, 0.6, (0,0,0), 4)
                cv2.putText(debug_canvas, lbl, (lx+5, ly-5), 0, 0.6, (255,255,255), 2)

            results.append({'drag_index': drag_val, 'points': pts})

        if debug_canvas is not None and viz_path:
            cv2.imwrite(viz_path, debug_canvas)

        return results

# ============================================================================
# EXCEL EXPORT
# ============================================================================

class ExcelExporter:
    SHEETS = {0: "Sea Level", 5000: "5,000 ft", 10000: "10,000 ft", 15000: "15,000 ft", 20000: "20,000 ft", 25000: "25,000 ft", 30000: "30,000 ft", 35000: "35,000 ft", 40000: "40,000 ft", 45000: "45,000 ft", 50000: "50,000 ft"}

    @staticmethod
    def export(all_data: List[Dict], output_path: str):
        engines = sorted(set(e['engine'] for e in all_data))
        for eng in engines:
            eng_data = [e for e in all_data if e['engine'] == eng]
            rows = []
            for entry in eng_data:
                for curve in entry['curves']:
                    for mach, sr in curve['points']:
                        rows.append({
                            'Altitude (ft)': entry['altitude'], 'Gross Weight (lb)': entry['weight'],
                            'Drag Index': curve['drag_index'], 'Mach Number (Ma)': mach,
                            'Specific Range (NM / lb)': sr, 'Fuel Flow (lb/h)': calculate_fuel_flow(entry['altitude'], mach, sr)
                        })

            if not rows: continue
            df_eng = pd.DataFrame(rows)
            suffix = "One_Engine" if eng == 1 else "Two_Engine"
            out_file = output_path.replace('.xlsx', f'_{suffix}.xlsx') if output_path.endswith('.xlsx') else f"{output_path}_{suffix}.xlsx"
            with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
                for alt in sorted(df_eng['Altitude (ft)'].unique()):
                    sh = ExcelExporter.SHEETS.get(int(alt), f"{int(alt)} ft")
                    df_eng[df_eng['Altitude (ft)'] == alt].to_excel(writer, sheet_name=sh, index=False)
            print(f"   [OK] Saved: {out_file}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_dir', required=True)
    parser.add_argument('--original_dir', required=True)
    parser.add_argument('--output', default='curve_data.xlsx')
    parser.add_argument('--x_min', type=float, default=0.20)
    parser.add_argument('--x_max', type=float, default=0.90)
    parser.add_argument('--y_min', type=float, default=0.01)
    parser.add_argument('--y_max', type=float, default=0.12)
    parser.add_argument('--min_area', type=int, default=30)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        Path("debug_extraction").mkdir(exist_ok=True)
        print("[INFO] Debug folder 'debug_extraction' checked/created.")

    print(f"\n--- CURVE DATA EXTRACTION ---")

    seg_dir = Path(args.segmentation_dir)
    if not seg_dir.exists():
        print(f"[Error] Dir not found: {args.segmentation_dir}")
        return

    actual_seg_dir = seg_dir
    if (seg_dir / "segmentation").exists():
        actual_seg_dir = seg_dir / "segmentation"
        print(f"[INFO] Source masks: {actual_seg_dir}")
    elif seg_dir.name == "overlay" and (seg_dir.parent / "segmentation").exists():
        actual_seg_dir = seg_dir.parent / "segmentation"

    mask_files = sorted([f for f in actual_seg_dir.iterdir() if f.suffix.lower() == '.png'])
    if not mask_files:
        for sub in ["segmentation", "overlay"]:
            if (seg_dir / sub).exists():
                tmp_dir = seg_dir / sub
                mask_files = sorted([f for f in tmp_dir.iterdir() if f.suffix.lower() == '.png'])
                if mask_files:
                    actual_seg_dir = tmp_dir
                    break

    if not mask_files:
        print(f"[Error] No PNG masks found in {actual_seg_dir}")
        return

    extractor = CurveExtractor(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        min_area=args.min_area
    )
    all_data = []

    overlay_dir = None
    if (seg_dir / "overlay").exists():
        overlay_dir = seg_dir / "overlay"
    elif actual_seg_dir.name == "segmentation" and (actual_seg_dir.parent / "overlay").exists():
        overlay_dir = actual_seg_dir.parent / "overlay"

    for mf in mask_files:
        meta = parse_filename_metadata(mf.name)
        if not meta and '_segmentation' in mf.name:
            meta = parse_filename_metadata(mf.name.replace('_segmentation', ''))
        if not meta: continue

        orig = None
        for ext in ['.png', '.jpg', '.jpeg']:
            for stem_opt in [mf.stem, mf.stem.replace('_segmentation', '')]:
                cand = Path(args.original_dir) / (stem_opt + ext)
                if cand.exists():
                    orig = str(cand)
                    break
            if orig: break

        overlay_path = None
        if overlay_dir:
            cand_overlay = overlay_dir / mf.name
            if cand_overlay.exists():
                overlay_path = str(cand_overlay)

        debug_path = f"debug_extraction/{mf.stem}_debug.jpg" if args.debug else None

        print(f"[Process] {mf.name}", end=" ", flush=True)
        if overlay_path:
            print(f"(Visual Sync: OK)", end=" ", flush=True)
        try:
            curves = extractor.extract(str(mf), orig, debug_img_path=debug_path, overlay_img_path=overlay_path)
            if curves:
                all_data.append({'altitude': meta['altitude'], 'weight': meta['weight'], 'engine': meta['engine'], 'curves': curves})
                print(f"-> {len(curves)} curves.")
            else:
                print("-> 0 curves found.")
        except Exception as e:
            print(f"-> [Error] {e}")

    if all_data:
        ExcelExporter.export(all_data, args.output)
        print("\n--- COMPLETED ---")
    else:
        print("\n[Warning] No data extracted.")

if __name__ == '__main__':
    main()
