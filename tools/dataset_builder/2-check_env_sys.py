"""
====================================================================
ORTAM SISTEM KONTROL SCRIPT'I (Environment System Check Script)
====================================================================

AMAC (Purpose):
    Bu script, projenin calismasi icin gerekli olan Python paketlerinin
    sisteminizde dogru sekilde kurulu olup olmadigini kontrol eder.

    This script checks whether all required Python packages for the
    project are properly installed on your system.

KONTROL EDILEN PAKETLER (Checked Packages):
    - pdf2image:       PDF dosyalarini goruntuye donusturme (Convert PDF to images)
    - pytesseract:     Goruntulerden metin cikarma / OCR (Text extraction from images)
    - pandas:          Veri isleme ve analiz (Data processing and analysis)
    - openpyxl:        Excel dosyalari ile calisma (Work with Excel files)
    - torch:           PyTorch - Derin ogrenme kutuphanesi (Deep learning library)

KULLANIM (Usage):
    Terminal/PowerShell'den su komut ile calistirin:

    python check_env_sys.py

CIKTI ANLAMLANDIRMASI (Output):
    - "OK" yazisi = Paket basariyla kurulu ve kullanilabilir
    - "MISSING" yazisi = Paket eksik, kurulum gerekli

EKSIK PAKET KURMA (Installing Missing Packages):
    pip install pdf2image pytesseract pandas openpyxl torch

    Not: torch paketi buyuk olabilir. GPU destegi icin:
    https://pytorch.org/get-started/locally/
====================================================================
"""

modules = ['pdf2image','pytesseract','pandas','openpyxl','torch']
import importlib
for m in modules:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'MISSING', e)

# ====================================================================
# TURKISH CHARACTER REPLACEMENT UTILITY
# ====================================================================

import os
import re

def replace_turkish_chars(text):
    """Replace Turkish characters with ASCII equivalents"""
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }

    for turkish, ascii_char in replacements.items():
        text = text.replace(turkish, ascii_char)

    return text

def process_file_for_turkish_chars(filepath):
    """Process a single Python file to replace characters"""
    print(f"Processing: {filepath}")

    try:
        # Read with UTF-8
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace Turkish characters
        new_content = replace_turkish_chars(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  ✓ Done")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def fix_turkish_characters_in_project():
    # Use current directory instead of hardcoded path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    python_files = [
        '2-check_env_sys.py',
        '3-synthetic_production.py',
        '5-segment_curves.py',
        'extract_curve_data.py',
        'synthetic_data_gui.py',
        'synthetic_main.py',
        'train_unet.py'
    ]

    print("\\n" + "=" * 60)
    print("TURKISH CHARACTER REPLACEMENT TOOL")
    print("=" * 60)
    print()

    success_count = 0
    for filename in python_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            if process_file_for_turkish_chars(filepath):
                success_count += 1
        else:
            print(f"Skipping: {filename} (not found)")

    print()
    print("=" * 60)
    print(f"Completed: {success_count}/{len(python_files)} files processed")
    print("=" * 60)

if __name__ == '__main__':
    # Add a prompt to run turkish char fixing
    if len(os.sys.argv) > 1 and os.sys.argv[1] == '--fix-chars':
        fix_turkish_characters_in_project()
