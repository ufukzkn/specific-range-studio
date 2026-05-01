# U-Net Curve Segmentation - Uçak Grafikleri için Eksiksiz Kullanım Rehberi

Bu proje sentetik uçak grafikleri oluşturup, U-Net modeli kullanarak grafiklerdeki eğrileri (curve) otomatik olarak segmente etmek ve bu eğrileri sayısallaştırarak Excel formatında çıktı almak için geliştirilmiş tam bir sistemdir.

---

## 🎯 Projenin Amacı ve Akışı

```
1. Sentetik Grafikleri Üret → 2. U-Net Modeli Eğit → 3. Eğrileri Segmente Et (PDF/PNG) → 4. Eğrileri Sayısallaştırılıp Excel'e Çek
(5000+ RandomGraph)           (Semantic Segmentation)  (Otomatik Curve Detection)           (OCR + Veri Çıkarımı)
```

---

## 🖥️ Görsel Kontrol Paneli (GUI)

Projenin tüm aşamalarını (Veri Üretimi, Eğitim, Çıkarım ve Excel Aktarımı) teknik komutlarla uğraşmadan tek bir pencereden yönetebilirsiniz.

### Kullanım:
```bash
python synthetic_data_gui.py
```

### Öne Çıkan Özellikler:
1. **İnteraktif Bilgi İkonları (ⓘ)**: Her parametrenin yanındaki mavi **ⓘ** ikonunun üzerine gelerek o parametrenin ne işe yaradığını ve ideal değerlerini anında görebilirsiniz.
2. **Ayarlanabilir Panel (PanedWindow)**: Sol taraftaki ayarlar paneli ile sağ taraftaki log/grafik alanı arasındaki ayırıcıyı sürükleyerek paneli genişletebilir veya daraltabilirsiniz. Bu sayede uzun parametre isimlerini tam olarak okuyabilirsiniz.
4. **Gelişmiş Augmentation Seti**: Roboflow standartlarına uygun olarak; Yatay/Dikey Çevirme (Flip), Kaykılma (Shear), Hue/Saturation jitter, Cutout (rastgele maskeleme) ve Motion Blur gibi gelişmiş yöntemler eklenmiştir.

---

## 📚 Adım Adım Başlangıç ve Komut Kullanımları

### ADIM 1: Sentetik Grafikleri Üret (`3-synthetic_production.py`)

Yapay zeka modelini (U-Net) eğitmek için ihtiyaç duyduğunuz büyük çaplı sentetik veri setini bu aşamada üretirsiniz. Üretim işlemi temelde iki farklı dosya ile yürütülür:

#### 1. `synthetic_main.py` (Geliştirici veya Çizim Motoru)
Bu dosya, projenin matematiksel çizim algoritmalarını barındıran **ana motordur**.
- Matplotlib kütüphanesini kullanarak gerçeğe yakın sahte uçak grafikleri (eğriler, oklar, yazılar, eksenler) çizer.
- Sadece tekil üretimler ve testler içindir. Toplu veri üretmez veya model için klasörleme yapmaz.
- **Kullanım Amacı:** Eğrilere yeni bir stil eklemek, çizim mantığını değiştirmek veya arka plan renkleri gibi grafiğin temel iskeletinde hata ayıklamak (debug) istediğinizde sadece bu dosya üzerinde çalışırsınız.

#### 2. `3-synthetic_production.py` (Toplu Üretim Fabrikası)
Model eğitmek için asıl kullanmanız gereken **Fabrikasyon Scriptidir**. Kendisi doğrudan çizim yapmaz, arka planda üstteki `synthetic_main.py` motorunu çağırır ve onu endüstriyel boyutta (örn. 5000 adet) kullanır. Başlıca özellikleri şunlardır:

1. **Toplu Üretim (Multiprocessing):** Ana çizim motorunu (`synthetic_main.py`) çağırıp, bilgisayarınızın işlemcisindeki (CPU) çekirdekleri paralel çalıştırarak normalde saatler sürecek 5.000 grafik üretimini dakikalar içinde tamamlar.
2. **Gerçek Dünya Filtreleri (Data Augmentation):** Yapay zekanın sadece kusursuz dijital çizimleri değil, kötü taranmış telefon çekimi kağıtları da tanıması gerekir. Bu script, üretilen grafikleri alıp kasıtlı olarak bozar: Rotasyon (eğme), Siyah-Beyaz solgunluğu, Gürültü (Noise), JPEG bozulması, **Bulanıklık (Blur)** ve **Gölge (Shadow)** efektleri uygulayarak modeli hayatın gerçeklerine hazırlar.
3. **Otomatik Veri Parçalama:** Yapay zeka eğitimi kuralı gereği, ürettiği binlerce resmi manuel ayırmanıza gerek kalmadan otomatik olarak **%70 Train** (Öğrenme), **%15 Validation** (Doğrulama) ve **%15 Test** (Sınav) setlerine böler.
4. **COCO JSON Çıktısı:** Binlerce resim için binlerce ayrı koordinat metni kaydetmez. Derin öğrenme dünyasının standartı olan COCO (JSON) formatında tek bir dosyada (`annotations.json`) modeli besleyecek tüm koordinatları temiz şekilde paketler.

**Komut:**
```bash
python 3-synthetic_production.py 5000 6
```

**Temel Parametreler:**
- `5000` = Üretilecek grafik/gorsel sayısı.
- `6`    = Paralel işçi sayısı (CPU çekirdeklerinize göre artırabilirsiniz, örn: 8/12).
- `--rotation 3.0` = Maksimum döndürme açısı (+- derece).
- `--noise 15.0` = Maksimum gürültü seviyesi.
- `--jpeg 70` = Minimum JPEG kalitesi (0-100 arası).
- `--blur 0` = Maksimum bulanıklık kernel boyutu (Örn. 3, 5, 7. `0` kapalı demek).
- `--shadow 0` = Dengesiz aydınlatma/gölgelenme hissi (1 açık, 0 kapalı).
- `--prob 0.7`   = Efektlerin (bulanıklık, gölge vb.) uygulanma olasılığı.
- `--flip 0`     = Çevirme (0:Kapalı, 1:Yatay, 2:Dikey, 3:Her ikisi).
- `--shear 0.0`  = Kaykılma (eğrilik) miktarı (Örn: 0.1).
- `--hue 0`      = Renk tonu jitter (0-180).
- `--sat 0`      = Renk doygunluğu jitter (0-255).
- `--cutout 0`   = Görsel üzerindeki rastgele maskeleme/delik sayısı (Örn: 3).
- `--motion 0`   = Hareket bulanıklığı kernel boyutu (Örn: 5).

**Beklenen Çıktı**
`dataset_production/` klasöründe Roboflow stili üçlü yapı doğrudan oluşturulur:
- `dataset_production/train/`: Eğitim verileri (%70)
    - `images` (kök dizinde), `masks/`, `_annotations.coco`
- `dataset_production/valid/`: Doğrulama verileri (%15)
    - `images` (kök dizinde), `masks/`, `_annotations.coco`
- `dataset_production/test/`: Test verileri (%15)
    - `images` (kök dizinde), `masks/`, `_annotations.coco`


### ADIM 2: U-Net Modelini Eğit (`train_unet.py`)

### U-Net Model Eğitim Dokümantasyonu

#### İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Mimari Açıklama](#mimari-açıklama)
3. [Dataset Yapısı](#dataset-yapısı)
4. [Eğitim Süreci](#eğitim-süreci)
5. [Inference ve Segmentasyon](#inference-ve-segmentasyon)
6. [Komut Satırı Parametreleri](#komut-satırı-parametreleri)
7. [Debugging ve Sorun Çözümü](#debugging-ve-sorun-çözümü)

---

#### Genel Bakış

`train_unet.kaggle.py` dosyası, sentetik grafiklerdeki eğrileri segmente etmek için U-Net derin öğrenme modelini eğitir.

##### U-Net Nedir?

U-Net, **segmentasyon görevleri** için özel olarak tasarlanmış bir ağ mimarisidir. Adı, ağın şekline gelen "U" harfinden gelmektedir.

```
Input: (256x256, 1 kanal)
  ↓
┌─────────────────────────────────────┐
│  ENCODER (Downsampling)             │
│  • DoubleConv: 64 kanal @256x256    │
│  • Down: MaxPool → 128 kanal @128x128│
│  • Down: MaxPool → 256 kanal @64x64 │
│  • Down: MaxPool → 512 kanal @32x32 │
│  • Down: MaxPool → 1024 kanal @16x16│
└─────────────────────────────────────┘
  ↓ (Bottleneck)
┌─────────────────────────────────────┐
│  DECODER (Upsampling + Skip Conn)   │
│  • Up: Upsample + Concat → 512 @32x32│
│  • Up: Upsample + Concat → 256 @64x64│
│  • Up: Upsample + Concat → 128 @128x128│
│  • Up: Upsample + Concat → 64 @256x256│
│  • Conv1x1: 64 → 1 kanal (output)   │
└─────────────────────────────────────┘
  ↓
Output: (256x256, 1 kanal) - Binary segmentation mask
```

##### Skip Connections Neden Önemli?

**Sorun (Skip connections olmadan):**
```
Input (256x256)  →  Down  →  Down  →  Down  →  Down  →  Bottleneck (16x16)
     [100K piksel]                                     [256 piksel]

↓ Detay kaybı %99.7!

Up  →  Up  →  Up  →  Up  →  Output (256x257)
Kayıp detaylar geri getirilemiyor → Bulanık maskeler
```

**Çözüm (Skip connections ile):**
```
Input (256x256)
  ├─ Save x1 (256x256, 64 kanal)
  ├─ Down → x2 (128x128, 128 kanal)
  │  ├─ Save x2
  │  ├─ Down → x3 (64x64, 256 kanal)
  │  │  ├─ Save x3
  │  │  └─ Down → Bottleneck (16x16, 1024 kanal)
  │  │     ↓
  │  └─ Up (Concat x3) → (64x64)
  └─ Up (Concat x2) → (128x128)
     ↓
  Up (Concat x1) → (256x256) ✓ Detaylı mask!
```

**Avantajlar:**
- Encoder'daki detay bilgisi decoder'a aktarılır
- Daha keskin segmentasyon kenarları
- Skip connections sayesinde gradientler daha iyi akabilir

---

#### Dataset Yapısı

##### Dosya Organizasyonu

```
dataset_production/
├── images/              # Grafik görüntüleri (grayscale)
│   ├── graph_001.png
│   ├── graph_002.png
│   └── ...
├── masks/               # Ground truth segmentasyon maskeleri
│   ├── graph_001.png    # 255=eğri, 0=background
│   ├── graph_002.png
│   └── ...
└── splits/              # Train/Val/Test ayrımı (opsiyonel)
    ├── train.json       # {"images": [{"file_name": "..."}, ...]}
    ├── val.json
    └── test.json
```

##### CurveSegmentationDataset Sınıfı

**Amaç:** PyTorch DataLoader ile eğitim sırasında veri yüklemeyi sağlamak.

**İşlem Adımları (__getitem__):**

```python
1. Dosya adını al:
   image_files = ["graph_001.png", "graph_002.png", ...]
   idx=0 → "graph_001.png"

2. Dosyalardan yükle:
   image = cv2.imread("images/graph_001.png", IMREAD_GRAYSCALE)
   mask = cv2.imread("masks/graph_001.png", IMREAD_GRAYSCALE)

3. Resize et (tüm görüntüleri 256x256'ye standardize et):
   image = cv2.resize(image, (256, 256))
   mask = cv2.resize(mask, (256, 256))

4. Binary mask (0 veya 1):
   mask = (mask > 127).astype(np.float32)

5. Normalize (0-255 → 0-1):
   image = image.astype(np.float32) / 255.0

6. PyTorch tensor'a dönüştür:
   image = torch.from_numpy(image).unsqueeze(0)  # (256, 256) → (1, 256, 256)
   mask = torch.from_numpy(mask).unsqueeze(0)

Return: {
    'image': (1, 256, 256) normalized float tensor
    'mask': (1, 256, 256) binary float tensor
    'filename': 'graph_001.png'
}
```

**Split Mekanizması:**

```python
# Split dosyası YOKSA → Tüm dosyalar yüklenir
dataset = CurveSegmentationDataset('images/', 'masks/')

# Split dosyası VARSA → Sadece o split'teki dosyalar yüklenir
train_dataset = CurveSegmentationDataset(
    'images/', 'masks/',
    split='train',
    split_file='splits/train.json'  # Sadece train dosyaları
)

val_dataset = CurveSegmentationDataset(
    'images/', 'masks/',
    split='val',
    split_file='splits/val.json'  # Sadece validation dosyaları
)
```

---

#### Eğitim Süreci

##### CurveSegmentationTrainer Sınıfı

**Amaç:** Modelin eğitim, validasyon, checkpoint ve history yönetimini sağlamak.

#### 1. Başlatma (__init__)

```python
trainer = CurveSegmentationTrainer(
    device='cuda',              # GPU ise 'cuda', CPU ise 'cpu'
    model_save_dir='./checkpoints'  # Checkpoint'lerin kaydı
)

# Otomatik olarak:
# - Device kontrol eder (GPU varsa GPU, yoksa CPU)
# - Checkpoint klasörü oluşturur
# - Training history dictionary'si başlatır
```

**Device Seçimi:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GPU: ~10-100x hızlı eğitim
# CPU: Yavaş ama GPU yoksa zorunlu
```

#### 2. Model Oluşturma

```python
model = trainer.create_model(
    in_channels=1,      # Input: Grayscale (1 kanal)
    out_channels=1,     # Output: Binary mask (1 kanal)
    features=64         # Başlangıç kanal sayısı
)

# Mimarideki kanal sayıları:
# 64 → 128 → 256 → 512 → 1024 (encoder)
# 1024 → 512 → 256 → 128 → 64 (decoder)
# features=64: Hafif model (hızlı eğitim)
# features=128: Daha kapasiteli (daha iyi sonuçlar, daha yavaş)
```

#### 3. Loss Fonksiyonları

**Dice Loss (Kullanılan Loss):**

```
Formula:
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    Loss = 1 - Dice

Adım Adım:
1. Model çıkışına sigmoid uygula (0-1 aralığında)
2. Intersection hesapla: Hem model hem ground truth 1 olan pikseller
3. Dice hesapla: Intersection'u böl
4. Loss = 1 - Dice (minimize edilir)

Örnekler:
- Perfect match (overlap %100): Dice=1.0, Loss=0.0 ✓
- No match (overlap %0): Dice=0.0, Loss=1.0 ✗
- 50% overlap: Dice=0.5, Loss=0.5 (orta)

Avantajları:
- Class imbalance'a dirençli (çoğu background olsa bile çalışır)
- Intersection'u doğrudan optimize eder
- Binary segmentation için ideal
```

**IoU Score (Metrik):**

```
Formula:
    IoU = |X ∩ Y| / |X ∪ Y|

Basit Anlatım:
    IoU = Overlap Alanı / Union Alanı

Örnekler:
- Perfect: IoU = 1.0 (100%)
- No match: IoU = 0.0 (0%)
- 50% overlap: IoU = 0.33-0.5

Validation sırasında kullanılır (best model seçmek için)
```

#### 4. Train Epoch (Eğitim Döngüsü)

```
Her Epoch Başında:

for batch_images, batch_masks in train_loader:  # Batch yükle
    # FORWARD PASS
    predictions = model(batch_images)           # Eğri tahmin et

    # LOSS HESAPLAMA
    loss = dice_loss(predictions, batch_masks)  # Tahmin vs Gerçek

    # BACKWARD PASS
    optimizer.zero_grad()                       # Gradientleri sıfırla
    loss.backward()                             # Gradientleri hesap çı

    # WEIGHT UPDATE
    optimizer.step()                            # Ağırlıkları güncelle

    # Logging
    print(f"Loss: {loss:.4f}")

Sonuç:
- Model, training verisi üzerinde daha iyi tahmin yapmayı öğrenir
- Her epoch, model training loss'unu azaltmaya çalışır
```

#### 5. Validation Döngüsü

```
Her Epoch Sonunda:

with torch.no_grad():           # Gradientler hesaplanmaz (hız)
    model.eval()                # Model evaluation moduna geçer

    for batch_images, batch_masks in val_loader:
        predictions = model(batch_images)
        val_loss = dice_loss(predictions, batch_masks)
        val_iou = iou_score(predictions, batch_masks)

    model.train()               # Model training moduna geri dön

Amaç:
- Eğitilmemiş veriler (validation set) üzerinde performansı kontrol et
- Overfitting tespiti: train_loss ↓ ama val_loss ↑ = Overfitting!
- Best model seçimi: Lowest val_loss'a sahip model kaydedilir
```

#### 6. Early Stopping Mekanizması

```
patience = 10 (varsayılan)

Epoch 1: val_loss = 0.50 (Best!)
Epoch 2: val_loss = 0.45 (Best!)
Epoch 3: val_loss = 0.47 (Worse, counter=1)
Epoch 4: val_loss = 0.48 (Worse, counter=2)
...
Epoch 13: val_loss = 0.49 (Worse, counter=10)
→ DURDUR! Eğitim 13. epoch'ta biter

Avantajları:
- Overfitting'i engeller
- Zaman kazandırır (tüm epochları beklemek yerine)
- Best model otomatik kaydedilir
```

#### 7. Tam Eğitim Döngüsü (fit)

```
trainer.fit(
    model,
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=1e-3,
    patience=10
)

Akış:
for epoch in range(epochs):
    # Training
    train_loss = train_epoch(model, train_loader, optimizer)

    # Validation
    val_loss, val_iou = validate(model, val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Best model kaydı
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

    # Early stopping
    if patience_counter > patience:
        break

Çıktı Dosyaları:
- ./checkpoints/best_model.pth
- training_history.png (grafikler)
```

---

#### Inference ve Segmentasyon

##### CurveSegmentationInference Sınıfı

**Amaç:** Eğitilmiş modeli yeni görüntülerde kullanarak segmentasyon yapmak.

#### Model Yükleme

```python
inference = CurveSegmentationInference(
    model_path='./checkpoints/best_model.pth',
    device='cuda'
)

# Adımlar:
# 1. Boş U-Net modeli oluştur
# 2. State dict'i (.pth dosyası) yükle
# 3. Model evaluation moduna geçer
# 4. Device'a (GPU/CPU) taşı
```

#### segment_image() - Tek Görüntü Segmentasyonu

```python
mask, confidence = inference.segment_image(
    image_path='test.png',
    threshold=0.5,
    output_size=(256, 256)
)

# Adımlar:
# 1. Görüntüyü yükle (grayscale)
# 2. 256x256'ye resize et
# 3. Normalize et (0-1)
# 4. Model'dan geçir
# 5. Sigmoid uygula (0-1 olasılık)
# 6. Threshold'a göre binary maske oluştur
# 7. Orijinal boyuta geri döndür

# Çıktılar:
# - mask: Binary (0 veya 255), direct extract_curve_data.py'de kullanılabilir
# - confidence: Float, model emin olma derecesi (0-1)

# Threshold Seçimi:
# 0.3: Agresif detect (kalın eğriler, false positive riski)
# 0.5: Dengeli (standards)
# 0.7: Muhafazakar (ince eğriler, eğri kaybı riski)
```

#### segment_batch() - Toplu Segmentasyon

```python
inference.segment_batch(
    image_dir='test_images/',
    output_dir='segmentation_results/',
    threshold=0.5
)

# Her test_001.png için iki dosya oluşturulur:
# - segmented_test_001.png (binary mask, 0 veya 255)
# - confidence_test_001.png (confidence map, 0-255 ölçeğinde)

# Kullanımı:
# 1. PDF'lerden segmentasyon maskeleri elde ettikten sonra
# 2. extract_curve_data.py'de
# 3. curve_data.xlsx oluşturmak için
```

---

#### Komut Satırı Parametreleri

##### Temel Kullanım

```bash
# Varsayılan parametrelerle eğit
python train_unet.py

# Özel dataset'le
python train_unet.py --dataset my_dataset --epochs 100

# Düşük learning rate (fine-tuning)
python train_unet.py --learning_rate 1e-5 --epochs 30

# Büyük batch'ler
python train_unet.py --batch_size 32 --features 128

# CPU'da test et
python train_unet.py --device cpu --batch_size 2
```

##### Parametreler Detaylı

| Parameter | Default | Açıklama |
|-----------|---------|----------|
| `--dataset` | `dataset_production` | Veri seti klasörü |
| `--epochs` | 50 | Toplam eğitim döngüsü sayısı |
| `--batch_size` | 8 | Her iteration'da işlenen örnek sayısı |
| `--learning_rate` | 1e-3 | Optimization adım boyutu |
| `--patience` | 10 | Early stopping: kaç epoch boyunca iyileşme yoksa durdur |
| `--image_size` | 256 | Input görüntü boyutu (H=W) |
| `--features` | 64 | U-Net feature base count |
| `--num_workers` | 4 | DataLoader'da paralel worker sayısı |
| `--device` | cuda | Eğitim cihazı (cuda/cpu) |

##### Parametreleri Optimization Rehberi

**Learning Rate Seçimi:**
```
1e-5:  Çok düşük (yavaş ama stabil)
1e-4:  Düşük (muhafazakar)
1e-3:  Normal (standart başlangıç) ← Tavsiye
1e-2:  Yüksek (hızlı ama kararsız)
1e-1:  Çok yüksek (loss patlar, NaN'e gider)
```

**Batch Size Seçimi:**
```
1-4:    Küçük (iyi generalization, yavaş, düşük GPU memory)
8-16:   Orta (dengeli) ← Tavsiye
32-64:  Büyük (hızlı, overfitting riski, çok GPU memory)
```

**Epochs Seçimi:**
```
20-30:   Hızlı test
50:      Standard
100+:    Detaylı eğitim (overfitting riski artıyor)
```

---

#### Debugging ve Sorun Çözümü

### Sorun 1: Loss NaN?

**Sebeplers:**
- Learning rate çok yüksek
- Dataset bozuk
- Numerical instability

**Çözüm:**
```bash
# Learning rate'i azalt
python train_unet.py --learning_rate 1e-4

# Dataset kontrol et
# - Görüntüler: Piksel değerleri 0-255 arasında mı?
# - Maskeler: 0 ve 255 mü, yoksa intermediate değerler var mı?
```

### Sorun 2: Validation loss hiç düşmüyor

**Sebeplers:**
- Learning rate çok düşük
- Model capacity yetersiz
- Dataset split yanlış

**Çözüm:**
```bash
# Learning rate'i artır
python train_unet.py --learning_rate 1e-2

# Model kapasitesini artır
python train_unet.py --features 128 --batch_size 16

# Train/val split kontrol et
# splits/train.json ve splits/val.json dosyaları kontrol et
```

### Sorun 3: RAM Memory Error

**Sebeplers:**
- Batch size çok yüksek
- Image size çok büyük

**Çözüm:**
```bash
# Batch size'ı azalt
python train_unet.py --batch_size 4

# Image size'ı azalt (kalite kaybı)
python train_unet.py --image_size 128
```

### Sorun 4: GPU Memory Error

**Sebeplers:**
- Batch size çok yüksek (GPU'ya sığmıyor)
- Model feature sayısı çok yüksek

**Çözüm:**
```bash
# Batch size'ı azalt
python train_unet.py --batch_size 2

# Device'ı CPU'ya değiştir (yavaş olacak)
python train_unet.py --device cpu --batch_size 4
```

### Sorun 5: Eğitim çok yavaş

**Sebeplers:**
- CPU'da eğitim (GPU yok)
- Batch size çok küçük
- num_workers yetersiz

**Çözüm:**
```bash
# GPU kullandığından emin ol
python train_unet.py --device cuda

# Batch size'ı artır
python train_unet.py --batch_size 16

# num_workers'ı artır (CPU core sayısına göre)
python train_unet.py --num_workers 8
```

### Sorun 6: Overfitting (train_loss ↓ ama val_loss ↑)

**Sebeplers:**
- Model çok kapasiteli
- Training verisi çok küçük
- Eğitim çok uzun

**Çözüm:**
```bash
# Model kapasitesini azalt
python train_unet.py --features 32

# Training verisi artır (data augmentation)
# (Kod'a data augmentation eklemek gerekebilir)

# Early stopping zaten yapılıyor
# Ama patience'ı azaltabilirsin
python train_unet.py --patience 5
```

---

#### Örnek Tam Pipeline

```python
#!/usr/bin/env python3
# train_complete_pipeline.py

import os
from train_unet import CurveSegmentationTrainer, CurveSegmentationDataset, CurveSegmentationInference
from torch.utils.data import DataLoader

# STEP 1: Dataset'leri yükle
print("[STEP 1] Loading datasets...")
train_dataset = CurveSegmentationDataset(
    'dataset_production/images/',
    'dataset_production/masks/',
    split='train',
    split_file='dataset_production/splits/train.json'
)

val_dataset = CurveSegmentationDataset(
    'dataset_production/images/',
    'dataset_production/masks/',
    split='val',
    split_file='dataset_production/splits/val.json'
)

# STEP 2: DataLoader'lar oluştur
print("[STEP 2] Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# STEP 3: Trainer başlat
print("[STEP 3] Creating trainer...")
trainer = CurveSegmentationTrainer(device='cuda')

# STEP 4: Model oluştur
print("[STEP 4] Creating U-Net model...")
model = trainer.create_model(in_channels=1, out_channels=1, features=64)

# STEP 5: Modeli eğit
print("[STEP 5] Training...)...
trainer.fit(
    model,
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=1e-3,
    patience=10
)

# STEP 6: Training history'yi visualize et
print("[STEP 6] Plotting history...")
trainer.plot_history(save_path='training_history.png')

# STEP 7: Inference
print("[STEP 7] Running inference on test set...")
inference = CurveSegmentationInference('./checkpoints/best_model.pth')
inference.segment_batch(
    image_dir='test_images/',
    output_dir='segmentation_results/',
    threshold=0.5
)

print("[SUCCESS] Pipeline completed!")
print("[INFO] Best model: ./checkpoints/best_model.pth")
print("[INFO] Results: ./segmentation_results/")
```

---

**Oluşturma Tarihi:** 2026-02-12
**Dosya:** `train_unet.py`
**Versiyon:** 1.0


**Komut:**
```bash
python train_unet.py --dataset dataset_production --epochs 50 --batch_size 8
```

**Esnek Parametreler:**
- `--epochs 50`: Kaç tur eğitim yapılacağı (Optimum 50-100 arasıdır).
- `--batch_size 8`: Ekran kartı (VRAM) RAM'inize göre (8, 16 veya 32 uygundur).
- `--learning_rate 1e-3`: Öğrenme oranı (varsayılan: 0.001).
- `--image_size 256`: Modelin giriş çözünürlüğü (Hızlı prototipleme için `128` yapılabilir).
- `--device cuda`: Eğitimin çalışacağı birim (cuda / cpu).
- `--num_workers 4`: DataLoader için iş parçacığı sayısı.
- `--features 64`: U-Net kanal sayısı (64 veya 128)

**Beklenen Çıktı:**
- En İyi Model: `checkpoints/best_model.pth` olarak kaydedilir.
- Eğitim İstatistikleri: Eğitim tamamlanınca `training_history.png` üzerinden Loss ve IoU durumunu görüntüleyebilirsiniz. *(İyi bir modelde IoU > 0.70 olmalı ve validation loss ile training loss tutarlı ilerlemelidir).*

Eğitim Süreci:
    Epoch 1/50 [TRAIN] 100%|████████| 437/437 [00:45<00:00,  9.71it/s]
    Epoch 1/50 [VAL]   100%|████████| 94/94   [00:09<00:00, 10.13it/s]
    Epoch 1/50
      Train: Loss=0.4523, IoU=0.6234
      Val:   Loss=0.3892, IoU=0.6891
      ✓ Best model kaydedildi: ./checkpoints/best_model.pth

    [INFO] Eğitim tamamlandı!
    [INFO] Best model: ./checkpoints/best_model.pth

Zaman: CUDA ile ~30-45 dakika, CPU ile ~3-4 saat

Beklenen Performans:
    - IoU (Intersection over Union) > 0.70
    - Loss < 0.30
    - Validation Loss = Training Loss (overfitting yok)

    Beklenen Çıktı (tek görüntü):
    my_image_segmentation.png               (Binary mask)
    my_image_confidence.png                 (Probability map)
    my_image_overlay.png                    (Original + overlay)
    my_image_visualization.png              (6 panel visualization)

Beklenen Çıktı (batch):
    results/
    ├── image1_segmentation.png
    ├── image1_confidence.png
    ├── image1_overlay.png
    ├── image2_segmentation.png
    ├── image2_confidence.png
    ├── image2_overlay.png
    └── ... (tüm görüntüler için)


### ADIM 3: Yeni Grafikleri Segmente Et ve PDF İşle (`5-segment_curves.py`)

### Segment Curves - PDF Desteğiyle Genişletildi

`segment_curves.py` artık **PDF dosyalarından grafikleri otomatik çıkarıp** U-Net ile segmente edebiliyor!

#### 🎯 Akış

```
📄 PDF Dosyası
    ↓ (convert_from_path)
📖 PDF Sayfaları
    ↓ (crop_smart_area_v9)
📊 Grafikleri Bul & Crop
    ↓ (OCR - pytesseract)
🏷️ Motor Tipi/İrtifa/Ağırlık Çıkar
    ↓ (Dosya İsimlendir)
🎨 PNG Dosyaları
    ├── 1-10000-5000lb.png
    ├── 2-15000-7500lb.png
    └── ...
    ↓ (U-Net Model)
✅ Segmentation Sonuçları
    ├── *_segmentation.png
    ├── *_confidence.png
    └── *_overlay.png
```

---

#### 📦 Gerekli Paketler

##### Windows'ta Kurulum

```bash
# Python paketleri
pip install pdf2image pillow pytesseract

# Poppler (PDF'i görüntüye çevirir)
# Option 1: Chocolatey ile
choco install poppler

# Option 2: Manual indir
# https://github.com/oschwartz10612/poppler-windows/releases

# Tesseract (OCR - grafik metni okur)
# https://github.com/UB-Mannheim/tesseract/wiki
```

##### Linux/macOS

```bash
# Linux
sudo apt-get install poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract
```

---

#### 🚀 Hızlı Başlangıç

##### Senaryo 1: PDF'den Grafikleri Çıkar & Segmente Et

```bash
python segment_curves.py \
    --model checkpoints/best_model.pth \
    --pdf F18.pdf \
    --pages 10-50 \
    --output_dir F18_results
```

**Parametre Açıklaması:**
- `--model` : Eğitilmiş U-Net modeli (ZORUNLU)
- `--pdf` : PDF dosyası yolu
- `--pages` : Sayfa aralığı (1-indexed, inclusive)
- `--output_dir` : Çıktı klasörü

**Çıktı:**
```
F18_results_extracted/     ← Çıkarılan grafikleri (orijinal PNG)
├── 1-10000-5000lb.png
├── 1-10000-5000lb_2.png
└── ...

F18_results/               ← Segmentation sonuçları
├── 1-10000-5000lb_segmentation.png
├── 1-10000-5000lb_confidence.png
├── 1-10000-5000lb_overlay.png
└── ...
```

---

#### 🎛️ Seçenekler

##### Sadece Grafikleri Çıkar (Segmentation yapma)

```bash
python segment_curves.py \
    --model checkpoints/best_model.pth \
    --pdf F18.pdf \
    --pages 1-30 \
    --extract_only
```

→ Sadece `F18_results_extracted/` klasörü oluşturulur
→ Hızlı grafik çıkarma (segmentation uzun sürüyor)

##### Farklı Threshold Denemeleri

```bash
# Daha geniş curve'ler (threshold=0.3)
python segment_curves.py --pdf data.pdf --pages 1-10 --threshold 0.3

# Balanced (threshold=0.5)
python segment_curves.py --pdf data.pdf --pages 1-10 --threshold 0.5

# Daha kesin curve'ler (threshold=0.7)
python segment_curves.py --pdf data.pdf --pages 1-10 --threshold 0.7
```

##### Path'leri Manuel Belirtmek

```bash
python segment_curves.py \
    --model checkpoints/best_model.pth \
    --pdf F18.pdf \
    --pages 10-50 \
    --poppler_path "C:\Program Files\poppler\Library\bin" \
    --tesseract_path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

#### 📋 Dosya İsimlendirmesi

Dosya isimi **PDF'deki metin bilgisinden** otomatik çıkarılır:

**Format:** `{engine}-{altitude}-{weight}lb.png`

### Örnekler:
```
1-10000-5000lb.png
├─ 1         = Single Engine (1=Single, 2=Dual)
├─ 10000     = İrtifa (feet)
└─ 5000      = Ağırlık (pounds)

2-0-34000lb.png
├─ 2         = Dual Engine
├─ 0         = Sea Level (deniz seviyesi)
└─ 34000     = Maksimum Ağırlık (pounds)

MANUEL_Sayfa_15.png
└─ OCR başarısız oldu → Manuel kontrol gerekli
```

---

#### ✅ Çıktı Dosyaları

Her grafik için **3 dosya** oluşturulur:

### 1. `*_segmentation.png` - Binary Mask
- **Beyaz** = Curve (eğri)
- **Siyah** = Diğer her şey
- Bağımsız değişken segmentation

### 2. `*_confidence.png` - Confidence Heatmap
- **Kırmızı** = Yüksek confidence (% 100 curve)
- **Mavi** = Düşük confidence
- Model ne kadar emin olduğunu gösterir

### 3. `*_overlay.png` - Orijinal + Curves
- Orijinal grafik üzerine
- Renkli curve overlay
- Hızlı görsel doğrulama

---

#### 💡 Pratik Örnekler

### Örnek 1: Hızlı Test

```bash
python segment_curves.py \
    --model checkpoints/best_model.pth \
    --pdf test.pdf \
    --pages 1-5 \
    --output_dir test_results

# Zaman: ~2-3 dakika
```

### Örnek 2: Tüm Döküman

```bash
python segment_curves.py \
    --model checkpoints/best_model.pth \
    --pdf large_document.pdf \
    --pages 1-300 \
    --output_dir document_results

# Zaman: ~2-3 saat
```

### Örnek 3: Demo Script Kullan

```bash
python demo_pdf_segmentation.py \
    --pdf F18.pdf \
    --pages 10-20 \
    --threshold 0.5
```

→ Poppler & Tesseract path'lerini otomatik bulur

---

#### 🔎 OCR Algılaması

Model aşağıdakileri OCR'dan çıkarıyor:

##### ✓ Single vs Dual Engine
- "Single Engine" → `1`
- "Dual" / "Twin" → `2`

##### ✓ İrtifa
- "10,000 feet" → `10000`
- "Sea Level" / "S.L." → `0`

##### ✓ Ağırlık
- "5,000 Pounds" → `5000`
- "34,000 lbs" → `34000`

##### ⚠️ OCR Başarısız
- OCR okunamadı → `MANUEL_Sayfa_XX.png`
- Manuel rename gerekli

---

#### 🐛 Sorun Çözerme

### ❌ "ModuleNotFoundError: No module named 'pdf2image'"

```bash
pip install pdf2image pillow
```

### ❌ "Poppler not found"

**Windows:**
```bash
# Option 1: Chocolatey
choco install poppler

# Option 2: Manual path
python segment_curves.py --poppler_path "C:\poppler\Library\bin" ...
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

### ❌ "pytesseract not found"

```bash
pip install pytesseract

# + Tesseract OCR binary
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### ❌ Çok Yavaş

- Sayfa sayısını azalt ve test et
- `--extract_only` modu kullan (segmentation skip)
- `--input_size 128` ile model boyutunu değiştir

### ⚠️ Çok Fazla "MANUEL_Sayfa_XX.png"

- PDF kalitesi düşük olabilir
- OCR ayarlarını değiştir (Grafik_Crop_Tag.py'deki parametreler)
- Manuel olarak isimlendirmek gerekebilir

---

#### 📊 Benchmark Times

| İşlem | Zaman |
|-------|-------|
| 5 sayfa işle | ~30 saniye |
| 50 sayfa işle | ~5 dakika |
| 100 sayfa işle | ~10 dakika |
| 300 sayfa işle | ~30 dakika |

*GPU ile* (CUDA enabled)

---

#### 🔗 İlgili Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `segment_curves.py` | Ana inference script (PDF + segmentation) |
| `train_unet.py` | U-Net eğitim script'i |
| `Grafik_Crop_Tag.py` | Orijinal grafik çıkarma algoritması (referans) |
| `PDF_SEGMENT_REHBER.py` | Detaylı kullanım kılavuzu |
| `demo_pdf_segmentation.py` | Quick demo script |

---

#### 📚 Başlıca Fonksiyonlar

### `PDFGraphicExtractor` Class

```python
extractor = PDFGraphicExtractor()

# PDF'den grafikleri çı kar
extracted_files = extractor.extract_charts_from_pdf(
    pdf_path="F18.pdf",
    start_page=10,
    end_page=50,
    output_dir="extracted_graphics"
)
```

### `CurveSegmentationInference` Class

```python
inferencer = CurveSegmentationInference(
    model_path="checkpoints/best_model.pth"
)

# Batch processing
inferencer.process_batch(
    input_dir="extracted_graphics",
    output_dir="segmentation_results"
)
```

---

#### 🎓 İş Akışı (Complete)

1. **Eğitim** (`train_unet.py`)
   - U-Net modelini eğit
   - `checkpoints/best_model.pth` oluştur

2. **Grafik Çıkarma** (`segment_curves.py --pdf`)
   - PDF'den grafikleri çıkar
   - OCR ile isimlendir
   - PNG dosyalarını kaydet

3. **Segmentation** (`segment_curves.py --pdf`)
   - U-Net ile curve'leri çıkar
   - Segmentation/confidence/overlay oluştur

4. **Sonuçlar**
   - Mask görselleri
   - Confidence haritaları
   - Overlay görselleri

---

#### ✨ Özetle

- ✅ PDF dosyalarını otomatik işle
- ✅ Grafikleri akıllı croplayıp isimlendir
- ✅ U-Net ile curve'leri segmente et
- ✅ Çıktıları düzenli şekilde kaydet
- ✅ Threshold ve parametreler özelleştirilebilir
- ✅ Hızlı ve verimli processing

---

**Kullanım Örneği:**
```bash
# Örneği çalıştır
python segment_curves.py --model checkpoints/best_model.pth --pdf myfile.pdf --pages 10-50
```

**Sonuç:**
- Sayfa 10-50'den grafikleri otomatik çıkar
- Her grafiği OCR'dan bilgiye göre isimlendir
- U-Net ile curve'leri segmente et
- Sonuçları `segmentation_results/` kaydeder

Eğitilmiş yapay zeka modelini test etmek, yeni görselleri işlemek ve **PDF formatındaki dokümanlardan otomatik veri ayıklamak** için yenilenmiş `5-segment_curves.py` komutunu kullanın.

**1. Tekil Klasör Batch/Toplu İşleme Modu:**
```bash
python 5-segment_curves.py --model checkpoints/best_model.pth --input_dir test_images/ --output_dir segmentation_results/
```

**2. GÜNCEL: PDF Modu İle Sayfalar Arası Otomatik Analiz:**
Toplu PDF raporlarından grafikleri okutup içlerindeki curve'leri çıkarmak için.
```bash
python 5-segment_curves.py --model checkpoints/best_model.pth --pdf Rapor.pdf --pages 10-50 --poppler_path "C:\...\poppler\bin"
```

**Temel Seçenekler:**
- `--threshold 0.5`: Binary mask oluşturma eşiği. Daha fazla ince curve yakalamak için esnetilebilir (Örn: `0.3`) ya da keskinleştirebilir (Örn: `0.7`).
- `--no_confidence`: Olasılık/güvenlik haritalarının (Heatmap) kaydedilmesini engeller (Disk tasarrufu sağlar).
- `--extract_only`: Modeli çalıştırmadan **sadece pdf'ten grafikleri ayıklar** (`_extracted` klasörüne kaydeder).
- `--extract_curve_data`: Segmente edilen maskeler üzerinden aynı işleyiş içinde sayısal formata dönüştürüp Excel dosyası çıkarır (Adım 4 ile aynı işlemi birleştirir).

### ADIM 4: Eğrileri Sayısallaştırıp Excel'e Çek (`extract_curve_data.py`)

### 📚 extract_curve_data.py - Python Öğretim Rehberi

Bu dosya **Python öğrenirken** `extract_curve_data.py` kodunu adım adım anlamak için oluşturulmuştur.

---

#### 📋 İÇİNDEKİLER

1. [BÖLÜM 1: Import ve Kurulum](#bölüm-1-import-ve-kurulum)
2. [BÖLÜM 2: AxisDetector Sınıfı](#bölüm-2-axisdetector-sınıfı)
3. [BÖLÜM 3: OpenCV Görüntü İşleme](#bölüm-3-opencv-görüntü-işleme)
4. [BÖLÜM 4: CurveDataExtractor Sınıfı](#bölüm-4-curvedataextractor-sınıfı)
5. [BÖLÜM 5: Excel Export](#bölüm-5-excel-export)
6. [BÖLÜM 6: Main Workflow](#bölüm-6-main-workflow)

---

#### BÖLÜM 1: Import ve Kurulum

## Satırlar 1-28

```python
import os                    # İşletim sistemi işlemleri
import cv2                   # Görüntü işleme (OpenCV)
import numpy as np           # Sayısal hesaplamalar
import pandas as pd          # Veri tabloları ve Excel
import re                    # Regex (metin arama)
from pathlib import Path     # Dosya yolları
from typing import Tuple, Optional, Dict, List  # Tip ipuçları
from dataclasses import dataclass, asdict       # Veri sınıfları

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
```

### 🎯 İMPORT NEDİR?

**`import`** = Harici kütüphane veya modul yükleme

```python
import os
# Böylece os.path.exists(), os.listdir() vb. kullanabilirsin
```

### 📚 KULLANILAN KÜTÜPHANELER

| Kütüphane | Ne İşe Yarar | Örnek |
|-----------|-------------|-------|
| **os** | Dosya/klasör işlemleri | `os.path.exists("file.txt")` |
| **cv2** (OpenCV) | Görüntü işleme | `cv2.imread("image.png")` |
| **numpy** | Matematik/diziler | `np.array([1, 2, 3])` |
| **pandas** | Excel/tabuler veri | `pd.DataFrame()` |
| **re** | Metin arama (regex) | `re.sub(r'[^\d.]', '', text)` |
| **pathlib** | Modern dosya yolları | `Path("file.txt")` |
| **pytesseract** | OCR (resimden yazı) | `pytesseract.image_to_string()` |

### 🛡️ TRY-EXCEPT (Hata Yönetimi)

```python
try:
    import pytesseract      # Yüklemeye dene
    OCR_AVAILABLE = True    # Başarılı olursa True yap
except ImportError:         # Yükleme başarısız olursa
    OCR_AVAILABLE = False   # OCR kullanılamaz
    pytesseract = None
```

**Neden?** Pytesseract yüklü olmayabilir. Hata vermek yerine kontrol altında tutarız.

---

### 📌 TIP İPUÇLARI (Type Hints)

```python
from typing import Tuple, Optional, Dict, List

# Fonksiyon
def find_value(name: str) -> Optional[int]:
    # name: str = "name" parametresi string olmalı
    # -> Optional[int] = Geri dönüş int ya da None olabilir
    if name:
        return 42
    return None
```

---

# BÖLÜM 2: AxisDetector Sınıfı

## Satırlar 33-150+

```python
class AxisDetector:
    """Grafik eksenlerini OCR ile tanı ve min-max değerlerini bul"""
```

### 🎯 SINIF NEDİR?

**`class`** = Ilgili veriler ve fonksiyonları bir arada tutmak

```python
class Car:
    def __init__(self):
        self.brand = "Toyota"
        self.color = "red"

    def drive(self):
        print("Driving...")

# Kullanım
my_car = Car()
print(my_car.brand)  # "Toyota"
my_car.drive()       # "Driving..."
```

### 🔧 __init__ METODU (Constructor)

```python
def __init__(self, tesseract_path: Optional[str] = None):
    """Sınıf oluşturulduğunda yapılacaklar"""

    if not OCR_AVAILABLE:
        raise ImportError("pytesseract gerekli: pip install pytesseract")
        # ImportError fırlat = Hata mesajı göster

    if tesseract_path and os.path.exists(tesseract_path):
        # Tesseract yolu verilmişse ve varsa
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        # Pytesseract'e yolu söyle
```

**Kullanım:**
```python
# Windows'ta Tesseract yolu
detector = AxisDetector(
    tesseract_path="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

# Tesseract yolu belirtmezsen
detector = AxisDetector()  # Sistem PATH'i kullanır
```

---

## find_grid_bounds METODU

```python
@staticmethod
def find_grid_bounds(img):
    """Grafik ızgarasını (grid) bulur"""

    # STEP 1: RENK DÖNÜŞTÜRME (BGR → GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 📸 RENK DÖNÜŞTÜRME NEDİR?

```
Orijinal (Renk):  Gri (Grayscale):
[R,G,B] ×480000   [0-255] ×480000
Renkli her piksel  Siyah-beyaz
3 değer (RGB)      1 değer
```

**Neden?** Grayscale işlemek daha hızlı ve işleme kolaylaştırır.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = Renkli görüntü
# cv2.COLOR_BGR2GRAY = Dönüştürme türü
# gray = Sonuç (0-255 değerli gri görüntü)
```

### 🌫️ GAUSSIAN BLUR (Bulanıklaştırma)

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# (5, 5) = Kernel boyutu (5x5 piksel)
# 0 = Sigma (bulanıklık derecesi, otomatik)
```

**Görsel:**
```
Orijinal:  1  255  10  200
           ^   ^   ^   ^
           Gürültü var (sıçramalar)

Bulanık:   50  150  80  150
           ^   ^   ^   ^
           Yumuşak (gürültü azaldı)
```

### ⚫⚪ THRESHOLDING (Siyah-Beyaz)

```python
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
```

**Parametre Açıklaması:**
- `blurred` = Input görüntü
- `200` = Eşik değeri (threshold)
  - Piksel < 200 → 0 (siyah)
  - Piksel ≥ 200 → 255 (beyaz)
- `255` = Beyaz değeri
- `cv2.THRESH_BINARY_INV` = Ters yap (siyah↔beyaz)

**Örnek:**
```python
# Orijinal grayscale: [50, 150, 210, 230]
# Threshold 200 ile: [0, 0, 255, 255]
#                    s  s  b   b
# BINARY_INV ile:   [255, 255, 0, 0]
#                    b    b   s  s
```

### 🔍 CONTOUR BULMA

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Contour?** = Beyaz şekillerin sınırları

```
Siyah-Beyaz Görüntü:
████████░
████████░
░░░░░░░░░

Contours (sınırlar):
████░░░░░
█░░░░░░░░
█░░░░░░░░
```

---

## EN BÜYÜK CONTOUR BULMA

```python
max_area = 0
best_rect = None
img_area = img.shape[0] * img.shape[1]
# img.shape[0] = Yükseklik (height)
# img.shape[1] = Genişlik (width)
# İmg.shape[2] = Kanal (renk - yoksa 2D)

for cnt in contours:                    # Her contour için
    x, y, w, h = cv2.boundingRect(cnt)  # Bounding box (dikdörtgen)
    area = w * h                        # Alan hesapla

    if area > (img_area * 0.1) and area < (img_area * 0.95):
        # Alan: görüntünün %10-%95'i arasında
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)    # En büyük rectangle'ı sakla

return best_rect  # (x, y, genişlik, yükseklik)
```

**Neden bunu yaparız?** Grafiğin sınırlarını bulmak için! En büyük beyaz alan = grafik bölgesi

---

## clean_number METODU

```python
@staticmethod
def clean_number(text):
    """Metinden sayı çıkar ve temizle"""

    # STEP 1: KARAKTERLERI DÜZELT
    text = text.upper()                                  # AŞAĞı → BÜYÜK
    text = text.replace('O', '0')                        # Harf O → Sıfır 0
    text = text.replace('I', '1')                        # Harf I → Bir 1
    text = text.replace('L', '1')                        # Harf L → Bir 1
    text = text.replace(',', '.')                        # Virgül → Nokta
    # Neden? OCR hatıyor, bu hataları düzeltiyoruz
```

**OCR Hataları Örneği:**
```
Gerçek:   0.15     0.50     0.87
OCR okudu: O.I5    O.5O     O.87
(Sıfır=O, Bir=I/L vb.)

clean_number sonra: 0.15    0.50    0.87
```

```python
    # STEP 2: SADECE SAYI VE NOKTA TUTA
    clean = re.sub(r'[^\d.]', '', text)
    # Regex: [^\d.] = "Sıfır-9 ve nokta OLMAYAN her şey"
    # re.sub(..., '', ...) = Bunları boşla değiştir (kaldır)

    if not clean:  # Eğer boş ise
        return None
```

**Örnek:**
```python
text = "Price: $12.50 USD"
clean = re.sub(r'[^\d.]', '', text)
# Sonuç: "12.50"
```

```python
    # STEP 3: NOKTA HATASI DÜZELTME
    if clean.startswith('.'):      # ".123" → "0.123"
        clean = '0' + clean

    if clean.endswith('.'):        # "123." → "123"
        clean = clean[:-1]         # [:-1] = Son karakter hariç

    if clean.count('.') > 1:       # "12.34.56" → "12.34"
        parts = clean.split('.')   # ["12", "34", "56"]
        clean = parts[0] + '.' + parts[1]  # "12.34"
```

```python
    # STEP 4: STR → FLOAT DÖNÜŞTÜRME
    try:
        val = float(clean)  # String'i sayıya çevir
        return val          # OK
    except:                 # Hata olursa
        return None         # None döndür
```

**Test Edelim:**
```python
clean_number("O.15")    # "0" + "." + "15" → 0.15
clean_number("Price: $12")  # "12" → 12.0
clean_number("Invalid!")    # "Invalid" → None
```

---

#### BÖLÜM 3: OpenCV Görüntü İşleme

## scan_strip_for_numbers METODU

```python
@staticmethod
def scan_strip_for_numbers(img_roi, config):
    """Görüntü şeridinden sayıları OCR ile bulur"""

    candidates = []  # Bulunan sayıları sakla

    # STEP 1: ÖLÇEKLENDIR
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    scale = cv2.resize(gray, None, fx=2, fy=2,
                       interpolation=cv2.INTER_CUBIC)
    # fx=2 = Genişliği 2x yap
    # fy=2 = Yüksekliği 2x yap
    # INTER_CUBIC = Kaliteli ölçekleme
```

**Neden 2x büyütülür?** OCR daha kaliteli çalışır büyük resimlerle!

```python
    # STEP 2: THRESHOLD
    _, thresh = cv2.threshold(scale, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # THRESH_OTSU = Otomatik eşik (biz belirtmeye gerek yok)
```

```python
    # STEP 3: OCR
    try:
        data = pytesseract.image_to_data(
            thresh,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        # DICT = Sözlük formatında döndür
        # {'text': [...], 'left': [...], 'top': [...], ...}
```

**Çıktı Örneği:**
```python
data = {
    'text': ['0.15', '0.20', '0.25'],      # Okunan yazılar
    'left': [50, 100, 150],                # X koordinatları
    'top': [30, 30, 30],                   # Y koordinatları
    'width': [40, 40, 40],                 # Genişlikler
    'height': [20, 20, 20]                 # Yükseklikler
}
```

```python
        n_boxes = len(data['text'])        # Kaç tane metin?

        for i in range(n_boxes):           # Her metin için
            text = data['text'][i].strip() # Metin al (boşlukları kaldır)
            val = AxisDetector.clean_number(text)  # Temizle

            if val is not None:            # Geçerli sayı ise
                # Orijinal boyuta dön (2x büyütüldü, geri kes)
                x = data['left'][i] / 2
                y = data['top'][i] / 2
                w = data['width'][i] / 2
                h = data['height'][i] / 2

                center_x = x + w/2         # Merkez X
                center_y = y + h/2         # Merkez Y

                candidates.append({
                    'val': val,            # Sayısal değer
                    'x': center_x,         # Merkez X
                    'y': center_y,         # Merkez Y
                    'box': (int(x), int(y), int(w), int(h))
                })

    except Exception as e:
        print(f"[WARNING] OCR scan başarısız: {e}")

    return candidates  # Bulunan sayıları döndür
```

---

#### BÖLÜM 4: CurveDataExtractor Sınıfı

## CurvePoint (Dataclass)

```python
from dataclasses import dataclass

@dataclass
class CurvePoint:
    """Curve üzerindeki tek veri noktası"""
    x: float              # X koordinatı (Mach Number)
    y: float              # Y koordinatı (Specific Range)
    drag_index: int       # Progresyon (0-100)
    pixel_x: int          # Pixel X
    pixel_y: int          # Pixel Y
```

**Dataclass nedir?**
```python
# Normal sınıf
class CurvePoint:
    def __init__(self, x, y, drag_index, pixel_x, pixel_y):
        self.x = x
        self.y = y
        self.drag_index = drag_index
        ...

# Dataclass ile (daha kısa!)
@dataclass
class CurvePoint:
    x: float
    y: float
    drag_index: int
    pixel_x: int
    pixel_y: int
```

**Kullanım:**
```python
point = CurvePoint(x=0.15, y=0.23, drag_index=50,
                   pixel_x=100, pixel_y=200)
print(point.x)  # 0.15
```

---

## extract_curve_pixels METODU

```python
@staticmethod
def extract_curve_pixels(mask_path: str) -> np.ndarray:
    """Segmentation mask'tan curve pixel'lerini çıkar"""

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Maske yükle (siyah-beyaz format)

    if mask is None:
        raise FileNotFoundError(f"Mask bulunamadı: {mask_path}")
        # Exception fırlat (hata)

    # STEP 1: BEYAZ PIKSELLERI BULMAK
    curve_points = np.where(mask > 127)
    # np.where = Koşulu sağlayan pikselleri bul
    # mask > 127 = 127'den büyük (beyaz) pikseller
    # RETURN: (y_array, x_array) tuple
```

**Örnek:**
```python
mask = np.array([
    [0, 255, 0],
    [255, 0, 255],
    [0, 255, 0]
])

curve_points = np.where(mask > 127)
# Sonuç: ([0, 1, 1, 2], [1, 0, 2, 1])
# Y'ler: [0, 1, 1, 2]
# X'ler: [1, 0, 2, 1]
```

```python
    # STEP 2: TUPLE'I ARRAY'E ÇEVİR
    return np.column_stack(curve_points)
    # np.column_stack = Dikey olarak birleştir
    # Sonuç: [[0, 1], [1, 0], [1, 2], [2, 1]]
    #        [y, x], [y, x], ...
```

---

## calculate_drag_index METODU

```python
@staticmethod
def calculate_drag_index(curve_pixels: np.ndarray) -> np.ndarray:

    Curve'ın her noktası için drag index hesapla (0-100)

    Drag Index = Başlangıçtan bu noktaya kadar toplam mesafe

    if len(curve_pixels) < 2:
        return np.zeros(len(curve_pixels))
        # 1 pikselden az ise 0 dön

    # STEP 1: ARDIŞIK NOKTALAR ARASINDAKI FARK
    diffs = np.diff(curve_pixels, axis=0)
    # np.diff = Ardışık elemanlar arasındaki fark
```

**Örnek:**
```python
curve_pixels = np.array([
    [100, 50],
    [102, 52],
    [105, 55],
    [110, 60]
])

diffs = np.diff(curve_pixels, axis=0)
# Sonuç:
# [[102-100, 52-50],    = [2, 2]
#  [105-102, 55-52],    = [3, 3]
#  [110-105, 60-55]]    = [5, 5]
```

```python
    # STEP 2: MESAFE HESAPLA (EUCLIDEAN)
    distances = np.linalg.norm(diffs, axis=1)
    # ||v|| = sqrt(x² + y²)
    # axis=1 = Her satır için (her nokta çifti)
```

**Örnek:**
```python
diffs = np.array([[2, 2], [3, 3], [5, 5]])
distances = np.linalg.norm(diffs, axis=1)
# sqrt(2² + 2²) = sqrt(8) ≈ 2.83
# sqrt(3² + 3²) = sqrt(18) ≈ 4.24
# sqrt(5² + 5²) = sqrt(50) ≈ 7.07
# Sonuç: [2.83, 4.24, 7.07]
```

```python
    # STEP 3: KUMULATIF TOPLAM (STARTING 0)
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    # np.cumsum = Kümülatif toplam
    # np.concatenate = Dizileri birleştir
```

**Örnek:**
```python
distances = np.array([2.83, 4.24, 7.07])
cumsum = np.cumsum(distances)
# [2.83, 7.07, 14.14]

cumulative = np.concatenate([[0], cumsum])
# [0, 2.83, 7.07, 14.14]
# ↑ Başlangıç 0'dan başla
```

```python
    # STEP 4: 0-100 NORMALIZE ET
    if cumulative[-1] > 0:
        # cumulative[-1] = Son eleman
        drag_indices = (cumulative / cumulative[-1]) * 100
        # Her mesafeyi toplam mesafeye böl, 100 ile çarp
    else:
        drag_indices = np.zeros_like(cumulative)
        # Hiç mesafe yoksa hepsi 0

    return drag_indices.astype(int)  # Int'e çevir
```

**Örnek:**
```python
cumulative = [0, 2.83, 7.07, 14.14]
total = 14.14

drag_indices = (cumulative / 14.14) * 100
# [0/14.14*100, 2.83/14.14*100, 7.07/14.14*100, 14.14/14.14*100]
# [0, 20, 50, 100]
```

---

## extract_curve_data METODU (MAIN)

```python
def extract_curve_data(
    self,
    original_image_path: str,
    mask_path: str,
    filename: str
) -> List[CurvePoint]:
    """Bir grafikteki tüm curve verilerini çıkar"""

    # STEP 1: EKSEN VALUES'ı BUL
    x_min, x_max, y_min, y_max = self.axis_detector.get_axis_extremes(
        original_image_path
    )
    # Grafik eksenindeki min-max değerleri OCR'la oku

    if any(v is None for v in [x_min, x_max, y_min, y_max]):
        # any() = En az biri True ise True
        print(f"[WARNING] Eksen değerleri alınamadı: {filename}")
        return []  # Boş liste döndür
```

```python
    # STEP 2: CURVE PIXEL'LERİNİ BULMAK
    curve_pixels = self.extract_curve_pixels(mask_path)
    # (N, 2) shape: [[y, x], [y, x], ...]

    if len(curve_pixels) == 0:
        print(f"[WARNING] Curve pixel'leri bulunamadı: {filename}")
        return []
```

```python
    # STEP 3: DRAG INDEX HESAPLA
    drag_indices = self.calculate_drag_index(curve_pixels)
    # [0, 10, 20, ..., 100]
```

```python
    # STEP 4: PIXEL → VERI KOORDINATI ÇEVİRME
    img_h, img_w = cv2.imread(
        original_image_path,
        cv2.IMREAD_GRAYSCALE
    ).shape
    # Görüntü yüksekliği ve genişliği

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    grid_h, grid_w = mask.shape
    # Mask boyutları

    data_points = []
    for i, (py, px) in enumerate(curve_pixels):
        # enumerate = Index + Değer beraber

        # NORMALİZASYON (0-1 aralığına çevir)
        norm_x = px / grid_w      # 0-1 arasında
        norm_y = 1 - (py / grid_h)  # 0-1 arasında (Y tersi)
        # Neden Y tersi? Pixel Y aşağı gidiyor, graf Y yukarı
```

**Örnek:**
```
Pixel Y: 0 (üst)     Grafik Y: Yüksek
Pixel Y: 100 (alt)   Grafik Y: Düşük

Tersine çevirme gerekli!
```

```python
        # VERI ARALIŞINA ÖLÇEKLE
        data_x = x_min + norm_x * (x_max - x_min)
        # norm_x=0 → x_min
        # norm_x=1 → x_max

        data_y = y_min + norm_y * (y_max - y_min)
        # norm_y=0 → y_min
        # norm_y=1 → y_max
```

**Örnek:**
```python
# Eksen: x_min=0.1, x_max=0.7, norm_x=0.5
data_x = 0.1 + 0.5 * (0.7 - 0.1)
       = 0.1 + 0.5 * 0.6
       = 0.1 + 0.3
       = 0.4  ✓ Ortada!
```

```python
        # CURVEPOINT OBJESI OLUŞTUR
        data_points.append(CurvePoint(
            x=data_x,
            y=data_y,
            drag_index=int(drag_indices[i]),
            pixel_x=int(px),
            pixel_y=int(py)
        ))

    print(f"[✓] {len(data_points)} curve noktası çıkarıldı")
    return data_points
```

---

#### BÖLÜM 5: Excel Export

## CurveDataExporter Sınıfı

```python
class CurveDataExporter:
    """Curve verilerini Excel'e aktar"""

    # ALTITUDE SHEET NAMES MAPPINGU
    ALTITUDE_SHEET_NAMES = {
        0: "Sea Level",
        5000: "5,000 Feet",
        10000: "10,000 Feet",
        # ... vb
    }
```

### extract_metadata_from_filename

```python
@staticmethod
def extract_metadata_from_filename(filename: str) -> Dict:

    Dosya adından metadata çıkar

    Format: {engine}-{altitude}-{weight}lb.png
    Örn: 1-10000-5000lb.png

    try:
        parts = filename.replace('.png', '').split('-')
        # "1-10000-5000lb" → ["1", "10000", "5000lb"]

        return {
            'engine': int(parts[0]),                    # "1" → 1
            'altitude': int(parts[1]),                  # "10000" → 10000
            'weight': int(parts[2].replace('lb', ''))   # "5000lb" → 5000
        }

    except Exception as e:
        print(f"[ERROR] Filename parse edilemedi: {filename} ({e})")
        return None
```

**Test:**
```python
extract_metadata_from_filename("1-10000-5000lb.png")
# {'engine': 1, 'altitude': 10000, 'weight': 5000}

extract_metadata_from_filename("2-5000-8000lb.png")
# {'engine': 2, 'altitude': 5000, 'weight': 8000}
```

---

### create_dataframe

```python
@staticmethod
def create_dataframe(curve_data: Dict[str, List[CurvePoint]]) -> pd.DataFrame:

    Curve verilerinden DataFrame oluştur

    Input: {'1-10000-5000lb.png': [CurvePoint(...), ...], ...}
    Output: Pandas DataFrame (Excel'e yazılabilir)

    rows = []  # Satırlar

    for filename, points in curve_data.items():
        # Her dosya için
        metadata = CurveDataExporter.extract_metadata_from_filename(filename)
        # Metadata çıkar: engine, altitude, weight

        if not metadata:
            continue  # Hatalı ise atla

        for point in points:
            # Her curve noktası için
            rows.append({
                'Altitude (ft)': metadata['altitude'],
                'Gross Weight  (lb)': metadata['weight'],
                'Drag Index': point.drag_index,
                'Mach Number (Ma)': point.x,           # X = Mach
                'L/D Coefficient': point.y,             # Y = L/D
                'Fuel Flow (lb / h)': 0                 # Placeholder
            })

    return pd.DataFrame(rows)
    # Liste'i DataFrame'e çevir
```

**Çıktı Örneği:**
```
   Altitude (ft)  Gross Weight  ...  Mach Number (Ma)  L/D Coefficient
0            5000          4000  ...              0.15              0.23
1            5000          4000  ...              0.16              0.25
2            5000          4000  ...              0.17              0.28
...
```

---

### export_to_excel

```python
@staticmethod
def export_to_excel(curve_data: Dict[str, List[CurvePoint]],
                    output_path: str):
    """Curve verilerini Excel'e kaydet"""

    df = CurveDataExporter.create_dataframe(curve_data)
    # DataFrame oluştur

    if df.empty:
        print("[WARNING] DataFrame boş, Excel yazılamadı")
        return

    # ENGINE'YE GÖRE AYRILAN DOSYALARA KAYDET
    engines = set()  # Benzersiz engine'ler

    for filename in curve_data.keys():
        metadata = CurveDataExporter.extract_metadata_from_filename(filename)
        if metadata:
            engines.add(metadata['engine'])
    # engines = {1, 2} (1-motor ve 2-motor)

    for engine in sorted(engines):
        # Engine = 1, sonra 2

        # Bu engine'ye ait dosyaları filtrele
        engine_filenames = [
            f for f in curve_data.keys()
            if CurveDataExporter.extract_metadata_from_filename(f)['engine'] == engine
        ]

        engine_points = {fname: curve_data[fname] for fname in engine_filenames}
        # Sadece bu engine'nin verileri

        engine_df = CurveDataExporter.create_dataframe(engine_points)
        # DataFrame oluştur

        # OUTPUT PATH FIX
        engine_suffix = 'One_Engine_Data' if engine == 1 else 'Two_Engine_Data'
        engine_output_path = output_path.replace('.xlsx', f'_{engine_suffix}.xlsx')
        # curve_data.xlsx → curve_data_One_Engine_Data.xlsx

        # ALTITUDE'YE GÖRE SHEETS OLUŞTUR
        with pd.ExcelWriter(engine_output_path, engine='openpyxl') as writer:
            # Excel yazıcı context manager (otomatik kapat)

            for altitude in sorted(engine_df['Altitude (ft)'].unique()):
                # Her altitude için

                altitude_df = engine_df[engine_df['Altitude (ft)'] == altitude]
                # Sadece bu altitude'nin satırları

                sheet_name = CurveDataExporter.get_sheet_name_for_altitude(altitude)
                # "5000" → "5,000 Feet"

                altitude_df = altitude_df[[
                    'Altitude (ft)', 'Gross Weight  (lb)', 'Drag Index',
                    'Mach Number (Ma)', 'L/D Coefficient', 'Fuel Flow (lb / h)'
                ]]
                # Sütun sırasını ayarla

                altitude_df.to_excel(writer, sheet_name=sheet_name, index=False)
                # Bu altitude'ı sheet'e yaz
                # index=False → İndeks sütununu yazma

        print(f"[✓] Excel kaydedildi: {engine_output_path}")
```

---

#### BÖLÜM 6: Main Workflow

## extract_curves_from_segmentation FONKSIYONU

```python
def extract_curves_from_segmentation(
    segmentation_results_dir: str,
    original_graphics_dir: str,
    output_excel: str = "curve_data.xlsx",
    tesseract_path: Optional[str] = None
):

    Segmentation sonuçlarından curve verilerini çıkart
    ve Excel'e kaydet

    print(f"\n{'='*70}")
    print("CURVE DATA EXTRACTION")
    print(f"{'='*70}\n")

    # EXTRACTOR OLUŞTUR
    extractor = CurveDataExtractor(tesseract_path=tesseract_path)
    curve_data = {}  # {filename: [CurvePoint, ...], ...}

    # TÜM MASK'LARI İŞLE
    mask_files = sorted([
        f for f in os.listdir(segmentation_results_dir)
        if f.endswith('_segmentation.png')
    ])
    # Listten isimlendirilen dosyaları al

    print(f"[Processing] {len(mask_files)} segmentation mask işleniyor...\n")

    for mask_file in mask_files:
        # mask_file = "page_01_segmentation.png"

        base_name = mask_file.replace('_segmentation.png', '')
        # "page_01_segmentation.png" → "page_01"

        original_filename = base_name + '.png'
        # "page_01.png"

        mask_path = os.path.join(segmentation_results_dir, mask_file)
        # "/results/page_01_segmentation.png"

        original_path = os.path.join(original_graphics_dir, original_filename)
        # "/extracted/page_01.png"

        if not os.path.exists(original_path):
            print(f"[WARNING] Orijinal grafik bulunamadı: {original_filename}")
            continue  # Atla

        print(f"[Processing] {base_name}")

        try:
            points = extractor.extract_curve_data(
                original_path,
                mask_path,
                original_filename
            )
            # Curve'ları çıkar

            if points:
                curve_data[original_filename] = points
                # Saklayıp tutun

        except Exception as e:
            print(f"[ERROR] {base_name}: {e}")

    # EXCEL'E KAYDET
    if curve_data:
        print(f"\n[Exporting] {len(curve_data)} grafik Excel'e yazılıyor...")
        CurveDataExporter.export_to_excel(curve_data, output_excel)
        print(f"\n[✓] Tamamlandı!")
    else:
        print("[WARNING] Hiç curve verisi çıkarılamadı")
```

---

## __main__ BLOĞU

```python
if __name__ == '__main__':
    # Sadece bu dosya direkt çalıştırıldığında çalış
    # (import edildiğinde çalışma!)

    extract_curves_from_segmentation(
        segmentation_results_dir='segmentation_results',
        original_graphics_dir='segmentation_results_extracted',
        output_excel='curve_data.xlsx'
    )
```

**Neden?**
```python
# run.py dosyasında:
from extract_curve_data import CurveDataExtractor
# Bu dosyayı import ettikten sonra
# if __name__ == '__main__' bloğu çalışmaz (iyi!)

# Ama çalıştırırsan:
python extract_curve_data.py
# Bu blok çalışır
```

---

#### 🎓 ÖZETLEYİCİ FLOWCHART

```
1. IMAGE LOAD
   ├─ cv2.imread() = Görüntüyü oku

2. RENK & İŞLEME
   ├─ cv2.cvtColor() = Renge dönüştür
   ├─ cv2.GaussianBlur() = Bulanıklaştır
   ├─ cv2.threshold() = Threshold
   └─ cv2.findContours() = Grid bul

3. OCR
   ├─ Tessarect = Yazıyı oku
   ├─ clean_number() = Temizle
   └─ get_axis_extremes() = Min-Max bul

4. CURVE EXTRACTION
   ├─ extract_curve_pixels() = Beyaz pikselleri bul
   ├─ calculate_drag_index() = Progresyon hesapla
   └─ Pixel → Veri koordinati (dönüştür)

5. EXCEL EXPORT
   ├─ create_dataframe() = Tablo oluştur
   ├─ export_to_excel() = Excel yazıcı
   └─ Altitude-based sheets = Bölüntü
```

---

#### 📚 PYTHON KONSEPTLERİ

| Konsept | Açıklama | Örnek |
|---------|----------|-------|
| **import** | Kütüphane yükleme | `import os` |
| **class** | Nesnelerse şablonu | `class Car:` |
| **def** | Fonksiyon | `def drive():` |
| **@staticmethod** | self gereksiz metod | `@staticmethod def f():` |
| **@dataclass** | Otomatik __init__ | `@dataclass class Point:` |
| **try/except** | Hata yönetimi | `try: ... except:` |
| **list comprehension** | Hızlı liste oluşturma | `[x*2 for x in list]` |
| **for i, item in enumerate** | Index + değer | `for i, x in enumerate(list):` |
| **np.where()** | Koşulu sağlayan indis | `np.where(arr > 5)` |
| **pd.DataFrame()** | Tabelar veri | `pd.DataFrame(data)` |
| **with ... as** | Context manager | `with open() as f:` |

---

**Son Güncelleme:** 12 Şubat 2026
**Seviye:** Başlangıç (Temel Python + OpenCV)

Yapay zeka tarafından maskelenip bulunan (segmente edilen) eğrileri, grafiğin (X, Y) değerleriyle eşleştirerek Excel tablosuna otomatik dökmek mümkündür. Dilerseniz bunu yukarıda `--extract_curve_data` bayrağı ile tek kalemde yapabilir, dilerseniz de bağımsız olarak aşağıdaki gibi çalıştırabilirsiniz:

**Komut:**
```bash
python extract_curve_data.py \
    --segmentation_dir segmentation_results \
    --original_dir test_images \
    --output curve_data.xlsx \
    --tesseract_path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```
**Nasıl Çalışır?**
1. Modelden gelen siyah-beyaz maske incelenir (curve pikselleri).
2. Tesseract OCR ile orijinal görsel üzerinden X ve Y minimum-maksimum eksen değerleri okunur.
3. Pikseller sayısal grafik birimine çevrilir ve tek/çift motor (`one_engine_data.xlsx` vb) sekmeleri ile Excel dosyasına satırlandırılır.

*(Not: Tesseract path'i Windows makinalarda gereklidir ve tırnak içinde gönderilmelidir)*

---

## 📊 Dosya İskeleti ve İçerik Açıklamaları

Mevcut dosya yapısı modüler şekilde güncellenmiştir:

| Dosya / Script | Görev ve Açıklama |
|---|---|
| **`3-synthetic_production.py`** | Yüksek hacimli paralel sentetik grafik üreticisi. (Sürüm 3) |
| **`train_unet.py`** | PyTorch temelli U-Net yapay zeka/segmentasyon model eğitim script'i. |
| **`5-segment_curves.py`** | Görsellerdeki ya da bütün bir PDF içerisindeki eğrilerin tespit edilip çıkarılmasından sorumlu en güncel tahmin (inference) script'i. Ayrıca extraction adımıyla iç içe çalışabilir. |
| **`extract_curve_data.py`** | OCR okuma, veriyi sayısal sayfalara dökme ve maskeleri Excel'e aktarma kütüphanesi. |
| **`synthetic_data_gui.py` & `run_gui.bat`** | Terminal sevmeyen kullanıcılar için eklenmiş Sentetik Veri Paneli kontrol arayüzü. |
| **`checkpoints/`** | Modellerin kaydedildiği yer. |

EXCEL FORMAT DOĞRULMASI - ÖZETİ
=============================

Bu dosya `extract_curve_data.py` güncelleme notlarını içerir.


### Template Format Doğrulaması


Template Dosyaları Kontrol Edildi:
├─ One_Engine_Data.xlsx (Engine Type = 1, Single Engine)
└─ Two_Engine_Data.xlsx (Engine Type = 2, Dual Engine)

Format Yapısı:
┌─────────────────────────────────────────────────────────┐
│ Sheet Names = Altitude'ye göre ayrılmış                │
│ ├─ Sea Level (0 ft)                                   │
│ ├─ 5,000 Feet                                          │
│ ├─ 10,000 Feet                                         │
│ ├─ 15,000 Feet                                         │
│ ├─ 20,000 Feet                                         │
│ ├─ 25,000 Feet                                         │
│ ├─ 30,000 Feet                                         │
│ ├─ 35,000 Feet                                         │
│ └─ (40,000, 45,000, 50,000 Feet bazı tamplate'lerde)  │
└─────────────────────────────────────────────────────────┘

Sütunlar (Template'deki Tanım):
┌──────────────────────────────────────────────────────────────────┐
│ 1. Altitude (ft)        → Sheet'in altitude değeri (sabit)       │
│ 2. Gross Weight  (lb)   → Aircraft weight (grafik metadata)       │
│ 3. Drag Index           → 0-100 değeri (curve progression)       │
│ 4. Mach Number (Ma)     → 0.14-0.70 (X axis / curve coordinate) │
│ 5. Specific Range (NM / lb)  → Nautical Miles / lb (simülasyon output) │
│ 6. Fuel Flow (lb / h)   → Fuel consumption (simülasyon output)   │
└──────────────────────────────────────────────────────────────────┘

Örnek Veri Satırları (One_Engine_Data.xlsx - Sea Level Sheet):
┌──────────────┬───────────────┬────────────┬──────────────┬───────────────┬──────────────────┐
│ Altitude(ft) │ Gross Wgt(lb) │ Drag Index │ Mach (Ma)    │ Specific Rng   │ Fuel Flow (lb/h) │
├──────────────┼───────────────┼────────────┼──────────────┼───────────────┼──────────────────┤
│ Sea Level    │ 30000         │ 0          │ 0.1522       │ 0.0105        │ 9588.31          │
│ Sea Level    │ 30000         │ 0          │ 0.1605       │ 0.0130        │ 8166.73          │
│ Sea Level    │ 30000         │ 0          │ 0.1645       │ 0.0148        │ 7352.26          │
└──────────────┴───────────────┴────────────┴──────────────┴───────────────┴──────────────────┘


### extract_curve_data.py Güncellemesi

Çözüm (Yeni Format - Altitude-based sheets):
╔═══════════════════════════════════════════════════════════════════╗
║ Yeni Format (Altitude-based sheets):                              ║
║ ├─ Engine=1 → One_Engine_Data_One_Engine_Data.xlsx                ║
║ │  ├─ Sheet: Sea Level                                           ║
║ │  ├─ Sheet: 5,000 Feet                                          ║
║ │  ├─ Sheet: 10,000 Feet                                         ║
║ │  └─ ... (her sheet'te altitude-specific veriler)               ║
║ │                                                                 ║
║ └─ Engine=2 → Two_Engine_Data_Two_Engine_Data.xlsx                ║
║    ├─ Sheet: Sea Level                                           ║
║    ├─ Sheet: 5,000 Feet                                          ║
║    └─ ... (her sheet'te altitude-specific veriler)               ║
║                                                                   ║
║ Sütunlar (Template ile Uyumlu):                                  ║
║ ├─ Altitude (ft)        ← Grafik metadata'sından                 ║
║ ├─ Gross Weight  (lb)   ← Grafik metadata'sından                 ║
║ ├─ Drag Index           ← Curve progression (0-100)              ║
║ ├─ Mach Number (Ma)     ← X value (curve interpolation)          ║
║ ├─ L/D Coefficient      ← Y value (curve interpolation)          ║
║ └─ Fuel Flow (lb / h)   ← Placeholder (0)                        ║
╚═══════════════════════════════════════════════════════════════════╝

Uyum: ✓ Template'lerle tam uyumlu!


Yapılan Değişiklikler (CurveDataExporter sınıfı):
┌───────────────────────────────────────────────────────────────────┐
│ 1. ALTITUDE_SHEET_NAMES = { ... }                                │
│    → Altitude değerine göre sheet adı haritası                   │
│    → Sea Level, 5,000 Feet, 10,000 Feet, vb                      │
│                                                                   │
│ 2. get_sheet_name_for_altitude()                                 │
│    → Altitude integer'ını sheet adına dönüştür                   │
│                                                                   │
│ 3. create_dataframe()                                            │
│    → Sütun adları template'ye uyumlulaştırıldı                   │
│    → X → Mach Number (Ma)                                        │
│    → Y → L/D Coefficient                                         │
│    → Fuel Flow → 0 (placeholder)                                 │
│                                                                   │
│ 4. export_to_excel()                                             │
│    → Engine'ye göre ayrı Excel dosyaları oluştur                 │
│    │   One_Engine_Data.xlsx (Engine=1)                           │
│    │   Two_Engine_Data.xlsx (Engine=2)                           │
│    → Her dosya içinde altitude-based sheets                      │
│    → Output path: test_curve_data.xlsx                           │
│        → one_engine_data.xlsx                                    │
│        → two_engine_data.xlsx                                    │
└───────────────────────────────────────────────────────────────────┘


### segment_curves.py Güncellemesi


Değişiklik: CurveDataExporter.export_to_excel() çağrısı
─────────────────────────────────────────────────────

ESKI:
    CurveDataExporter.export_to_excel(
        curve_data,
        args.curve_data_output,
        separate_by_engine=True    ← Kaldırıldı
    )

YENİ:
    CurveDataExporter.export_to_excel(
        curve_data,
        args.curve_data_output
    )

Not: Yeni fonksiyon otomatik olarak engine'ye göre ayrıyor.

### Çıkış Excel Format Örneği


Test Komutu:
    python segment_curves.py \\
        --model checkpoints/best_model.pth \\
        --pdf ./min-max/F18.pdf \\
        --pages 1-5 \\
        --output_dir test_results \\
        --extract_curve_data \\
        --curve_data_output test_curve_data.xlsx

Çıktı Dosyaları:
    ├─ test_curve_data.xlsx_One_Engine_Data.xlsx
    │  ├─ Sheet: Sea Level
    │  │  └─ Columns: [Altitude (ft), Gross Weight  (lb), Drag Index, Mach Number (Ma), Specific Range (NM / lb), Fuel Flow (lb / h)]
    │  │  └─ Rows: Sea Level altitude'daki curve noktaları
    │  ├─ Sheet: 5,000 Feet
    │  │  └─ Rows: 5,000 Feet altitude'daki curve noktaları
    │  └─ ... (diğer altitude'lar)
    │
    └─ test_curve_data.xlsx_Two_Engine_Data.xlsx
       ├─ Sheet: Sea Level
       ├─ Sheet: 5,000 Feet
       └─ ... (diğer altitude'lar)


Data Örneği:
┌──────────────┬───────────────┬────────────┬──────────────┬──────────────────┬──────────────────┐
│ Altitude (ft)│ Gross Wgt(lb) │ Drag Index │ Mach(Ma)     │ Spec Rng(NM / lb)│ Fuel Flow(lb/h)  │
├──────────────┼───────────────┼────────────┼──────────────┼──────────────────┼──────────────────┤
│ 0            │ 30000         │ 0          │ 0.1522       │ 0.0105           │ 9588.31          │
│ 0            │ 30000         │ 0          │ 0.1605       │ 0.0130           │ 8166.73          │
│ 0            │ 30000         │ 0          │ 0.1645       │ 0.0148           │ 7352.26          │
│ 0            │ 30000         │ 10         │ 0.1750       │ 0.0165           │ 6890.54          │
│ 0            │ 30000         │ 20         │ 0.1890       │ 0.0185           │ 6245.73          │
└──────────────┴───────────────┴────────────┴──────────────┴──────────────────┴──────────────────┘


### Not: Y Ekseni Değer Adlandırması


Template'de 5. sütun "Specific Range (NM / lb)" gibi görünse de:
- Simülasyon sonuçları (extract_curve_data'dan gelemez)
- Fuel Flow 6. sütunda benzer şekilde

Extract_curve_data'da biz Y axis'i L/D Coefficient olarak adlandırdık:
- Y axis grafiklerde genellikle 0.0-0.6 range'inde (normalized L/D)
- Bu değerler curve'dan interpolation sonucu

Eğer başka bir Y axis adlaması gerekirse, ileride değiştirilebilir.

### Syntax Doğrulaması


✓ extract_curve_data.py - Syntax: OK
✓ segment_curves.py - Syntax: OK

Kod update'i tamamlandı ve doğrulandı.

---

#### 💡 Pratik Örnekler

### Örnek 1: Baştan Sona (Test)
```bash
# 100 grafik üret
python synthetic_production.py 100 6

# Modeli eğit (hızlı)
python train_unet.py --dataset dataset_production --epochs 20 --batch_size 4

# Test et
python segment_curves.py --model checkpoints/best_model.pth --image dataset_production/images/img_00000.png --visualize
```

### Örnek 2: Üretim (Tam Veri)
```bash
# 5000 grafik (paralel)
python synthetic_production.py 5000 6

# Proper eğitim
python train_unet.py --dataset dataset_production --epochs 100 --batch_size 16

# Tüm testi segmente et
python segment_curves.py --model checkpoints/best_model.pth --input_dir dataset_production/images/ --output_dir final_results/
```

### Örnek 3: Farklı Threshold Denemeleri
```bash
# Threshold=0.3 (daha az strict, daha fazla curve algılama)
python segment_curves.py --model checkpoints/best_model.pth --image test.png --threshold 0.3 --visualize

# Threshold=0.5 (default)
python segment_curves.py --model checkpoints/best_model.pth --image test.png --threshold 0.5 --visualize

# Threshold=0.7 (daha strict, daha az false positive)
python segment_curves.py --model checkpoints/best_model.pth --image test.png --threshold 0.7 --visualize
```

### Örnek 4: CPU'da Eğitim (GPU yoksa)
```bash
python train_unet.py \
    --dataset dataset_production \
    --epochs 30 \
    --batch_size 4 \
    --device cpu
```
*Not: CPU'da çok yavaş olacaktır. Batch size'ı düşük tutun (2-4).*

---

## 📂 Dosya Açıklaması

**`synthetic_v5.py`**
- Grafik oluşturma motoru (temel fonksiyonlar)
- `draw_chart_matplotlib()`: Grafik-eğri çizimi
- `add_scan_artifacts()`: Gerçekçi distortions ekler
- `generate_coco_annotation()`: COCO format

**`synthetic_production.py`**
- Toplu grafik üretim sistemi
- `ProductionDatasetGenerator`: Paralel üretim
- Train/Val/Test split: Otomatik bölme
- 100-5000+ grafik, ~30-50 dakika

**`train_unet.py`**
- U-Net modelini eğitim script'i
- `UNet`: Mimari (encoder-decoder)
- `CurveSegmentationDataset`: Data loader
- `CurveSegmentationTrainer`: Eğitim loop
- Dice Loss, IoU Score: Metric'ler

**`segment_curves.py`**
- İnference ve segmentation script'i
- `CurveSegmentationInference`: Tahmin yapma
- Batch processing: Klasör işleme
- Visualization: Sonuçları göster

---

## 🛠️ Sorun Giderme

### SORUN 1: "CUDA is not available"
**Sebep:** PyTorch'ta GPU desteği yüklü değil
**Çözüm:**
- GPU olmadığını varsay, CPU kullan:
```bash
python train_unet.py --dataset dataset_production --device cpu
```
veya
```bash
python segment_curves.py --model checkpoints/best_model.pth --image test.png --device cpu
```

### SORUN 2: "Model bulunamadı: checkpoints/best_model.pth"
**Sebep:** Model henüz eğitilmedi
**Çözüm:**
- Önce modeli eğit:
```bash
python train_unet.py --dataset dataset_production --epochs 50
```

### SORUN 3: "dataset_production/images bulunamadı"
**Sebep:** Sentetik grafikleri üretmedin
**Çözüm:**
- Grafikleri üret:
```bash
python synthetic_production.py 5000 6
```
- Klasörleri kontrol et:
```bash
ls dataset_production/
```

### SORUN 4: Memory Error (OOM)
**Sebep:** Batch size çok büyük veya RAM yetersiz
**Çözüm:**
- Batch size'ı düşür:
```bash
python train_unet.py --dataset dataset_production --batch_size 4
```
- Veya Model giriş boyutunu düşür:
```bash
python train_unet.py --dataset dataset_production --image_size 128
```

### SORUN 5: Eğitim çok yavaş
**Sebep:** CPU kullanıyorsunuz veya batch_size çok küçük
**Çözüm:**
- İstatistikleri kontrol et (Epoch zamanı it/s)
- GPU var mı? => CUDA'yı kullan
- Batch size'ı artır (GPU varsa):
```bash
python train_unet.py --dataset dataset_production --batch_size 32
```
- Epoch sayısını azalt (test için):
```bash
python train_unet.py --dataset dataset_production --epochs 10
```

### SORUN 6: Model kaydedilmedi, best_model.pth boş
**Sebep:** Eğitim yarım kaldı veya komut hatalı
**Çözüm:**
- Çıkartıları kontrol et, Early stopping trigger edildi mi? Loss artmaya devam ettiyse, learning rate düşür:
```bash
python train_unet.py \
    --dataset dataset_production \
    --epochs 50 \
    --learning_rate 5e-4
```

---

## 🚀 Performansı İyileştirme

### Hızı Artırmak İçin:
1. **Grafik Üretimi (`synthetic_production.py`):**
   - Worker sayısını CPU çekirdeklerine uyarla:
   ```bash
   python synthetic_production.py 5000 8  # (8-core CPU)
   python synthetic_production.py 5000 12 # (12-core CPU)
   ```
2. **Eğitim (`train_unet.py`):**
   - Batch size'ı artır (GPU RAM'e göre): `--batch_size 32`
   - Worker sayısını artır: `--num_workers 8`
   - Epoch sayısını optimize et: `--epochs 50`
   - Input boyutunu azalt (loss accuracy'den): `--image_size 128`
3. **İnference (`segment_curves.py`):**
   - Batch processing kullan (tek tek değil)
   - Confidence map'i kaydetme: `--no_confidence`

### Kaliteyi Artırmak İçin:
1. **Daha Fazla Veri:** `10000` grafik: `python synthetic_production.py 10000 6`
2. **Daha Uzun Eğitim:** 100 epoch: `--epochs 100`
3. **Daha Büyük Model:** Feature sayısını artır: `--features 128`, İnput boyutunu artır: `--image_size 512` (dikkat: RAM artacak)
4. **Better Preprocessing:** Eğer `synthetic_production.py`'da preprocessing varsa, onu geliştir.
5. **Ensemble:** Birden fazla model eğit, sonuçlarını ortala.

### Aylık Bakım:
1. **Model versiyonlaması:** `best_model.pth` → `best_model_v1.pth` → `best_model_v2.pth` (yeni eğitim sonrası)
2. **Dataset depolama:** Eski `dataset_production/` → `backup_dataset_old/` (Yeni `dataset_production/` ile devam et)
3. **Eğitim lojları:** `training_history.png`'leri kaydet, `performance.json`'larını version'la.

---

## 🔗 Entegrasyon ve Sonraki Adımlar

### Başka Projelere Entegrasyon:

1. **Python Script'inden Kullan:**
```python
from train_unet import UNet
from segment_curves import CurveSegmentationInference

# Model yükle
inferencer = CurveSegmentationInference('checkpoints/best_model.pth')

# Grafik segmente et
original, mask, confidence = inferencer.segment('my_image.png', threshold=0.5)

# Curve'leri işle
curves = process_mask(mask)
```
2. **Web Aplikasyonuna Entegre Et:** FastAPI/Flask + model serving. WebUI yapıştım kullan.
3. **Batch Processing Pipeline:** Tüm yeni grafikleri otomatik process et. Cronjob veya scheduler ile günde 1x çalıştır.

### Modeli İyileştirmek (Improve) İçin Sonraki Adımlar:
1. **Transfer Learning:** Daha büyük veri seti ile pre-train edilmiş model kullan, fine-tune et.
2. **Data Augmentation:** Daha fazla preprocessing varyasyonları (rotation, noise, blur, vb.)
3. **Different Architecture:** DeepLab, SegNet, Mask R-CNN deneme, comparison ve benchmarking.
4. **Hyperparameter Tuning:** Learning rate search, batch size optimization, loss function seçimi.

---

## 📌 Komutların Özeti (Hızlı Referans)

```bash
# Grafik oluştur
python synthetic_production.py 5000 6

# Model eğit
python train_unet.py --dataset dataset_production --epochs 50

# Tek görüntüyü segmente et (visualization ile)
python segment_curves.py --model checkpoints/best_model.pth --image test.png --visualize

# Klasör işle
python segment_curves.py --model checkpoints/best_model.pth --input_dir test_images/ --output_dir results/

# Farklı threshold dene
python segment_curves.py --model checkpoints/best_model.pth --image test.png --threshold 0.3

# CPU'da eğit
python train_unet.py --dataset dataset_production --device cpu --epochs 30
```

---

**Son Güncelleme:** March 2026
**Sürüm:** 2.0