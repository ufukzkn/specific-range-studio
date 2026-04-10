# Agent Notes

Bu dosya proje envanteri, arayüz davranışları, önemli komutlar ve UX kararları için yaşayan referans notudur. Amaç: ileride bir sorun olduğunda veya yeni ajan devraldığında hızlıca bağlam kurabilmek.

## Proje Özeti

- Amaç: uçak performans tablolarından `specific_range` tahmini
- Veri kaynağı:
  - `One_Engine_Data.xlsx`
  - `Two_Engine_Data.xlsx`
- Ana modeller:
  - `FT-Transformer` ana odak
  - `XGBoost` baseline / kıyas
- Ek hedefler:
  - PSO ile optimizasyon
  - full-table comparison raporları
  - taslak nomogram / handbook-style grafik karşılaştırması
  - gelecekte ONNX / TensorRT / Jetson genişletmesi

## Önemli Yol Haritası

1. Excel verisini temiz CSV haline getir
2. Model(ler)i eğit
3. Full-table report üret
4. Hazır rapor ekranından sonuçları incele
5. Gerekirse tekil custom input ile ara değer test et
6. Taslak nomogram ile handbook-style grafik üret

## Ana Komutlar

```bash
python scripts/run_data_pipeline.py
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
python scripts/run_table_report.py --dataset data/processed/combined_specific_range.csv --model both --ft-device cuda
streamlit run scripts/launch_ui.py
python scripts/desktop_app_qt.py
```

## Proje Yapısı

- `src/data/load_data.py`: workbook okuma, sheet dolaşma, kolon normalize etme, `engine_type` ekleme
- `src/data/preprocess.py`: preprocessing, numerik/kategorik ayırma, clipping desteği
- `src/data/split.py`: train/valid/test split
- `src/models/xgboost_baseline.py`: XGBoost wrapper
- `src/models/ft_transformer.py`: FT-Transformer trainer/checkpoint mantığı
- `src/evaluation/table_report.py`: full-table row-level report ve PNG grafik üretimi
- `src/evaluation/nomogram_report.py`: taslak handbook-style slice/curve grafiği
- `src/inference/predictors.py`: artifact-backed predictorlar, hazır test senaryoları, exact/nearest real row yardımcıları
- `scripts/launch_ui.py`: Streamlit arayüz
- `scripts/desktop_app_qt.py`: modern Qt masaüstü arayüz

## Veri ve Artefactlar

- Temiz veri:
  - `data/processed/combined_specific_range.csv`
- XGBoost artefactları:
  - `artifacts/xgboost/model.json`
  - `artifacts/xgboost/preprocessor.joblib`
  - `artifacts/xgboost/metrics.json`
  - `artifacts/xgboost/reports/*.csv`
  - `artifacts/xgboost/reports/*.xlsx`
  - `artifacts/xgboost/reports/*.png`
- FT artefactları:
  - `artifacts/ft_transformer/model.pt`
  - `artifacts/ft_transformer/preprocessor.joblib`
  - `artifacts/ft_transformer/metrics.json`
  - `artifacts/ft_transformer/reports/*.csv`
  - `artifacts/ft_transformer/reports/*.xlsx`
  - `artifacts/ft_transformer/reports/*.png`

## Streamlit Arayüz Özellikleri

Dosya: `scripts/launch_ui.py`

### Hazır Rapor

- `Ozet` sekmesi
  - `Rows`
  - `MAE`
  - `RMSE`
  - `MAPE`
  - `R2`
  - `Slice Ozeti`
- `Satir Bazli Karsilastirma` sekmesi
  - `engine_type` filtresi
  - `altitude` filtresi
  - `absolute error` eşiği
  - sıralama modu
  - seçili satır detayı
  - CSV indirme
- `Grafikler` sekmesi
  - rapor PNG önizlemeleri
  - önizlemeye tıklayınca sayfa içi büyük önizleme
  - PNG indirme
- `Taslak Nomogram` sekmesi
  - `engine_type`
  - `altitude`
  - `gross_weight`
  seçilerek nomogram üretimi

### Tekil Tahmin

- Yeni başlık: `Tekil Tahmin (Custom Input)`
- Hazır test senaryosu listesi
- Manuel giriş alanları:
  - `altitude`
  - `gross_weight`
  - `drag_index`
  - `mach`
  - `fuel_flow`
  - `engine_type`
- Model sonucu kartları
- Gerçek veriyle kıyas:
  - exact match varsa actual ve absolute error
  - yoksa nearest real rows ve nearest actual farkı

### Setup

- Veri pipeline çalıştırma
- XGBoost eğitim + rapor
- FT-Transformer eğitim + rapor
- Toplu rapor üretme
- PSO çalıştırma
- Model karşılaştırma
- Quickstart akışını UI içinden tetikleme
- Her komut için:
  - tahmini süre
  - üreteceği artefactlar
- Log alanı
- Quickstart için ilerleme çubuğu

## Qt Masaüstü Uygulama Özellikleri

Dosya: `scripts/desktop_app_qt.py`

### Setup

- Quickstart komut önizlemesi
- tahmini süre ve artefact özeti
- cihaz seçimi:
  - XGBoost `cpu/cuda`
  - FT `cpu/cuda`
- butonlar:
  - `Quickstart Setup`
  - `Sadece Veri Pipeline`
  - `Sadece Toplu Rapor`
  - `XGBoost egit + rapor`
  - `FT-Transformer egit + rapor`
  - `PSO calistir`
  - `Model karsilastir`
- canlı log alanı
- ilerleme çubuğu

### Hazır Rapor

- `Rapor Kontrolleri` kartı
- model seçimi
- FT cihaz seçimi
- önerilen akış kutusu
- filtreler:
  - `engine_type`
  - `altitude`
  - `error` bandı
  - sıralama
- rapor özeti
- row-level comparison tablosu
- seçili satır detay paneli
- iki büyük grafik önizlemesi
- her grafik için:
  - ayrı pencerede aç
  - dosyayı aç
  - klasörü aç

### Tekil Tahmin

- hazır test senaryosu seçimi
- manuel giriş formu
- model tahmini
- exact / nearest match kıyası
- nearest rows tablosu

### Taslak Nomogram

- parametre formu
- nomogram üretimi
- nomogram önizlemesi
- ayrı pencere / dosya / klasör açma

### Qt Görsel Görüntüleyici

- sol tık: zoom in / zoom out toggle
- orta mouse: panning / sürükleyerek gezinme
- butonlar:
  - `Zoom +`
  - `Zoom -`
  - `Sigdır`
  - `Tam ekran`

## Hazır Test Senaryoları

Kaynak: `src/inference/predictors.py`

- `Custom input`
- her `engine_type` için birden fazla `Exact row`
- daha fazla `Intermediate` ara değer senaryosu
- senaryo üst limiti artırıldı

## Bilinen Çevre Notları

- `.venv` içinde PyTorch CUDA destekli
- FT-Transformer `--device cuda` ile gerçek GPU kullanabiliyor
- XGBoost eğitimde GPU kullanabiliyor ama prediction/report tarafında CPU input yüzünden warning görülebilir
- `cupy` / `cudf` yoksa XGBoost GPU prediction tam optimize değil

## UX Notları

### Güçlü Taraflar

- Hazır rapor ve tekil tahmin akışları ayrılmış durumda
- Setup ekranı quickstart karşılığı olduğu için demoda da geliştirmede de işe yarıyor
- Tekil tahminde gerçek veriyle kıyas görünür
- Raporlarda filtreleme ve satır detayı var

### İyileştirme Fırsatları

- Streamlit lightbox içinde gerçek `+/-` zoom kontrolleri eklenebilir
- Setup ekranında daha net durum rozetleri olabilir
- Demo modu için sade bir “presentation mode” düşünülebilir

## Not

- Kullanıcı bu dosyayı ileride “sorun olursa danışılacak referans” olarak kullanmak istiyor.
- Bu yüzden yeni eklenen her önemli arayüz özelliği, kırık durum veya environment notu burada güncellenmeli.
