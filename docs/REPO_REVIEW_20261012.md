# Repo incelemesi ve Jetson hazırlık kaydı — 12 Ekim 2026

14 Ekim devam notu: Bu belge 12 Ekim inceleme ve çalıştırma kayıtlarını korur.
Güncel adım adım kurulum/aktarım kılavuzu [JETSON_SAHA_KILAVUZU.md](JETSON_SAHA_KILAVUZU.md)
içindedir. Commit/push için 14 Ekim'de kullanıcı yetkisi verilmiştir; aşağıdaki
12 Ekim'deki uzak depo/teslim durumu tarihsel kayıttır.

## Sonuç

İstenen çekirdek amaç mevcut: uçak performans tablosundan **özgül menzil** tahmini
için XGBoost, sayısal özellik tokenizasyonlu tabular Transformer ve kübik spline
interpolasyonu uygulanmış. Bu, bütün uçuş performans parametrelerini hesaplayan
genel bir uçuş mekaniği çözücüsü değildir; mevcut hedef `specific_range` sütunudur.
Yakıt debisi için ayrı bir regresyon hedefi bu tur eklenmedi.

Başlangıç sürümü `8ba72f1a5086cdb9319ffe0051439a58738513d3`, uzak depo
[ufukzkn/specific-range-studio](https://github.com/ufukzkn/specific-range-studio).
Klon çalışma alanının `work/specific-range-studio/` altındadır. Değişiklikler yerel
`jetson-benchmark` dalındadır; uzak depo değiştirilmedi. Yeni ajan/görev açılmadı.

## Koddan doğrulanan yöntemler

| İstek | Mevcut kod ve gerçek kapsam |
|---|---|
| XGBoost regresyonu | `src/models/xgboost_baseline.py`: gerçek `xgboost.XGBRegressor`, ağaç eğitimi/tahmini, JSON model kaydı. |
| Tabular Transformer | `src/models/ft_transformer.py`: özellik başına sayısal affine tokenizer, kategori embeddingleri, CLS, PyTorch TransformerEncoder ve regresyon başlığı. Sınıf kendi tanımında **FT-Transformer-style**; özgün yayının birebir kodu diye adlandırılmamalı. GELU/norm-first uyarlaması korunmuştur. |
| Kübik interpolasyon | `src/interpolation/specific_range.py`: doğal kübik spline; Mach → drag → ağırlık → irtifa boyunca ardışık uygulama. İki noktada doğrusal, tek diğer-eksen düğümünde sabit davranış; Mach sınırı dışında doğrusal uzatma ve negatif çıktıların sıfıra kırpılması vardır. |
| Kullanıcı tahmini | Flask `/api/predict`: üç yöntemi aynı koşulda çalıştırabilir. |
| Mevcut benchmark | `src/evaluation/pi_benchmark.py`: gerçek Python çıkarımı var; Pi sıcaklık/throttle sorguları Jetson telemetrisi yerine geçmez. |
| ONNX/TensorRT | `src/evaluation/benchmark.py` export/build fonksiyonları `NotImplementedError`; bunlar tamamlanmış dağıtım bileşeni değildir. |

## Karşılaştırmayı bozan bulgular ve düzeltmeler

1. **Yakıt debisiyle ortak olmayan girdiler.** Eski ML yolu beş sayısal girdi kullanıyor,
   spline dört kullanıyordu. Okunan tablolarda her irtifada
   `fuel_flow × specific_range / mach` sayısal olarak sabit. Bu güçlü hedef ilişkisi
   bağımsız ölçülmüş bir yakıt girdisi kanıtı değildir. Yeni araştırma profili iki
   modele de yalnız irtifa/ağırlık/drag/Mach ve motor tipini verir; yakıt debisi bağlam
   olarak dosyada kalır. Eski artefaktlar ve legacy eğitim yolu silinmedi.
2. **Test ayrımı olmayan benchmark doğruluğu.** Eski Pi betiği bütün tablodan örnek
   seçiyordu; eğitim satırı seçebilirdi. Yeni betik yalnız dondurulmuş `test.csv`
   içinden aynı örnekleri seçer. Spline referansı da yalnız `train.csv` düğümleridir.
3. **Aynı koşulun tekrarları/çelişkileri.** 803 satır aynı dört sayısal koşul ve motor
   tipinde farklı hedefler içeriyor; tamamı `quarantine.csv` içine alınmıştır. Aynı
   koşul/hedef tekrarlarından 49 satır çıkarılmıştır. Çelişkili hedeflere ortalama
   alınıp doğrulanmış etiket gibi davranılmamıştır. Sayısal eşitlik kriteri tam
   eşitliktir; yaklaşık yinelenmeler ayrıca araştırılabilir.
4. **Deniz seviyesinin düşmesi.** `Sea Level` metni son sayısal dönüşümde NaN oluyordu.
   Loader düzeltildi; temiz çıktı 54.620'den 62.740 satıra yükseldi. 8.120 satır geri
   kazanıldı; kalan bir satır eksik zorunlu alan nedeniyle düşüyor. ExcelFile handle'ı
   kapatılmadığı için Windows'ta dosya kilidi oluşması da giderildi.
5. **Ölçülmemiş interpolasyon değerleri.** Flask model özetindeki sabit RMSE/MAE/MAPE/R²
   ve 0,25 MB değeri kaldırıldı. Hata metrikleri yeni benchmarkın gerçek ortak test
   örneklerinden gelir; saklanan referans dosyasının boyutu gerçek dosyadan alınır.
6. **Başka ortamın ölçümüne otomatik geçiş.** Yerel Pi raporu yoksa repo içindeki
   reference JSON'un gösterilmesi kaldırıldı. Araştırma modundaki maliyet ekranı yeni
   edge raporunu okur ve cihaz/deney etiketiyle gösterir.
7. **Kaynak ölçümünün karışması.** Yeni runner her yöntem/tekrarı ayrı süreçte çalıştırır.
   CPU süresi ısınmadan sonra başlar. FT cihazı açık kaydedilir; CUDA isteğinde CPU'ya
   sessiz düşüş engellenir. RSS örnekleme, CUDA allocator ve disk boyutu ayrı kaydedilir.

## Güncel dondurulmuş veri

| Alan | Sayı |
|---|---:|
| Loader sonrası satır | 62.740 |
| Çelişkili koşul satırı, karantina | 803 |
| Aynı koşul/hedef tekrarı çıkarılan | 49 |
| Deneyde kalan benzersiz koşul | 61.888 |
| Eğitim | 44.193 |
| Doğrulama | 8.847 |
| Test | 8.848 |

Her eğrinin Mach uçları eğitimde tutuldu; iç düğümlerin %30'u seed=42 ile ayrılıp
doğrulama/test arasında paylaştırıldı. Uçları sabitleme nedeniyle bütün tablodaki
oranlar tam 70/15/15 değildir. Aynı koşul hiçbir ayrımda tekrarlanmaz. Bu protokol
**bilinen eğrilerin görülmemiş iç Mach düğümleri** için geçerlidir; görülmemiş uçuş,
uçak veya bütün eğri genellemesi iddiası değildir.

Dosyalar kullanıcının sayısallaştırılmış tabloları olarak ele alındı. Ham Excel
başlığındaki `Specific Range (NM)` ifadesi tek başına özgül menzilin fiziksel birimini
doğrulamıyor; yeni grafikler/raporlar kaynak hedefinin sayısal birimini korur. Nomogram
kaynağı/sürümü ve sayısallaştırma doğruluğu bu kod kontrolüyle bağımsız doğrulanmış
sayılmaz. Karantina politikası nihai veri temizliği onayı değildir.

## Yerel doğrulama

Ortak sürümler: NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, scikit-learn 1.5.2,
XGBoost 2.1.4, joblib 1.4.2, openpyxl 3.1.5, psutil 7.2.2. Yerel PyTorch
2.10.0+cu130; FT kısa eğitiminde CUDA kullanıldı. Jetson wheel'i ayrıca seçilecektir.
Yerel test ortamı mevcut sistem paketlerini gören bir `.venv`dir; global paketler
değiştirilmedi. Dataset Builder/OpenCV iş akışı bu test kapsamına dahil değildir.

- **15 unittest kontrolü geçti.** Deniz seviyesi okuma, doğal spline–SciPy uyumu,
  çelişki karantinası, ayrımların ayrıklığı, uç sınırlar, seed tekrarlanabilirliği,
  bundle bütünlüğü, iki motor kategorisinin gerçek model tahmini, yakıt debisi
  değişiminden bağımsız çıktı ve arayüz raporlarının yalnız test satırlarını içermesi.
- Gerçek XGBoost 300 ağaçla ve FT 5 epoch ile kısa işlev eğitimi yapıldı. Artefaktlar
  `artifacts/research_bundle_portable/` içinde, etiketi **functional_check**.
- CPU'da üç yöntem × iki bağımsız süreç tekrarı, 24 ortak test örneği, 5 ısınma,
  20 zamanlı çağrı çalıştı. Son kayıt:
  `artifacts/benchmarks/edge_20261012T195915Z_d140fdbf/`. `complete=true`,
  `verified_jetson=false`. Ham p95 değerleri özetle; tahmin dosyalarının satır kimlikleri
  ortak örnekle; altı ayrı PID ile süreç izolasyonu ayrıca karşılaştırıldı. Manifest
  özeti mevcut paketle eşleşti.
- FT GPU yolu da yerel CUDA'da, aynı 24 örnek ve kısa prosedürle çalıştı:
  `artifacts/benchmarks/edge_20261012T195149Z_056be07e/`. Bu daha eski koşu rapor
  PNG/CSV'leri eklenmeden önceki manifesti kaydeder; model dosyaları aynıdır.
- `--require-jetson` bu Windows/x86_64 makinede cihaz kimliği hatası vererek çalışmayı
  durdurdu. Bu makineye Jetson etiketi atanmadı.
- Flask gerçek tarayıcıda Playwright CLI ile açıldı. Genel bakış test metrikleri ve
  maliyet tablosu görüldü; maliyet ekran görüntüsü
  `output/playwright/research-benchmark.png` görsel kontrol edildi.
- JavaScript sözdizimi ve `git diff --check` geçti.

Bu süre/örnek sayıları işlev kontrolü içindir. Yerelde eşzamanlı arayüz/kurulum
kontrolleri de yapıldığı için gecikmeler nihai idle-machine performans sonucu değildir.
Bu tur herhangi bir modelin genel üstünlüğü veya Jetson yeterliliği sonucu çıkarılmadı.

## Teslim ve açık kalan işler

Çekirdek kaynak, iki Excel girdisi, doğrulanmış kısa eğitim modeli ve test raporlarını
aktaracak `scripts/export_jetson_package.py` eklendi. Sanal ortam, Git geçmişi,
büyük arşiv raporları ve Dataset Builder demo varlıkları exporta dahil değildir.
Her dosyanın özeti paket manifestinde tutulur. Yeniden kurulum/benchmark komutları
[JETSON_BENCHMARK.md](JETSON_BENCHMARK.md) içindedir.

Yeni kaynak üretildi ve yerel kontroller tamamlandı; kullanıcı kabulü yok. Hedef
Jetson modeli/JetPack henüz bilinmiyor, gerçek Jetson'a erişilmedi. Gerçek cihaz
kurulumu/CPU–FT GPU ölçümleri, nihai eğitim ve veri kökeni/doğruluğu değerlendirmesi
açıktır. Docker yalnız cihazla eşleştirilecek bir tarif olarak hazırlandı; build
veya dağıtım yapılmadı. Akademik Word taslakları ve eski yerel D: uygulaması değişmedi.
