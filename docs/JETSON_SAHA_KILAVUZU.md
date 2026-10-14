# Jetson'da kurulum ve benchmark — saha kılavuzu

14 Ekim 2026. Bu kılavuz, cihazın başında terminalden sırayla uygulanmak içindir.
İlk yol **Docker kullanmadan**, repodan sanal ortamda çalıştırmadır. Cihaz üzerinde
eğitim yapılmaz; aynı hazır modellerle çıkarım ölçülür. Ayrıntılı deney açıklaması:
[JETSON_BENCHMARK.md](JETSON_BENCHMARK.md).

## 1. Gitmeden önce yanında bulundur

- Bu kılavuzun yerel kopyası.
- `Specific_Range_Jetson_Hazirlik_20261014.zip`: kaynak kodu ve kısa eğitim modelleri;
  USB'ye kopyala. GitHub'a erişemesen de kaynak/model dosyaların yanında olur.
- `Research_Bundle_Functional_Check.zip`: yalnız model/veri paketi; Git'ten klonlarsan
  bu ZIP'i ayrıca açacaksın. **Modeller Git clone ile gelmez.**
- Paketlerin yanındaki `KONTROL_KAYDI.json`: ZIP SHA-256 değerleri ve kontrol kaydı.
- İnternet erişimi: Python paketleri ve gerekiyorsa NVIDIA PyTorch kurulumu için.

Yanında verilen model paketinin etiketi **functional_check** ve FT eğitimi 5 epoch'tur.
İlk cihaz kurulumunu ve ölçüm yolunu doğrular. Makalede nihai doğruluk kıyası için
son model seçimi/eğitimi ayrı yapılmalıdır; adım 11 bu işlemi açıklar.

Kaynak/model ZIP'i tek başına tam çevrimdışı kurulum paketi değildir. Cihazın
Python/JetPack sürümü bilinmeden doğru ARM64 bağımlılık ve GPU wheel'leri seçilemez.
İnternet olmayacaksa cihaz bilgilerini önceden edinip o sürüme uygun wheel'leri de
USB'ye koymak gerekir. Windows `.venv` klasörünü Jetson'a kopyalama.

## 2. Jetson'u tanı

Jetson'da Terminal aç. Kod bloklarını `$` işareti eklemeden çalıştır:

```bash
uname -m
cat /proc/device-tree/model
cat /etc/nv_tegra_release
python3 --version
free -h
df -h "$HOME"
```

`uname -m` için `aarch64` beklenir. Bu proje ortamı Python **3.10–3.12** için
hazırlanmıştır. Python 3.8/3.9 çıkarsa bu kurulum sırasını zorlayarak devam etme:
eski cihaz/JetPack için ayrı ortam uyarlaması gerekir. Cihazın diskini yeniden
kurmak veya JetPack'i yükseltmek bu benchmarkın zorunlu adımı değildir.

`/etc/nv_tegra_release` L4T sürümünü verir; bu, JetPack sürüm numarasıyla aynı
numara değildir. Kurulu JetPack paket bilgisi varsa aşağıdaki sorgu da yardımcıdır:

```bash
dpkg-query -W nvidia-jetpack
```

Paket bulunamaması tek başına JetPack/CUDA'nın kurulu olmadığı anlamına gelmez.

## 3. Kaynak kodunu al — iki yoldan birini seç

### A. İnternet varsa Git'ten

Git ve temel kurulum araçları eksikse:

```bash
sudo apt-get update
sudo apt-get install git unzip python3-venv python3-pip libopenblas-dev libgomp1
```

Sonra:

```bash
cd "$HOME"
git clone --branch jetson-benchmark --single-branch https://github.com/ufukzkn/specific-range-studio.git specific-range-jetson
cd specific-range-jetson
git rev-parse HEAD
```

Bu klasör zaten varsa tekrar clone yapma. Daha önce bu dalı klonladıysan:

```bash
cd "$HOME/specific-range-jetson"
git status --short
git pull --ff-only
```

Yerel dosya değişiklikleri varsa onları silerek güncelleme yapma; yeni bir klasöre
clone yapabilir veya mevcut dosyalarını koruyarak ilerleyebilirsin.

**Model ZIP'ini aç:** dosyayı dosya yöneticisiyle `Downloads` klasörüne kopyalayıp,
repo klasörünün içindeyken şu komutu çalıştır:

```bash
unzip -n "$HOME/Downloads/Research_Bundle_Functional_Check.zip" -d .
```

### B. Kaynak + model ZIP'inden

`Specific_Range_Jetson_Hazirlik_20261014.zip` dosyasını dosya yöneticisiyle
`Downloads` klasörüne kopyala. `unzip` kurulu olmalı:

```bash
mkdir -p "$HOME/specific-range-jetson-usb"
unzip -n "$HOME/Downloads/Specific_Range_Jetson_Hazirlik_20261014.zip" -d "$HOME/specific-range-jetson-usb"
cd "$HOME/specific-range-jetson-usb"
```

Bu ZIP'in içinde model paketi de vardır; ikinci model ZIP'ini açman gerekmez.
Bu klasör bir Git checkout'u değildir; dosya sürümleri `PACKAGE_MANIFEST.json`
ve benchmarkın kaynak dosya özetleriyle kaydedilir.

**Her iki yolun sonunda kontrol et:**

```bash
ls scripts/run_edge_benchmark.py
ls artifacts/research_bundle_portable/manifest.json
ls artifacts/research_bundle_portable/xgboost/model.json
ls artifacts/research_bundle_portable/ft_transformer/model.pt
```

Dört dosya da görünmelidir. Bundan sonraki bütün komutlar bu proje klasöründeyken
çalıştırılacak. Yanlışlıkla alt klasöre geçtiysen `cd "$HOME/specific-range-jetson"`
veya USB yolundaki klasöre geri dön.

## 4. Python ortamını kur

İlk CPU koşusu için mevcut NVIDIA kurulumunu değiştirmeyen ayrı bir ortam oluştur:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-edge.txt
python -m pip install --only-binary=:all: "torch==2.5.1"
python -m pip check
```

Bu PyTorch kurulumu ilk **CPU ölçümü** içindir; Jetson GPU desteği sağladığı
varsayılmaz. PyPI'nin 2.5.1 dağıtımında Python 3.10/3.11/3.12 için aarch64 wheel'leri
bulunuyor; cihazındaki uyumluluk aşağıdaki gerçek model kontrolüyle değerlendirilecek.
[PyTorch 2.5.1 dosyaları](https://pypi.org/project/torch/2.5.1/#files).

`--only-binary=:all:` uygun wheel yoksa kaynak koddan uzun bir derlemeye başlamadan
hata verir. `pip check` uyumsuzluk bildirirse hatayı kaydet ve önce ortamı düzelt.
Jetson'da bütün `requirements.txt` dosyasını kurmana gerek yok; benchmark için
`requirements-edge.txt` kullanılır. Uygulamanın Dataset Builder/OCR araçları bu kurulumun
parçası değildir.

Yeni terminal açarsan ortamı tekrar etkinleştir:

```bash
cd "$HOME/specific-range-jetson"
source .venv-cpu/bin/activate
```

USB yolunu kullandıysan ilk satırdaki klasör adını ona göre değiştir.

## 5. Dosyaları ve ortamı doğrula

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from pathlib import Path; from src.evaluation.research_bundle import verify_bundle; m=verify_bundle(Path('artifacts/research_bundle_portable')); print('Paket doğrulandı:', m['experiment_label']); print(m['data_audit']['partition_rows'])"
mkdir -p artifacts/device
python scripts/inspect_edge_environment.py > artifacts/device/jetson_environment_cpu.json
```

Paket doğrulama çıktısında `functional_check` ve train/valid/test satır sayıları
görünmelidir. CPU ortamında `CUDA: False` çıkması CPU ölçümünü engellemez. CUDA
ölçümü için adım 8'i uygula.

İsteğe bağlı kod/gerçek model kontrolleri:

```bash
python -m unittest discover -s tests -v
```

Hazır model paketinin beklenen klasörde olması durumunda gerçek tahmin ve rapor
kontrolleri de çalışır; model bulunmadığı için skip edilen kontroller tam doğrulama
sayılmaz.

## 6. Önce kısa CPU denemesi

```bash
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cpu --sample-size 24 --warmup 5 --repetitions 20 --trials 1 --require-jetson --output artifacts/jetson_smoke_cpu
```

Beklenen: üç yöntemin p95 satırı, sonunda `complete=True, Jetson=True`.
Bu kısa koşu kurulum testidir; nihai performans ölçümü değildir. Bir model başarısız
olursa çıktı klasöründeki ilgili `*.log` dosyasını aç. `--require-jetson` hatasını
gizlemek için bu seçeneği kaldırarak sonucu Jetson ölçümü diye sunma.

## 7. CPU benchmarkını al

Arayüz sunucusunu, eğitim işlerini ve gereksiz ağır uygulamaları kapat. Aynı güç modu,
soğutma ve thread ayarlarıyla ölçüm yap. Şu betik güç modunu/saatleri değiştirmez.

```bash
RUN_GROUP="jetson_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$RUN_GROUP" > artifacts/device/last_run_group.txt
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cpu --sample-size 200 --warmup 50 --repetitions 1000 --trials 3 --threads 1 --seed 42 --require-jetson --output "artifacts/$RUN_GROUP/cpu"
```

Üç yöntem × üç tekrar vardır. Aynı test örnekleri kullanılır. Çıktı `cpu` klasöründe
benzersiz `edge_*` alt klasörü ve `edge_benchmark_latest.json` olur. Doğruluk bu 200
test örneği içindir; bütün test kümesi değildir. Paket etiketi hâlâ `functional_check`
olduğundan bu koşunun hatalarını nihai eğitilmiş model sonuçları gibi yazma.

## 8. FT-Transformer GPU ortamını hazırla

GPU koşusu için **JetPack ile uyumlu NVIDIA PyTorch** gerekir. Cihaz modeli/JetPack
bilinmediğinden tek bir wheel URL'si bütün cihazlar için doğru değildir.
[NVIDIA uyumluluk tablosu](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
ve [NVIDIA kurulum adımları](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
ile kendi sürümüne ait dosyayı seç. `cp310` Python 3.10, `cp312` Python 3.12 wheel'idir;
`aarch64` mimarisiyle ve cihazın CUDA sürümüyle de eşleşmelidir. NVIDIA'nın 24.06 ve
sonraki PyTorch sürümlerinde ayrıca cuSPARSELt önkoşulu vardır.

**Cihazın sistem Python'unda NVIDIA PyTorch zaten çalışıyorsa:**

```bash
deactivate
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Son satır `True` ise aynı sistem PyTorch'unu gören ayrı GPU ortamı kurabilirsin:

```bash
python3 -m venv --system-site-packages .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --only-binary=:all: -r requirements-edge.txt
python -m pip check
```

**Sistem Python'unda uygun PyTorch yoksa:**

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-edge.txt
```

Sonra NVIDIA kılavuzundaki cihazına uygun önkoşulları uygula ve seçtiğin NVIDIA
wheel'ini indir. Örneğin dosyayı `Downloads` klasörüne koyduktan sonra:

```bash
read -r -p 'İndirdiğin NVIDIA torch wheel dosyasının tam yolu: ' TORCH_WHEEL
python -m pip install "$TORCH_WHEEL"
python -m pip check
```

Yukarıdaki komut gerçek dosya yolunu sorar; `torch-...whl` gibi bir yer tutucuyu
aynen yazma. Bu ortamda genel PyPI Torch komutunu tekrar çalıştırmak NVIDIA
kurulumunu değiştirebilir. Model ön işleyicisinin NumPy/pandas/SciPy/scikit-learn/joblib
sürümleri paketteki sürümlerle aynı kalmalıdır; CUDA wheel'i bunlarla uyumlu olmalıdır.

**GPU'yu gerçek işlemle doğrula:**

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA görünmüyor'; x=torch.ones(1, device='cuda'); torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x.item())"
python scripts/inspect_edge_environment.py > artifacts/device/jetson_environment_gpu.json
```

Bu geçmeden GPU benchmarkına geçme. GPU kurulumu çözülemezse başarılı CPU koşusu
korunur; GPU ölçümü yapılmış sayılmaz. Docker taban imajı seçimi de aynı cihaz
eşleştirmesine bağlıdır; ilk cihaz denemesinde gerekli değildir.

## 9. Kısa GPU denemesi, ardından GPU benchmarkı

```bash
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cuda --sample-size 24 --warmup 5 --repetitions 20 --trials 1 --require-jetson --output artifacts/jetson_smoke_gpu
```

Başarılıysa, CPU koşusunun aynı grup adını geri alıp tam koşuyu çalıştır:

```bash
RUN_GROUP="$(cat artifacts/device/last_run_group.txt)"
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cuda --sample-size 200 --warmup 50 --repetitions 1000 --trials 3 --threads 1 --seed 42 --require-jetson --output "artifacts/$RUN_GROUP/gpu"
```

Bu koşuda **yalnız FT-Transformer GPU kullanır**; XGBoost ve spline CPU'da kalır.
`device` alanında FT için `cuda`, diğer ikisi için `cpu` beklenir. Aynı güç/soğutma
koşullarını koru. Script CUDA yokken FT'yi sessizce CPU'da çalıştırmaz.

## 10. Sonuçları gör ve bilgisayarına götür

CPU ortamı da GPU ortamı da arayüzü açabilir; arayüz açıkken benchmark çalıştırma:

```bash
RUN_GROUP="$(cat artifacts/device/last_run_group.txt)"
export SPECIFIC_RANGE_BUNDLE="$PWD/artifacts/research_bundle_portable"
export SPECIFIC_RANGE_EDGE_RESULTS="$PWD/artifacts/$RUN_GROUP/cpu/edge_benchmark_latest.json"
export SPECIFIC_RANGE_NO_BROWSER=1
python scripts/web_app/server.py
```

Jetson tarayıcısında `http://localhost:5000` aç. `Maliyet` sekmesi yöntem/tekrar,
gerçek cihaz, p50/p95/p99, örneklenmiş RAM, CPU, RMSE ve MAPE'yi gösterir. GPU raporu
için ortam değişkenindeki `/cpu/` bölümünü `/gpu/` yapıp sunucuyu yeniden başlat.
Sunucuyu terminalde `Ctrl+C` ile durdur.

Sonuçları doğrula:

```bash
python -c "import json,os; from pathlib import Path; base=Path('artifacts')/os.environ.get('RUN_GROUP', Path('artifacts/device/last_run_group.txt').read_text().strip()); p=base/'cpu/edge_benchmark_latest.json'; d=json.loads(p.read_text()); print('Tamamlandı:', d['complete'], 'Jetson:', d['verified_jetson'], 'Model etiketi:', d['experiment_label']); print([(x['model_key'], x.get('device'), x['available']) for x in d['trials']])"
```

GPU sonuçları için bu komuttaki `cpu/edge_benchmark_latest.json` yerine
`gpu/edge_benchmark_latest.json` yaz. `complete=false` veya `available=false` varsa
ilgili `.log` dosyasını da incele.

Sunucu kapalıyken, proje klasöründen bir sonuç arşivi oluştur:

```bash
RUN_GROUP="$(cat artifacts/device/last_run_group.txt)"
tar -czf "$HOME/${RUN_GROUP}_sonuclar.tar.gz" "artifacts/$RUN_GROUP" artifacts/device artifacts/research_bundle_portable/manifest.json
```

Bu `.tar.gz` dosyasını USB'ye veya kendi bilgisayarına kopyala. Yalnız tablo ekran
görüntüsünü alma; arşivde bütün ham çağrı süreleri, tahminler, JSON özetleri, hata
kayıtları ve varsa UTC damgalı `tegrastats.log` birlikte kalır. Her koşudaki
`summary.csv` doğrudan Excel'de açılabilir.

## 11. Makale için nihai model paketi hazırlama

Bu adım **iş istasyonunda** yapılır, Jetson çıkarım deneyinin parçası değildir.
Aynı ortak bağımlılık sürümleriyle ve çalışan PyTorch ortamıyla:

```bash
python scripts/run_data_pipeline.py
python scripts/prepare_research_bundle.py --ft-device cpu --epochs 50 --output artifacts/research_bundle_final --label research_candidate
python scripts/export_jetson_package.py --bundle artifacts/research_bundle_final --output Specific_Range_Jetson_Final.zip
```

CUDA iş istasyonu için `--ft-device cuda` kullanılabilir. 50 epoch bir model seçimi
başarısı garantisi değildir; hiperparametre seçimi/doğrulama ayrı çalışmadır.
Export betiği, seçilen paketi dağıtımda `artifacts/research_bundle_portable/` yolu
altına koyar; yeni ZIP'i yeni klasöre açınca saha komutlarını aynı yollarla
kullanabilirsin. Eski/yeni modelleri aynı çıktı klasörüne karıştırma. Bütün cihazlarda
aynı nihai paket/manifest kullanılmalı; kısa ve nihai paket sonuçları ayrı tutulmalıdır.

## Sık karşılaşılan sorunlar

| Durum | Yapılacak işlem |
|---|---|
| `manifest.json` yok | Modeller Git'te değildir; model ZIP'ini repo köküne aç. `artifacts/research_bundle_portable/` iç içe iki kez oluşmadığını kontrol et. |
| `version mismatch` | Aktif ortamı kontrol et. Paketin `requirements-portable.txt` dosyasındaki sürümleri kur; NVIDIA Torch sürümünü ayrıca koru. |
| `Bundle integrity failed` | Paket içindeki dosyaları düzenleme. Orijinal ZIP'i yeni klasöre aç; farklı koşul/model dosyalarını birleştirme. |
| `CUDA requested but unavailable` | GPU ortamını etkinleştir; NVIDIA wheel/JetPack eşleşmesini ve adım 8'deki gerçek CUDA işlemini doğrula. |
| `Jetson identity could not be verified` | Model/L4T dosyalarının okunabildiğini ve `aarch64` çıktısını kontrol et. Container yerine önce native ortamda dene. |
| `No matching distribution` | Python sürümü/ARM wheel uyumu eksik; kaynak derleme veya JetPack güncellemesine otomatik geçme. |
| `ModuleNotFoundError` | Doğru `.venv-cpu` / `.venv-gpu` etkin mi kontrol et; `python -m pip` aynı interpreter'a kurar. |
| Güç/telemetri bilgisi eksik | Mevcutsa `nvpmodel -q` ve `jetson_clocks --show` sorgularını yetkili terminalde ayrıca kaydet. Ölçülmeyen alanları sayıyla doldurma. |
| İşlem kapanıyor / bellek yetmiyor | Diğer ağır uygulamaları kapat; `.log` ve cihaz RAM durumunu incele. Yöntemler zaten ayrı süreçlerde çalışır; kısa denemeyle başla. |

Bu ölçüm, bilinen eğrilerin ayrılmış iç Mach düğümleri üzerinde yapılır. Verinin
fiziksel doğruluğu, yeni uçuş/eğri genellemesi, gerçek zaman garantisi ve sertifikasyon
sonucu değildir. Cihaz üzerinde yapılan işlem ve alınan ham kayıtlar raporun temelidir.
