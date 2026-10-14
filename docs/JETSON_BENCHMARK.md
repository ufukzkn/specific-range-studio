# Jetson benchmark hazırlığı — güncelleme 14 Ekim 2026

Jetson'a gittiğinde sırayla uygulayacağın kısa kılavuz:
[JETSON_SAHA_KILAVUZU.md](JETSON_SAHA_KILAVUZU.md). Bu belge ayrıntılı deney protokolüdür.

## Karar ve sınır

İlk ölçüm **repodan, Python sanal ortamında** yapılmalı. Bu, aynı model ve test paketinin
CPU yolunu doğrulamak ve cihaz ortamını görmek için uygun başlangıçtır. Docker seçeneği
hazır bir tarif olarak eklendi; Jetson modeli/JetPack bilinmediği için belirli bir GPU
taban imajı seçilmedi ve Docker build/Jetson çalıştırması yapılmış sayılmaz.

Benchmarkın çalışması için ONNX veya TensorRT şart değildir. İlk karşılaştırma doğal
XGBoost CPU, mevcut PyTorch FT CPU/GPU ve kübik spline CPU çalışma yollarını ölçer.
ONNX/TensorRT/FP16/INT8 bir sonraki, ayrı deney olacaktır; mevcut export stubları
gerçek dönüşüm veya ölçüm olarak kullanılmaz. XGBoost GPU bu ilk pakette kullanılmaz.

Bu hazırlıkta verilen modeller **5 epoch işlev kontrolü** içindir ve manifestte
`functional_check` olarak etiketlidir. Nihai bilimsel model seçimi veya Jetson
üstünlük sonucu değildir. Nihai eğitim için aşağıdaki 50 epoch komutu yeni bir
klasörde çalıştırılabilir; hiperparametre seçimi ayrıca doğrulama kümesiyle yapılır.

## Önce cihazı tanı

Jetson terminalinde, kurulumdan önce:

```bash
uname -m
cat /proc/device-tree/model
cat /etc/nv_tegra_release
python3 --version
```

Bu sürüm Python **3.10 veya üstünü** hedefler. Eski JetPack/Python kombinasyonunda
bu requirements dosyasını zorlayarak kurmayın; cihaz sürümüne uygun ayrı ortam
gerekecektir. Model/JetPack/RAM bilgisi bilinmeden tüm Jetson ailesi için uyumluluk
garantisi verilemez. Cihazda paketler kurulduktan sonra ayrıntılı kayıt:

```bash
python scripts/inspect_edge_environment.py > jetson_environment.json
```

Betik model, L4T, Python, RAM, paket sürümleri, CUDA ve erişilebilen güç/saat ayarlarını
okur. `nvpmodel` ve `jetson_clocks` durum sorguları yetki gerektirirse hata kayda girer;
betik güç modunu, saatleri veya fanı değiştirmez.

## İş istasyonunda ortak modeli oluştur

Kaynak `jetson-benchmark` dalındadır. Uzak depodan bu dalı seçerek klonlayın;
yalnız `main` dalını klonlamak bu çalışma dalının betiklerini getirmez. Eğitilmiş
modeller Git'te tutulmaz; hazırlık/model ZIP'inden veya iş istasyonundan ayrıca
aktarılır. Kod ve model paketinin sürüm kaydı korunmalıdır.

Dağıtım ZIP'i çekirdek kaynakları, iki özgün Excel girdisini ve doğrulanmış model
paketini içerir. Büyük arşiv raporları, Dataset Builder demo varlıkları ve `.venv`
dahil değildir; bunlar benchmark için gerekli değildir. Dosya listesi ve her dosyanın
SHA-256 değeri ZIP içindeki `PACKAGE_MANIFEST.json` dosyasında bulunur.

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell için: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-edge.txt
# İş istasyonuna uygun PyTorch CPU/CUDA kurulumu ayrıca yapılır.
python scripts/run_data_pipeline.py
python scripts/prepare_research_bundle.py --ft-device cpu --epochs 50 --output artifacts/research_bundle_final --label research_candidate
```

İş istasyonunda çalışan uygun CUDA ortamı varsa `--ft-device cuda` kullanılabilir.
Eğitimde XGBoost CPU, FT istenen cihazda çalışır. Eğitim Jetson benchmarkının parçası
değildir. `--threads 1` varsayılandır. Çıktı klasörü doluysa betik üzerine yazmaz.

Üretilen paket:

- `train.csv`, `valid.csv`, `test.csv`: aynı, dondurulmuş deney ayrımları ve satır kimlikleri.
- `quarantine.csv`, `data_audit.json`: aynı koşula çelişkili hedefler ve veri işlem kaydı.
- `xgboost/`, `ft_transformer/`: modeller, ön işleyiciler, konfigürasyon ve eğitim metrikleri.
- Her modelin `reports/` klasörü: yalnız test kümesinden CSV raporları ve PNG hata grafikleri; arayüzün genel bakış/karşılaştırma panellerini besler.
- `requirements-portable.txt`: eğitimdeki ortak Python paketlerinin tam sürümleri; PyTorch ayrı.
- `manifest.json`: bütün paket dosyalarının SHA-256 değerleri, kaynak dosya özeti, kod özeti ve deney etiketi.

Mevcut dağıtımın `artifacts/research_bundle_portable/` klasörü kısa işlev eğitimiyle
hazırlanmıştır. İlk kurulum/çalıştırma kontrolünde kullanılabilir. Yeni nihai paketi
oluşturduktan sonra her cihazın **aynı paketi** kullandığından emin olun.

## Jetson'da repodan çalıştır

Hazırlık kaynaklarını açın, model paketini aynı klasör yapısında aktarın. İş istasyonunun
`.venv` klasörünü kopyalamayın. Jetson üzerinde yeni bir sanal ortam kurun.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-edge.txt
```

JetPack ile uyumlu NVIDIA PyTorch wheel'ini bu ortama ayrıca kurun. Rastgele bir
`pip install torch` komutunun Jetson CUDA desteği sağladığı varsayılmamalıdır.
Kurulum eşleştirmesi [NVIDIA PyTorch for Jetson kılavuzundan](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
cihazın sürümüne göre yapılır. `requirements-edge.txt` PyTorch'u kurmaz/değiştirmez.
Ön işleyici sürümleri paketle aynı olmalıdır; betik sürüm farkında açık hata verir.

```bash
python scripts/inspect_edge_environment.py > jetson_environment.json
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cpu --sample-size 200 --warmup 50 --repetitions 500 --trials 3 --require-jetson --output artifacts/benchmarks_cpu
python scripts/run_edge_benchmark.py --bundle artifacts/research_bundle_portable --ft-device cuda --sample-size 200 --warmup 50 --repetitions 500 --trials 3 --require-jetson --output artifacts/benchmarks_gpu
```

Nihai paket için `--bundle artifacts/research_bundle_final` yazın. İlk komut bütün
yöntemleri CPU'da karşılaştırır. İkinci komutta yalnız FT GPU'ya taşınır; aynı test
örneklemi/seed kullanılır. CUDA yoksa FT için başarısız kayıt ve sıfırdan farklı
çıkış kodu oluşur; CPU'ya sessiz düşüş yoktur. Bir yöntem başarısız olduğunda genel
rapor `complete=false` olur, diğer ölçümler korunur. `--require-jetson`, cihaz kimliği
doğrulanmadan raporu Jetson deneyi olarak üretmez.

JetPack paket metadata'sı her kurulumda bulunmayabilir; L4T/model ve CUDA kaydı ayrıca
tutulur. Container içinde cihaz ağacı veya L4T görünmüyorsa `--require-jetson` engeli
giderilmeden container çıktısı doğrulanmış Jetson ölçümü sayılmaz.

## Ölçüm protokolü ve çıktılar

Örnekler yalnız test kümesinden bir kez seçilir. Her motor kategorisi en az bir örnekle
temsil edilir; tüm yöntemler aynı `test_sample.csv` dosyasını okur. `--sample-size`
test kümesinin tamamı değildir: doğruluk metrikleri belirtilen ortak **test örneklemi**
içindir. Tam test doğruluğu istenirse `--sample-size 8848` kullanılabilir; interpolasyon
doğruluk geçişinin süresi artacaktır. Bu sayı mevcut veri paketine aittir.

Her yöntem/tekrar yeni süreçte çalışır. Ölçüm sırası seed ile tekrarlanabilir biçimde
değişir. Model yükleme ve ilk hazırlık ayrı tutulur; ısınmadan sonra tek satır tahmin
gecikmesi ölçülür. Gecikme ön işlemeyi, PyTorch tensör oluşturmayı ve CPU/GPU aktarımını
içerir; HTTP, tarayıcı, diskten model yükleme veya eğitim süresini içermez. GPU ölçümü
senkronize edilir; [PyTorch CUDA zamanlama belgesi](https://docs.pytorch.org/docs/stable/notes/cuda)
bu gerekliliği açıklar. Bu bir kernel-only/TensorRT testi değildir.

CPU ortalaması yalnız ölçüm aralığındaki süreç CPU sürelerinden alınır; ısınma ve
sonraki doğruluk geçişi bu paya girmez. Tek mantıksal çekirdek %100'dür. RSS 10 ms
aralıklarla örneklenir; kesin tahsis tepe değeri veya yalnız modele ait RAM değildir.
Kütüphaneler ve yüklenmiş süreç ortamı dahildir. CUDA allocator peak ayrı kaydedilir;
Jetson'ın paylaşımlı RAM'inde bu sayılar toplanarak toplam bellek diye sunulmaz.
Artefakt boyutu ön işleyiciyi de kapsar; spline için saklanan eğitim referans CSV'sidir.

Çıktı klasöründe tarih/kimlikli `edge_*` alt klasörü oluşur:

- `summary.json`, `summary.csv`: her tekrarın p50/p95/p99, RSS, CPU, dosya boyutu ve RMSE/MAE/MAPE/R² sonuçları.
- `*.latencies.csv`: her çağrının süresi ve satır kimliği.
- `*.predictions.csv`: ortak örneklerin gerçek/tahmin değerleri.
- `*.json`, `*.log`, `*.worker.json`: süreç metadata'sı ve hata kayıtları.
- `tegrastats.log`: araç mevcutsa UTC zaman damgalı ham Jetson telemetrisi.
- Ana çıktı klasöründe `edge_benchmark_latest.json`: son koşunun özeti; geçmiş alt klasörler korunur.

[`tegrastats`](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html)
RAM, işlemci ve cihazın desteklediği sıcaklık/güç alanlarını sağlayabilir. Araç yoksa
bu metrikler ölçülmüş sayılmaz. Ham güç kayıtları saklanır; bu sürüm otomatik enerji
veya throttle yüzdesi hesabı iddia etmez. Güç/saat/soğutma koşulları iki deneyde aynı
tutulmalı ve kaydedilmelidir; betik kendiliğinden maksimum güç moduna geçmez.

## Arayüz

```bash
export SPECIFIC_RANGE_BUNDLE="$PWD/artifacts/research_bundle_portable"
export SPECIFIC_RANGE_EDGE_RESULTS="$PWD/artifacts/benchmarks_cpu/edge_benchmark_latest.json"
export SPECIFIC_RANGE_NO_BROWSER=1
python scripts/web_app/server.py
```

Tarayıcıda `http://localhost:5000` açılır. `Tekil Tahmin` üç yöntemi aynı koşulda
gösterir; `Maliyet` yeni ölçüm tablosunu gösterir. Başka makinenin referans raporu
yerel ölçüm yerine kullanılmaz. Kısa eğitim etiketi ve Jetson doğrulama durumu ekranda
yer alır. Bu profil frozen bundle kullanır; eski Setup/Pi çalıştırma uçları kapalıdır.
Legacy full-table raporlar eğitim ve test satırlarını birlikte içerebildiğinden
genelleme başarımı kanıtı olarak kullanılmamalıdır.

## Docker seçeneği

`deploy/jetson/Dockerfile` taban imajını zorunlu build argümanı olarak alır. NVIDIA'nın
cihaz/JetPack ile uyumlu, PyTorch içeren ARM64 imajını sürüm/digest ile seçmek gerekir.
Bu seçimi cihaz bilgisi olmadan yapmadık. İmajın Python sürümü en az 3.10 olmalıdır.

```bash
docker build --build-arg BASE_IMAGE=<uyumlu_imaj@sha256:digest> -f deploy/jetson/Dockerfile -t specific-range-edge .
docker run --rm --runtime nvidia -v "$PWD/artifacts/research_bundle_portable:/bundle:ro" -v "$PWD/artifacts/benchmarks_docker:/results" specific-range-edge --bundle /bundle --output /results --ft-device cuda --sample-size 200 --warmup 50 --repetitions 500 --trials 3
```

Köşeli yer tutucu gerçek imajla değiştirilmeden komut çalıştırılamaz. Tarifte model/veri
paketi imaja gömülmez; read-only mount edilir. Container telemetri erişimi native
ortamla aynı varsayılmaz; gerekli ise tegrastats host üzerinde ayrıca kaydedilir.
Native ve Docker ölçümleri ayrı koşu olarak raporlanmalı, aynı tablonun tek koşusu
gibi birleştirilmemelidir. Docker tarifi bu bilgisayarda build edilmedi.

Güncel bir model paketiyle yeniden ZIP üretmek için:

```bash
python scripts/export_jetson_package.py --bundle artifacts/research_bundle_final --output specific_range_jetson_final.zip
```

## Sonraki gerçek deney

Bu tur hazırlık ve yerel çalıştırma doğrulamasını tamamlar. Hedef cihaz belirlenince
PyTorch/JetPack eşleşmesi yapılmalı, CPU ve FT-GPU koşuları gerçek Jetson üzerinde
alınmalı ve veri kökeni/sayısallaştırma hatası açıklanmalıdır. Ölçülen gecikmeler
tek başına gerçek zaman garantisi, sertifikasyon veya genel yöntem üstünlüğü değildir.
