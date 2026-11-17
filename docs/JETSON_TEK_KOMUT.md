# Jetson: dalı klonla, başlatıcıyı çalıştır

Bu yol `jetson-benchmark` dalındadır. Kaynak kodu ve hazır model ZIP'i birlikte
gelir; USB gerekmez. Jetson'da eğitim yapılmaz. İnternet, Python bağımlılıkları
için gerekir. Kurulum ilk CPU ölçümünü hazırlar; sistemde çalışan NVIDIA
PyTorch/CUDA varsa ayrı GPU ölçümünü de dener.

Terminalde:

```bash
cd "$HOME"
git clone --branch jetson-benchmark --single-branch https://github.com/ufukzkn/specific-range-studio.git specific-range-jetson
cd specific-range-jetson
bash BASLAT.sh
```

Bu klasörü daha önce aynı daldan klonladıysan, dosyalarında yerel değişiklik yoksa:

```bash
cd "$HOME/specific-range-jetson"
git pull --ff-only
bash BASLAT.sh
```

Yerel değişiklik varsa onları silme; yeni bir klasöre klonla. `git` eksikse önce
`sudo apt-get update && sudo apt-get install -y git` gerekir. GitHub'dan bu dalın
ZIP'ini indirip açarak da içindeki `bash BASLAT.sh` komutunu çalıştırabilirsin.

## Başlatıcı ne yapar?

- Cihaz modeli, L4T/JetPack paket bilgisi, Python, RAM, disk, güç/saat ve CUDA
  durumunu otomatik kaydeder. Kimlik önbelleği ile güncel durum ayrı tutulur.
- Kaynakları Jetson HOME altında yeni bir çalışma klasörüne kopyalar; Git commit,
  yerel değişiklik durumu ve her çalışma dosyasının SHA-256 değerini kaydeder.
- Repodaki 3,27 MB model ZIP'ini sabit SHA-256 ile, açılan modelleri kendi
  manifestleriyle doğrular. USB veya mevcut `artifacts` klasörüne ihtiyaç duymaz.
- Ayrı CPU ortamını kurar; üç yönteme kısa deneme ve 3 × 1000 çağrılık benchmark
  uygular. Desteklenen kurulu Python sürümlerinde çalışan CUDA ortamını arar;
  varsa ayrı GPU ortamı ve benchmarkı dener. Hazır CUDA yoksa atlanma nedeni kaydedilir.
- Her çalışmayı `$HOME/specific-range-jetson-auto/run_.../` altında saklar.
  `output/` içinde cihaz bilgileri, kurulum logu, STATUS, paket sürümleri ve varsa
  benchmark ham dosyaları; yanında paylaşılacak `*_sonuclar.tar.gz` arşivi oluşur.
  Hata veya Ctrl+C durumunda da çıktı paketlemesi denenir.
- Başlangıçta bağlı USB'de tek bir `jetsonicinolan` klasörü varsa ve sonunda aynı
  USB hâlâ takılıysa sonuçları onun `output/` klasörüne de kopyalayıp doğrular.
  USB yoksa sonuçlar Jetson'da kalır; terminal arşivin tam yolunu gösterir.
- Kurulum/benchmark yolu tamamlanınca aynı çalışma klasöründe `ARAYUZU_AC.sh`
  oluşturur. Terminalde gösterilen `bash .../ARAYUZU_AC.sh` komutuyla arayüzü aç;
  Jetson tarayıcısında `http://localhost:5000` adresine git.

## Saat hatası çıkarsa

`Release is not valid yet` veya yanlış cihaz saati varsa repo kökünde:

```bash
bash JETSON_SAATI_DUZELT.sh
bash BASLAT.sh
```

Saat yardımcısı iki web sunucusundan uyumlu tarih alır, saati ayarlayıp HTTPS ile
doğrular ve exFAT desteği kurulumunu tekrar dener. USB hiç açılmasa da repo yolu
USB'ye bağımlı değildir. APT tarih veya imza kontrolü kapatılmaz.

## Uyumluluk ve sonuçların anlamı

Gerçek Jetson ARM64 kimliği, kurulu Python **3.10–3.12** ve en az **3 GiB** boş
yer gerekir. Eski JetPack'te yalnız Python 3.6/3.8 bulunursa başlatıcı cihaz
bilgilerini toplar ve uyarlama gerektiğini bildirerek durur. Böyle bir durumda
üretilen sonuç arşivini paylaş; JetPack/sürücü otomatik yükseltilmez.

Hazır modeller **functional_check** etiketlidir; FT modeli 5 epoch eğitilmiştir.
Bu paket kurulum ve benchmark yolunu denemek içindir. Nihai makale doğruluk
karşılaştırması için son model eğitimi/seçimi ayrıca yapılmalıdır. Jetson üzerinde
henüz ölçülmüş bir süre veya her JetPack'te çalışacağı garantisi yoktur.

Ayrıntılı protokol: [JETSON_BENCHMARK.md](JETSON_BENCHMARK.md).
