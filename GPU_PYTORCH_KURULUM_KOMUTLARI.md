# Yalnız GPU PyTorch kurulumunu düzelt

Bu adım **benchmark çalıştırmaz**. Orin Nano Super / L4T 36.4.x / Python 3.10 /
CUDA 12.6 için ayrı GPU ortamını kurar ve gerçek GPU hesabıyla kontrol eder.
Mevcut CPU ortamı ve ölçüm sonuçları korunur.
GPU paketi NVIDIA `2.5.0a0+872d972e41.nv24.08` build'idir; CPU ortamındaki
2.5.1 paketi ayrı kalır. Sonraki ölçümde bu farklı build/sürümler rapora kaydedilir.

**Aşağıdaki bloğun tamamını Jetson terminaline yapıştır.** Gerekirse Jetson
giriş şifreni yaz. GitHub girişini daha önce yaptıysan tekrar giriş gerekmez.

```bash
bash <<'JETSON_YALNIZ_GPU_PYTORCH'
cd "$HOME/specific-range-jetson" || exit 1

git pull --ff-only || exit 1

sudo -v </dev/tty || exit 1

bash GPU_PYTORCH_KUR.sh </dev/tty
JETSON_YALNIZ_GPU_PYTORCH
```

Yaklaşık **830 MB** NVIDIA paketi indirilir; Python bağımlılıkları ayrıca indirilir.
Kurulum HOME altındaki `specific-range-jetson-auto/pytorch_gpu_310_cuda126/`
klasörüne yapılır. GPU ortamı için yalnız eksik çalışma kütüphanesi gerekirse
sistem paketinden kurulur. JetPack, GPU sürücüsü veya güç modu yükseltilmez.

Başarılı olursa **`GPU PYTORCH HAZIR`** yazacak. Bu mesaj yalnız gerçek CUDA
matris hesabı ve küçük Transformer işlemi başarıyla geçerse çıkar. Sonraki
benchmarkın bu ortamı bulabilmesi için `gpu_runtime.json` kaydedilir.
Bu aşamada yeni CPU/GPU benchmark sonucu üretilmez.

Kurulum logu, paket sürümleri, indirilen dosya hashleri ve GPU kontrol sonucu
`run_gpu_setup_.../output/` altında saklanır ve aynı GitHub dalına gönderilmeye
çalışılır. Hata olursa kaydı yine gönderilir; yerel kopya korunur. Tam benchmark
tekrarını bu GPU kontrolünü doğruladıktan sonra yapacağız.

Paket kaynağı NVIDIA'nın kendi [JetPack wheel dizini](https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/).
Wheel'in içindeki CUDA sürümü 12.6 ve etiketi cp310/linux_aarch64 olarak doğrulandı.
cuSPARSELt 0.5.2.1 SHA-256 değeri NVIDIA'nın
[dağıtım manifestinden](https://developer.download.nvidia.com/compute/cusparselt/redist/redistrib_0.5.2.json)
alınır. Kurulum yöntemi [NVIDIA PyTorch rehberi](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
ve [PyTorch JetPack rehberi](https://docs.pytorch.org/TensorRT/v2.6.1/getting_started/jetpack.html)
esas alınarak hazırlandı. Gerçek cihazdaki uyumluluk son GPU işlemiyle doğrulanacak;
yalnız CUDA sürümünün eşleşmesi başarı garantisi değildir.
