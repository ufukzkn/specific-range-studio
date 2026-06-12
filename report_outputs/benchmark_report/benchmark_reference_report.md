# Benchmark ve Maliyet Karşılaştırma Raporu
**Run ID:** `20260512_202308`  
**Ölçüm zamanı:** `2026-05-12 17:23 UTC`  
**Ortam:** `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43`  
**Python:** `3.12.13`  
**Örnek / tekrar:** `200 sample`, `20 warmup`, `200 repetition`

## Yönetici Özeti
Bu ölçüm, Flask arayüzünde kullanılan gerçek artifact-backed prediction yolunu baz alır. Interpolasyon artık doğruluk yarışına sokulmaz; çünkü uygulamada tahmini gerçek/reference ailesi olarak kabul edilir. Buna rağmen runtime, RAM, CPU ve boyut maliyeti için aynı panelde gösterilir. XGBoost ve FT-Transformer ise hem doğruluk hem runtime açısından karşılaştırılır.

Kısa sonuç: bu WSL doğrulama koşusunda XGBoost en düşük p95 gecikmeye, en iyi RMSE/MAE/MAPE değerlerine ve en düşük birleşik ML maliyetine sahiptir. FT-Transformer model dosyası daha küçük olsa da bu ölçümde bellek kullanımı daha yüksek ve doğruluk metrikleri XGBoost'un gerisindedir. Interpolasyon cache optimizasyonundan sonra makul seviyeye inmiştir; p95 yaklaşık 6.78 ms ölçülmüştür.

## Sayısal Özet

| Yöntem | p50 ms | p95 ms | Peak RSS MB | CPU avg % | Model MB | RMSE | MAE | MAPE % | R2 | Fit Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interpolasyon | 5.693 | 6.778 | 166.5 | 355.0 | 2.642 | - | - | - | - | 74.0 |
| XGBoost | 1.780 | 2.296 | 318.0 | 117.7 | 1.919 | 0.003468 | 0.001372 | 1.560 | 0.998477 | 81.5 |
| FT-Transformer | 2.225 | 3.049 | 482.1 | 116.1 | 0.417 | 0.003671 | 0.002044 | 2.888 | 0.998293 | 77.6 |

## Grafikler

### Tekil tahmin gecikmesi

![Tekil tahmin gecikmesi](D:/ESKIDESKTOP/eski pcnin c'sinin indirilenleri/liftup/sonrasi/docs/benchmark_reference/figures/latency_p50_p95.png)

### RAM ve CPU kullanımı

![RAM ve CPU kullanımı](D:/ESKIDESKTOP/eski pcnin c'sinin indirilenleri/liftup/sonrasi/docs/benchmark_reference/figures/resource_usage.png)

### XGBoost ve FT-Transformer doğruluk metrikleri

![XGBoost ve FT-Transformer doğruluk metrikleri](D:/ESKIDESKTOP/eski pcnin c'sinin indirilenleri/liftup/sonrasi/docs/benchmark_reference/figures/accuracy_metrics.png)

### Maliyet bileşenleri

![Maliyet bileşenleri](D:/ESKIDESKTOP/eski pcnin c'sinin indirilenleri/liftup/sonrasi/docs/benchmark_reference/figures/cost_components.png)

### Fit Score

![Fit Score](D:/ESKIDESKTOP/eski pcnin c'sinin indirilenleri/liftup/sonrasi/docs/benchmark_reference/figures/fit_score.png)

## Maliyet Formülü Notu

Varsayılan ağırlıklar Accuracy %45, Latency %25, Memory %20 ve CPU %10 olarak kullanılır. Accuracy cost yalnız XGBoost ve FT-Transformer için hesaplanır. Interpolasyon deterministik referans ailesi olduğu için accuracy kısmı 'referans dışı' kabul edilir ve sadece runtime-only cost ile gösterilir. Fit Score, `100 / (1 + 0.30 * CompositeCost)` formülünden gelir; bu skor bir modeli otomatik 100 yapmaz, mutlak maliyet düştükçe yükselir.

## Yorum

XGBoost bu veri setinde hâlâ çok güçlü bir baseline davranışı gösteriyor: düşük gecikme, düşük hata ve makul RAM kullanımı aynı anda geliyor. FT-Transformer teorik olarak daha zengin feature etkileşimleri öğrenebilecek mimariye sahip olsa da bu eğitilmiş artifact ile XGBoost'u geçmiyor. Bu sonuç FT-Transformer'ın yanlış olduğu anlamına gelmez; daha iyi hiperparametre araması, eğitim süresi, veri artırımı veya quantization/ONNX export sonrası farklılaşabilir. Interpolasyon ise gerçek tablo/reference mantığını temsil ettiği için doğruluk yarışında değil, runtime ve kaynak maliyeti açısından izlenmelidir.
