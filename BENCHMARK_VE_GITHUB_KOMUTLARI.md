# Tüm benchmarkı çalıştır ve sonuçlarını GitHub'a gönder

GitHub girişini zaten yaptığın Jetson içindir. Önce önceki terminal işleminin
bitmesini bekle. **Aşağıdaki kutunun tamamını kopyalayıp terminale yapıştır ve
Enter'a bas.** Her komut ayrı satırdadır. Gerekirse Jetson giriş şifreni yaz.

```bash
bash <<'JETSON_BENCHMARK_VE_GITHUB'
cd "$HOME/specific-range-jetson" || exit 1

git pull --ff-only || exit 1

sudo -v </dev/tty || exit 1

bash BASLAT.sh </dev/tty
benchmark_kodu=$?

bash SONUCLARI_GONDER.sh </dev/tty
gonderim_kodu=$?

printf '\nBenchmark cikis kodu: %s\nGitHub gonderim cikis kodu: %s\n' "$benchmark_kodu" "$gonderim_kodu"

if [ "$benchmark_kodu" -ne 0 ]; then
    printf 'Benchmark tamamlanamadi. Hata raporlari da gonderilmeye calisildi.\n'
    exit "$benchmark_kodu"
fi

exit "$gonderim_kodu"
JETSON_BENCHMARK_VE_GITHUB
```

Bu blok cihaz bilgilerini toplar, hazır model paketini doğrular, gerekli ortamı
kurar ve üç yöntemin (kübik interpolasyon, XGBoost, FT-Transformer uyarlaması)
kısa CPU denemesini ve tam CPU benchmarkını çalıştırır. Tam koşu her yöntem için
3 ayrı deneme, 50 ısınma çağrısı, 1000 zamanlanan çağrı ve 200 test örneği kullanır.
Mevcut NVIDIA PyTorch/CUDA gerçek GPU işleminden geçerse ayrı GPU koşusunu da dener.
GPU hazır değilse atlandığı raporda açıkça belirtilir.

Çalışma bitince yeni sonuçları `jetson-benchmark` dalında `jetson-results/run_.../`
altına commit edip göndermeyi dener. Son gönderme adımı önceki yerel raporları da
kapsar ve ilk gönderim başarısız olduysa yeniden dener. Kurulum/benchmark hatası
oluşursa hata raporları da gönderilmeye çalışılır; yerel çıktılar korunur.

**Benchmark çıkış kodu 0** ve **GitHub gönderim çıkış kodu 0** olmalı.
`GITHUB SONUÇLARI GÖNDERİLDİ` mesajını ve CPU/GPU durumunu sonunda göreceksin.
Gönderim başarılı olması tek başına benchmarkın tamamlandığı anlamına gelmez;
hata raporu da gönderilmiş olabilir. Sonuçtaki STATUS.json bu ayrımı içerir.

USB bağlantısı gerekmez; bu blok USB teşhisi veya bağlama komutu çalıştırmaz.
Hazır modeller `functional_check` etiketlidir; FT modeli 5 epoch eğitilmiştir.
Bu koşu cihazdaki kurulum ve ölçüm yolunu doğrular; nihai bilimsel model eğitimi
ve model seçimi ayrı adımdır.
