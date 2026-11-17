# İlk adım komutları — Jetson USB kontrolü ve sonuç paylaşımı

**Aşağıdaki kutunun sağ üstündeki kopyalama düğmesine bas. Tamamını Jetson
terminaline yapıştırıp Enter'a bas.** Her komut ayrı satırdadır; blok sırayla çalışır.
Klasör yoksa repo indirilir, varsa doğru dal kontrol edilerek güncellenir.

Jetson şifreni isterse terminale yaz. GitHub girişinde gösterilen kodu tarayıcıda
kullanıp **ufukzkn hesabınla** giriş yap; tamamlanınca terminale dön. USB takılı kalsın.

```bash
bash <<'JETSON_ILK_ADIM'
set -u

cd "$HOME" || exit 1

if ! command -v git >/dev/null 2>&1; then
    sudo apt-get update </dev/tty || exit 1
    sudo apt-get install -y git </dev/tty || exit 1
fi

if [ ! -e "$HOME/specific-range-jetson" ]; then
    git clone --branch jetson-benchmark --single-branch https://github.com/ufukzkn/specific-range-studio.git specific-range-jetson || exit 1
fi

cd "$HOME/specific-range-jetson" || exit 1

if [ "$(git symbolic-ref --short HEAD)" != "jetson-benchmark" ]; then
    printf 'Bu klasor jetson-benchmark dalinda degil. Bu mesaji paylas.\n'
    exit 1
fi

git pull --ff-only || exit 1

if ! bash GITHUB_GIRIS.sh </dev/tty; then
    printf 'GitHub girisi tamamlanamadi; USB kontrolu yine de yapilacak.\n'
fi

bash USB_KONTROL.sh </dev/tty

bash SONUCLARI_GONDER.sh </dev/tty
JETSON_ILK_ADIM
```

USB kontrolü, uygun bölümü bulursa salt okunur açmayı dener. Açılmazsa kapasite,
bölümler ve bağlama hatası rapora kaydedilir. Son komut önceki benchmark
çıktılarını da aynı `jetson-benchmark` dalına göndermeyi dener.

**`GITHUB SONUÇLARI GÖNDERİLDİ` yazınca haber ver.** Hata olursa son mesajı paylaş.
Çıktılar her durumda Jetson'da `specific-range-jetson-auto` klasöründe tutulur.

Benchmark başlatma ve ayrıntılar: [Jetson kısa kılavuzu](docs/JETSON_TEK_KOMUT.md).
