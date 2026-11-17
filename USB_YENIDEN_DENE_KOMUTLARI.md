# USB'yi yeniden dene — GitHub girişi zaten yapılmışsa

**Önce önceki terminal komutlarının bitmesini bekle. Sonra aşağıdaki kutunun
tamamını kopyalayıp Jetson terminaline yapıştır.** Her komut ayrı satırdadır.

Gelen ilk raporda büyük exFAT bölüm yaklaşık 61,5 GB, küçük ikinci bölüm 32 KB.
Bağlama hatası `unknown filesystem type 'exfat'`; yönetici yetkisi alınamadığı
için kurulu FUSE sürücüsüyle ikinci deneme atlanmış. Yönetici şifresi alma
düzeltmesi bu dala eklendi.

Bu blok önce güncellemeyi alır, Jetson giriş şifreni ister, USB'yi tekrar salt
okunur açmayı dener ve yeni raporu aynı dala gönderir. GitHub'a yeniden giriş
yapman gerekmez. Yeni benchmark başlatmaz.

```bash
bash <<'USB_YENIDEN_DENE'
cd "$HOME/specific-range-jetson" || exit 1

git pull --ff-only || exit 1

sudo -v </dev/tty || exit 1

bash USB_KONTROL.sh </dev/tty

bash SONUCLARI_GONDER.sh </dev/tty
USB_YENIDEN_DENE
```

`USB ERISIM YOLU` yazarsa o yolu dosya yöneticisinde açabilirsin.
`GITHUB SONUÇLARI GÖNDERİLDİ` yazınca yeni rapor incelenebilir.

İlk raporda cihaz **Jetson Orin Nano Super**, Ubuntu **22.04.5**, Python **3.10.12**,
L4T **36.4.7** olarak görüldü. Benchmark sonucunun yüklenmesi ayrı adımdır;
[başlatma kılavuzu](docs/JETSON_TEK_KOMUT.md).
