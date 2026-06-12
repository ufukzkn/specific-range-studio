# Prompt for a New Codex Session

Use this prompt after cloning the handoff branch and restoring artifacts.

```text
Bu proje başka bilgisayardan devralındı. Önce PROJECT_CONTEXT.md, HANDOFF.md, README.md, docs/methods/xgboost.md, docs/methods/ft_transformer.md, src/interpolation/README.md, scripts/web_app/server.py, scripts/web_app/static/js/app.js ve src/ klasörünü oku.

Proje bağlamı:
- Amaç, muharip bir hava aracının specific range performans veri setini Interpolasyon, XGBoost ve FT-Transformer ile modelleyip kıyaslamak.
- Flask arayüzü ana uygulama.
- Interpolasyon runtime’da src/interpolation içinden gelir; external_apps kullanılmaz.
- Varsayılan interpolasyon Cubic Spline’dır.
- Interpolasyon custom inputta deterministik referans/tahmini gerçek değer olarak kullanılır; accuracy yarışına sokulmaz.
- XGBoost pratik ve güçlü learned-model baseline’dır.
- FT-Transformer akademik tabular-transformer mimari adayıdır.
- PSO hem XGBoost hem FT-Transformer için ağırlıklı çok hedefli maliyetle çalışır: RMSE + latency + model size.
- Maliyet sekmesi gerçek benchmark JSON dosyalarını kullanmalıdır.

Önce repo durumunu kontrol et:
git status --short --branch

Sonra uygulamanın ayağa kalkıp kalkmadığını test et:
python scripts/web_app/server.py

Eğer artifact eksikse HANDOFF.md içindeki restore veya rebuild komutlarını kullan.

Yeni görevlerde ana hedef: mevcut proje mimarisini bozmadan, raporlardaki anlatıyla uyumlu ve standalone çalışabilen bir arayüz/kod tabanı geliştirmek.
```

