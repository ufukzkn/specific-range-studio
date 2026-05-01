# FT-Transformer Yöntemi

FT-Transformer bu projenin ana araştırma modelidir. Klasik tabular regresyon yaklaşımından farklı olarak her kolonu ortak boyutta bir token temsiline dönüştürür ve bu tokenları Transformer encoder blokları içinde işler.

## Neden Kullanılıyor?

Uçak performans verisi yalnızca tek değişkenli bir eğri problemi değildir. `altitude`, `mach`, `gross_weight`, `drag_index`, `fuel_flow` ve `engine_type` birlikte değiştiğinde specific range yüzeyi doğrusal olmayan ve rejime bağlı bir yapı gösterir. FT-Transformer'ın temel avantajı, bu kolonlar arası etkileşimleri self-attention ile örnek bazında öğrenebilmesidir.

## Girdi ve Hedef

Kullanılan temel kolonlar:

- `altitude`
- `gross_weight`
- `drag_index`
- `mach`
- `fuel_flow`
- `engine_type`

Tahmin hedefi `specific_range` değeridir. Sayısal kolonlar tokenizer tarafından vektör temsiline projekte edilir; kategorik kolonlar embedding mantığıyla aynı temsil boyutuna taşınır.

## Mimari Özeti

Akış şu şekildedir:

```text
Tablo satırı -> Feature tokenizer -> Transformer encoder -> Regression head -> specific_range
```

Model tarafında takip edilen ana hiperparametreler:

- `d_model`
- `n_layers`
- `n_heads`
- `d_ff`
- `dropout`
- `learning_rate`
- `batch_size`

## Eğitim ve Artefact Yapısı

Tipik eğitim komutu:

```powershell
python scripts/train_ft_transformer.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
```

Önemli çıktılar:

- `artifacts/ft_transformer/model.pt`
- `artifacts/ft_transformer/metrics.json`
- `artifacts/ft_transformer/reports/ft_transformer_overall_summary.csv`
- `artifacts/ft_transformer/reports/ft_transformer_row_level_predictions.csv`
- `artifacts/ft_transformer/reports/plots/`

## Maliyet Simülatöründeki Kullanımı

Maliyet paneli FT-Transformer için rapor metriklerini ve `model.pt` dosya boyutunu okur. Tahmini latency ve peak RAM hesabında `d_model`, `n_layers`, `n_heads`, `d_ff`, `batch_size` ve model boyutu kullanılır.

Bu tahmin, gerçek GPU/CPU benchmark'ı yerine geçmez. Panelin amacı XGBoost ile FT-Transformer arasında doğruluk, gecikme ve bellek açısından hızlı bir karar desteği sunmaktır.
