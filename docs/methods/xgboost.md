# XGBoost Yöntemi

XGBoost bu projede ana araştırma modeli değil, güçlü ve pratik bir baseline olarak konumlanır. Amaç, FT-Transformer'ın performansını yalnızca sezgisel olarak değil, tabular veride hâlâ çok güçlü olan ağaç tabanlı bir modele karşı adil şekilde değerlendirmektir.

## Girdi ve Hedef

Model, temizlenmiş birleşik veri setindeki şu girdileri kullanır:

- `altitude`
- `gross_weight`
- `drag_index`
- `mach`
- `fuel_flow`
- `engine_type`

Tahmin hedefi `specific_range` değeridir. `engine_type` kategorik bilgi olarak kodlanır; sayısal kolonlar eğitim pipeline'ında ortak şemaya çekilir.

## Uygulamadaki Rolü

- Tekil tahmin ekranında FT-Transformer ile aynı uçuş koşulu için tahmin üretir.
- Karşılaştırma ekranında interpolasyon referansına göre hata değerleri hesaplanır.
- Toplu raporda satır bazlı tahmin, mutlak hata, yüzde hata ve slice özetleri üretir.
- Maliyet simülatöründe FT-Transformer ile birlikte değerlendirilir.

## Eğitim ve Artefact Yapısı

Tipik eğitim komutu:

```powershell
python scripts/train_xgboost.py --dataset data/processed/combined_specific_range.csv --device cuda --run-table-report
```

Önemli çıktılar:

- `artifacts/xgboost/model.json`
- `artifacts/xgboost/metrics.json`
- `artifacts/xgboost/reports/xgboost_overall_summary.csv`
- `artifacts/xgboost/reports/xgboost_row_level_predictions.csv`
- `artifacts/xgboost/reports/plots/`

## Maliyet Simülatöründeki Kullanımı

Maliyet paneli XGBoost için gerçek eğitim raporundaki `RMSE`, `MAE`, `MAPE` ve `R2` metriklerini okur. Model boyutu `model.json` dosyasının disk boyutundan alınır. Tahmini gecikme ve RAM tarafında ise `n_estimators`, `max_depth` ve model boyutu kullanılır.

Bu değerler gerçek benchmark değildir; karar desteği ve sunum amaçlı yaklaşık profil üretir. Nihai hız ve bellek yorumu için ayrıca gerçek donanım benchmark'ı alınmalıdır.
