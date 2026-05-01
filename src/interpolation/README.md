# Interpolasyon Servisi

`src/interpolation/` klasörü, klasik performans tablosu yaklaşımını proje içi bir servis katmanı olarak sunar.

Bu servis final arayüzde bir makine öğrenmesi modeli gibi değil, deterministik referans aile olarak kullanılır. Özellikle tekil tahmin ve karşılaştırma ekranlarında XGBoost ile FT-Transformer çıktılarının tablo tabanlı tahmini gerçek değere göre yorumlanmasına yardım eder.

## Desteklenen Yöntemler

- `spline`: Cubic Spline, varsayılan yöntem.
- `linear`: Parçalı doğrusal interpolasyon.
- `newton`: Newton divided-difference interpolasyonu.

## Girdi Mantığı

Interpolasyon servisinin temel eksenleri şunlardır:

- `engine_type`
- `altitude`
- `gross_weight`
- `drag_index`
- `mach`

Klasik interpolasyon tarafında `fuel_flow` doğrudan eksen olarak kullanılmaz. `fuel_flow`, XGBoost ve FT-Transformer gibi öğrenilmiş regresyon modellerinin girdisinde kalır.

## Arayüzdeki Rolü

- Tekil tahminde exact match yoksa tahmini gerçek değer olarak interpolasyon sonucu gösterilir.
- XGBoost ve FT-Transformer hataları bu referansa göre hesaplanabilir.
- Maliyet simülatörüne dahil edilmez; çünkü referans aile olarak her durumda ML modelleriyle aynı yarışa sokulması anlamlı değildir.
