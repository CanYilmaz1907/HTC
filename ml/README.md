# ML yön tahmini (Long / Short ihtimali)

Alarm kriterlerine uyan coin için ML modeli, sonraki ~15 dakikada fiyatın yukarı mı aşağı mı gideceğini tahmin eder ve **Long ihtimali %X** olarak bildirimde gösterir.

## Özellikler (features)

- Funding oranı (güncel + önceki + fark)
- 5m / 15m / 1h / 4h getiri (momentum)
- Hacim oranı (5dk hacim / 24h ortalama)
- 24h aralığında fiyat konumu (0–1)
- RSI(14) 5dk, ATR(14)% 5dk
- Son ardışık yeşil 5dk mum sayısı, bir önceki 5dk getirisi
- Saat ve haftanın günü

## Modeli oluşturma

1. **Veri seti oluştur** (Bybit’ten geçmiş funding + kline çeker; birkaç dakika sürebilir):

   ```bash
   python -m ml.dataset
   ```

   Bu komut `ml/dataset.csv` dosyasını üretir.

2. **Eğit ve kaydet**:

   ```bash
   python -m ml.dataset --train
   ```

   veya mevcut CSV ile:

   ```bash
   python -m ml.train ml/dataset.csv
   ```

   Kaydedilen dosyalar: `ml/model.joblib`, `ml/scaler.joblib`, `ml/model_meta.json`.

3. Botu çalıştır. Tarama sonuçlarında model yüklüyse her eşleşme için **Long ihtimali %X** satırı çıkar.

## Model yoksa

`model.joblib` veya `scaler.joblib` yoksa ML atlanır; bildirimde Long ihtimali satırı gösterilmez, diğer tüm özellikler aynen çalışır.
