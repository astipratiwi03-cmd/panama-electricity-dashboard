# Panama Electricity Demand Dashboard

Dashboard Streamlit yang dibuat dari output notebook (CSV di folder `data/`). Model tidak dilatih ulang saat aplikasi dibuka.

## Tab
- **Overview**: KPI model terbaik (dengan selisih terhadap naive 24 jam), grafik aktual vs prakiraan, peringkat RMSE, leaderboard
- **Models**: perbandingan metrik antar model, stabilitas sebelum/sesudah Maret 2020, error per horizon 1-24 jam
- **Forecasts**: prakiraan 24 jam per origin (tombol prev/next, overlay banyak model) dengan panel error per jam
- **Error analysis**: bias jam x hari, kalender MAPE harian, aktual vs prediksi, distribusi error, hari tersulit
- **Data explorer**: tren, pola jam x hari, musiman bulanan, distribusi, hubungan cuaca-demand, korelasi
- **Weather (Stage 1)**: error prakiraan cuaca per variabel
- **About**: desain prakiraan, daftar model per keluarga, unduhan CSV

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push folder ini ke GitHub, lalu deploy `app.py` di Streamlit Community Cloud.
