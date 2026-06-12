# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
preprocessing.py
================
Tahap Preprocessing & Feature Engineering untuk riset prediksi beban kerja Azure LLM.
Menyiapkan data agar siap untuk model CatBoost, LightGBM, LSTM, dan GRU.

Output (disimpan di folder analysis/preprocessed/):
  - {dataset}_resampled.csv        : Data setelah resampling per menit
  - {dataset}_features.csv         : Data lengkap dengan Lag Features (sebelum scaling)
  - {dataset}_train_scaled.npy     : Array fitur training (sudah MinMax Scaled)
  - {dataset}_test_scaled.npy      : Array fitur testing (sudah MinMax Scaled)
  - {dataset}_train_target.npy     : Array target training
  - {dataset}_test_target.npy      : Array target testing
  - {dataset}_scaler.pkl           : Objek MinMaxScaler (wajib disimpan untuk inverse_transform)
  - {dataset}_split_info.txt       : Ringkasan info pembagian data
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis', 'preprocessed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    'conv':      'AzureLLMInferenceTrace_conv.csv',
    'code':      'AzureLLMInferenceTrace_code.csv',
    'multimodal':'AzureLMMInferenceTrace_multimodal.csv.gz',  # format .gz didukung Pandas
}

# Lag features yang akan dibuat (dalam satuan menit)
LAG_STEPS = [1, 2, 3]
# Ukuran rolling window untuk fitur tambahan
ROLLING_WINDOW = 3
# Persentase data training (kronologis)
TRAIN_RATIO = 0.80


# ============================================================
# FUNGSI UTAMA
# ============================================================

def load_raw(csv_path: str) -> pd.DataFrame:
    """Membaca CSV dan menghitung TotalTokens."""
    df = pd.read_csv(csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['TotalTokens'] = df['ContextTokens'] + df['GeneratedTokens']
    return df


def resample_per_minute(df: pd.DataFrame) -> pd.Series:
    """
    Mengagregasi Total Token ke interval 1 menit.
    Menjamin tidak ada gap menit kosong dengan fill 0.
    """
    ts = df.set_index('TIMESTAMP')['TotalTokens']
    ts_resampled = ts.resample('1min').sum()         # sum total token per menit
    ts_resampled = ts_resampled.fillna(0)           # isi menit tanpa request dengan 0
    return ts_resampled


def create_lag_features(ts: pd.Series, lag_steps: list, rolling_window: int) -> pd.DataFrame:
    """
    Membuat DataFrame fitur berbasis Lag dan Rolling Mean.
    
    Kolom yang dihasilkan:
      TotalTokens         : nilai asli (target t)
      Lag_1 ... Lag_N     : nilai di t-1, t-2, ..., t-N
      Rolling_Mean_W      : rata-rata W menit terakhir
    """
    df_feat = pd.DataFrame({'TotalTokens': ts})

    # Lag features
    for lag in lag_steps:
        df_feat[f'Lag_{lag}'] = df_feat['TotalTokens'].shift(lag)

    # Rolling mean sebagai fitur tren pendek
    df_feat[f'Rolling_Mean_{rolling_window}'] = (
        df_feat['TotalTokens'].shift(1).rolling(window=rolling_window).mean()
    )

    # Buang baris awal yang mengandung NaN akibat shifting
    df_feat.dropna(inplace=True)

    return df_feat


def chronological_split(df_feat: pd.DataFrame, train_ratio: float):
    """
    Membagi data secara kronologis (80% train, 20% test).
    TIDAK diacak (shuffle=False) karena data time series.
    """
    split_idx = int(len(df_feat) * train_ratio)
    train_df = df_feat.iloc[:split_idx]
    test_df  = df_feat.iloc[split_idx:]
    return train_df, test_df, split_idx


def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Melakukan MinMax Scaling [0, 1] pada seluruh kolom fitur & target.
    Scaler di-fit HANYA pada data train agar tidak terjadi data leakage.
    
    Returns:
        train_scaled, test_scaled : array numpy setelah scaling
        scaler                    : objek MinMaxScaler (untuk inverse_transform)
    """
    feature_cols = train_df.columns.tolist()  # termasuk TotalTokens sebagai target

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[feature_cols])   # fit + transform di train
    test_scaled  = scaler.transform(test_df[feature_cols])         # transform saja di test

    return train_scaled, test_scaled, scaler, feature_cols


def save_outputs(name: str, ts_resampled: pd.Series, df_feat: pd.DataFrame,
                 train_df, test_df, train_scaled, test_scaled, scaler,
                 feature_cols: list, split_idx: int):
    """Menyimpan semua artefak preprocessing ke folder output."""

    # 1. Data resampled
    ts_resampled.to_csv(os.path.join(OUTPUT_DIR, f'{name}_resampled.csv'), header=True)

    # 2. DataFrame fitur lengkap (sebelum scaling)
    df_feat.to_csv(os.path.join(OUTPUT_DIR, f'{name}_features.csv'))

    # 3. Array numpy training dan testing (setelah scaling)
    # Kolom pertama adalah TotalTokens (target), sisanya adalah fitur
    np.save(os.path.join(OUTPUT_DIR, f'{name}_train_scaled.npy'), train_scaled)
    np.save(os.path.join(OUTPUT_DIR, f'{name}_test_scaled.npy'),  test_scaled)

    # Target saja (kolom 0 dalam scaled array)
    np.save(os.path.join(OUTPUT_DIR, f'{name}_train_target.npy'), train_scaled[:, 0])
    np.save(os.path.join(OUTPUT_DIR, f'{name}_test_target.npy'),  test_scaled[:, 0])

    # 4. Scaler object (dibutuhkan nanti untuk inverse_transform hasil prediksi)
    with open(os.path.join(OUTPUT_DIR, f'{name}_scaler.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_cols': feature_cols}, f)

    # 5. Ringkasan info
    info_lines = [
        f"Dataset          : {name}",
        f"Total menit      : {len(df_feat)}",
        f"Kolom fitur      : {feature_cols}",
        f"Train rows       : {len(train_df)} ({len(train_df)/len(df_feat)*100:.1f}%)",
        f"Test rows        : {len(test_df)} ({len(test_df)/len(df_feat)*100:.1f}%)",
        f"Split index      : {split_idx}",
        f"Train time range : {train_df.index[0]} to {train_df.index[-1]}",
        f"Test time range  : {test_df.index[0]} to {test_df.index[-1]}",
        f"Scaler range     : {scaler.data_min_[0]:.2f} - {scaler.data_max_[0]:.2f} (TotalTokens)",
    ]
    with open(os.path.join(OUTPUT_DIR, f'{name}_split_info.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(info_lines))

    for line in info_lines:
        print(f"  {line}")


# ============================================================
# PIPELINE UTAMA
# ============================================================

def run_preprocessing():
    print("=" * 60)
    print("TAHAP PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    for name, filename in DATASETS.items():
        csv_path = os.path.join(DATA_DIR, filename)
        print(f"\n{'- '*30}")
        print(f"  Memproses dataset  : {name.upper()} ({filename})")
        print(f"{'- '*30}")

        # LANGKAH 1: Load & hitung TotalTokens
        df_raw = load_raw(csv_path)
        print(f"  [1/4] Raw data loaded   : {len(df_raw):,} baris")

        # LANGKAH 2: Resample per menit
        ts_resampled = resample_per_minute(df_raw)
        print(f"  [2/4] Resampled (1 min) : {len(ts_resampled)} menit "
              f"({ts_resampled.index[0].strftime('%H:%M')} – {ts_resampled.index[-1].strftime('%H:%M')})")

        # LANGKAH 3: Buat Lag Features
        df_feat = create_lag_features(ts_resampled, LAG_STEPS, ROLLING_WINDOW)
        lag_names = [f'Lag_{l}' for l in LAG_STEPS] + [f'Rolling_Mean_{ROLLING_WINDOW}']
        print(f"  [3/4] Lag features      : {lag_names}")
        print(f"         DataFrame shape  : {df_feat.shape}  "
              f"(baris setelah buang NaN awal)")

        # LANGKAH 4: Chronological Split 80/20
        train_df, test_df, split_idx = chronological_split(df_feat, TRAIN_RATIO)

        # LANGKAH 5: MinMax Scaling
        train_scaled, test_scaled, scaler, feature_cols = scale_data(train_df, test_df)
        print(f"  [4/4] MinMax Scaling    : done (fit hanya pada train set)")

        # LANGKAH 6: Simpan semua output
        print(f"\n  Ringkasan Split:")
        save_outputs(name, ts_resampled, df_feat, train_df, test_df,
                     train_scaled, test_scaled, scaler, feature_cols, split_idx)

    print(f"\n{'='*60}")
    print(f"  Semua artefak disimpan di folder:")
    print(f"  {OUTPUT_DIR}")
    print('='*60)
    print("\n  File yang dihasilkan per dataset:")
    print("  - *_resampled.csv     : data per menit (mentah)")
    print("  - *_features.csv      : data + lag features (belum di-scale)")
    print("  - *_train_scaled.npy  : fitur train (scaled)")
    print("  - *_test_scaled.npy   : fitur test (scaled)")
    print("  - *_train_target.npy  : target train (scaled)")
    print("  - *_test_target.npy   : target test (scaled)")
    print("  - *_scaler.pkl        : scaler object (untuk inverse_transform)")


if __name__ == '__main__':
    run_preprocessing()
