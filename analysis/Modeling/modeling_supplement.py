# -*- coding: utf-8 -*-
"""
modeling_supplement.py
=======================
Analisis Suplemen untuk memperkuat makalah IEEE:

  1. Naive Baseline     : Prediksi t = nilai nyata di t-1 (Benchmark dasar)
  2. Diebold-Mariano    : Uji statistik untuk membuktikan superioritas LSTM
                          atas LightGBM dan CatBoost bukan karena kebetulan.
  3. Residual Plot      : Visualisasi Selisih (Aktual - Prediksi) pada dataset
                          CODE untuk mengamati perilaku model saat spike trafik.

Semua output (tabel & gambar) disimpan di folder analysis/Modeling/.
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model ML
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Model DL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Input

import random

# ==========================================
# KUNCI SEED GLOBAL (REPRODUCIBILITY LOCK)
# ==========================================
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Opsional: Jika menggunakan TF versi baru untuk memaksa operasi deterministik
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass
# ==========================================

# ─── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP_DIR = os.path.join(BASE_DIR, 'analysis', 'Preprocessing', 'preprocessed')
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))   # analysis/Modeling/
DATASETS = ['conv', 'code', 'multimodal']

# ─── Helpers ─────────────────────────────────────────────────────────────
def inverse_target(preds, X_ref, scaler):
    """Kembalikan prediksi ke satuan token asli via inverse_transform."""
    dummy = np.zeros((len(preds), X_ref.shape[1] + 1))
    dummy[:, 0] = np.array(preds).flatten()
    dummy[:, 1:] = X_ref
    return scaler.inverse_transform(dummy)[:, 0]


def load_data(ds):
    """Muat array numpy dan scaler untuk satu dataset."""
    train = np.load(os.path.join(PREP_DIR, f"{ds}_train_scaled.npy"))
    test  = np.load(os.path.join(PREP_DIR, f"{ds}_test_scaled.npy"))
    with open(os.path.join(PREP_DIR, f"{ds}_scaler.pkl"), 'rb') as f:
        sc = pickle.load(f)['scaler']
    y_train, X_train = train[:, 0], train[:, 1:]
    y_test,  X_test  = test[:, 0],  test[:, 1:]
    y_test_real = inverse_target(y_test, X_test, sc)
    return X_train, y_train, X_test, y_test, y_test_real, sc


def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano Test (Harvey et al., 1997 small-sample correction).
    H0 : Dua model memiliki akurasi prediksi yang SAMA.
    H1 : Ada perbedaan akurasi yang signifikan.

    Parameters
    ----------
    e1, e2 : array of forecast errors (aktual - prediksi)
    h      : forecast horizon (default=1 untuk 1 langkah ke depan)

    Returns
    -------
    dm_stat : float
    p_value : float
    """
    T = len(e1)
    d = e1**2 - e2**2                   # loss differential
    d_bar = np.mean(d)
    # Autocovariance hingga lag h-1
    gamma = [np.cov(d[lag:], d[:T-lag])[0, 1] if lag > 0 else np.var(d, ddof=1)
             for lag in range(h)]
    var_d = (1/T) * (gamma[0] + 2 * sum(gamma[1:]))
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d)
    # Harvey et al. small-sample: skala ke distribusi t
    k = ((T + 1 - 2*h + h*(h-1)/T) / T) ** 0.5
    dm_stat_adj = dm_stat * k
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_adj), df=T-1))
    return dm_stat_adj, p_value


# ─── Main Pipeline ────────────────────────────────────────────────────────
def run_supplement():
    all_results  = []          # Tabel MAE / RMSE
    dm_results   = []          # Tabel DM-Test
    code_errors  = {}          # Untuk Residual Plot (hanya dataset CODE)

    for ds in DATASETS:
        print(f"\n{'='*55}\n  Dataset: {ds.upper()}\n{'='*55}")

        X_train, y_train, X_test, y_test, y_test_real, sc = load_data(ds)

        # ── Naive Baseline ────────────────────────────────────────────
        print("  >> Naive Baseline...")
        # Fitur Lag_1 (kolom pertama X) = nilai t-1 setelah scaling
        naive_scaled = X_test[:, 0]
        naive_real   = inverse_target(naive_scaled, X_test, sc)
        mae_nb  = mean_absolute_error(y_test_real, naive_real)
        rmse_nb = np.sqrt(mean_squared_error(y_test_real, naive_real))
        all_results.append((ds.upper(), 'Naive Baseline', mae_nb, rmse_nb))
        err_nb = y_test_real - naive_real

        # ── LightGBM ─────────────────────────────────────────────────
        print("  >> LightGBM...")
        lgb = LGBMRegressor(random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        pred_lgb  = inverse_target(lgb.predict(X_test), X_test, sc)
        mae_lgb   = mean_absolute_error(y_test_real, pred_lgb)
        rmse_lgb  = np.sqrt(mean_squared_error(y_test_real, pred_lgb))
        all_results.append((ds.upper(), 'LightGBM', mae_lgb, rmse_lgb))
        err_lgb = y_test_real - pred_lgb

        # ── CatBoost ─────────────────────────────────────────────────
        print("  >> CatBoost...")
        cat = CatBoostRegressor(iterations=500, silent=True, random_seed=42)
        cat.fit(X_train, y_train)
        pred_cat  = inverse_target(cat.predict(X_test), X_test, sc)
        mae_cat   = mean_absolute_error(y_test_real, pred_cat)
        rmse_cat  = np.sqrt(mean_squared_error(y_test_real, pred_cat))
        all_results.append((ds.upper(), 'CatBoost', mae_cat, rmse_cat))
        err_cat = y_test_real - pred_cat

        # ── LSTM ─────────────────────────────────────────────────────
        epochs = 50 if ds == 'multimodal' else 100
        X_tr3 = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_te3 = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])
        in_shape = (1, X_train.shape[1])

        print(f"  >> LSTM (epochs={epochs})...")
        lstm = Sequential([Input(shape=in_shape),
                           LSTM(64, activation='relu'),
                           Dense(32, activation='relu'),
                           Dense(1)])
        lstm.compile(optimizer='adam', loss='mse')
        lstm.fit(X_tr3, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
        pred_lstm = inverse_target(lstm.predict(X_te3, verbose=0), X_test, sc)
        mae_lstm  = mean_absolute_error(y_test_real, pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_real, pred_lstm))
        all_results.append((ds.upper(), 'LSTM', mae_lstm, rmse_lstm))
        err_lstm = y_test_real - pred_lstm

        # ── GRU ──────────────────────────────────────────────────────
        print(f"  >> GRU (epochs={epochs})...")
        gru = Sequential([Input(shape=in_shape),
                          GRU(64, activation='relu'),
                          Dense(32, activation='relu'),
                          Dense(1)])
        gru.compile(optimizer='adam', loss='mse')
        gru.fit(X_tr3, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
        pred_gru  = inverse_target(gru.predict(X_te3, verbose=0), X_test, sc)
        mae_gru   = mean_absolute_error(y_test_real, pred_gru)
        rmse_gru  = np.sqrt(mean_squared_error(y_test_real, pred_gru))
        all_results.append((ds.upper(), 'GRU', mae_gru, rmse_gru))
        err_gru = y_test_real - pred_gru

        # ── Diebold-Mariano Test (LSTM vs competitors) ────────────────
        for rival_name, rival_err in [('Naive', err_nb),
                                       ('LightGBM', err_lgb),
                                       ('CatBoost', err_cat),
                                       ('GRU', err_gru)]:
            dm_stat, p_val = dm_test(rival_err, err_lstm)
            significant = 'Ya (p<0.05)' if (p_val is not None and p_val < 0.05) else 'Tidak'
            dm_results.append({
                'Dataset'      : ds.upper(),
                'LSTM vs'      : rival_name,
                'DM Statistic' : f"{dm_stat:.4f}" if dm_stat == dm_stat else 'N/A',
                'p-value'      : f"{p_val:.4f}"  if p_val  == p_val  else 'N/A',
                'LSTM Unggul?' : significant
            })

        # Simpan error khusus dataset CODE untuk residual plot
        if ds == 'code':
            code_errors = {
                'Aktual'        : y_test_real,
                'Naive Baseline': err_nb,
                'LightGBM'      : err_lgb,
                'CatBoost'      : err_cat,
                'LSTM'          : err_lstm,
                'GRU'           : err_gru,
            }

    # ─── Cetak Tabel Komparasi Lengkap ───────────────────────────────────
    print("\n\n=== TABEL KOMPARASI PERFORMA AKHIR (+ Naive Baseline) ===")
    print(f"{'Dataset':<14} {'Model':<16} {'MAE':>15} {'RMSE':>15}")
    print("-" * 63)
    for row in all_results:
        print(f"{row[0]:<14} {row[1]:<16} {row[2]:>15,.2f} {row[3]:>15,.2f}")

    # ─── Cetak Tabel Diebold-Mariano ─────────────────────────────────────
    print("\n\n=== DIEBOLD-MARIANO TEST (LSTM vs Rivals) ===")
    df_dm = pd.DataFrame(dm_results)
    print(df_dm.to_string(index=False))

    # ─── Simpan Tabel ke CSV ──────────────────────────────────────────────
    df_res = pd.DataFrame(all_results, columns=['Dataset','Model','MAE','RMSE'])
    df_res.to_csv(os.path.join(OUT_DIR, 'model_results_with_baseline.csv'), index=False)
    df_dm.to_csv(os.path.join(OUT_DIR, 'dm_test_results.csv'), index=False)
    print(f"\nTabel disimpan ke: {OUT_DIR}")

    # ─── Residual Plot – Dataset CODE ────────────────────────────────────
    print("\nMembuat Residual Plot untuk Dataset CODE...")
    plot_residuals(code_errors)
    print("Semua output selesai dibuat.")


def plot_residuals(errors):
    """
    Membuat 2 panel:
    - Panel Atas  : Perbandingan nilai AKTUAL vs Prediksi per model
    - Panel Bawah : Garis residu (Aktual - Prediksi) per menit untuk tiap model
    """
    colors  = {'Naive Baseline':'gray', 'LightGBM':'steelblue',
                'CatBoost':'darkorange', 'LSTM':'green', 'GRU':'crimson'}
    actuals = errors['Aktual']
    n       = len(actuals)
    menit   = np.arange(1, n+1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Workload Prediction Residual Analysis – CODE Dataset (Coding / Copilot)\n'
                 '[BOTTOM: Residual = 0 indicates perfect prediction; negative = over-predict]',
                 fontsize=13, fontweight='bold', y=0.99)

    # ── Panel 1: Aktual vs Prediksi ───────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(menit, actuals, alpha=0.15, color='black', label='_')
    ax1.plot(menit, actuals, color='black', linewidth=2.5,
             linestyle='--', label='Actual (Ground Truth)', zorder=5)

    for model_name, err in errors.items():
        if model_name == 'Aktual': continue
        pred = actuals - err
        ax1.plot(menit, pred, linewidth=1.4, color=colors.get(model_name,'purple'),
                 label=model_name, alpha=0.85)

    ax1.set_ylabel('Total Tokens per Minute', fontsize=12, fontweight='bold')
    ax1.set_title('Panel A – Actual vs Predicted Values Comparison', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    # Tandai titik SPIKE tertinggi
    peak_idx = np.argmax(actuals)
    ax1.annotate(f'SPIKE\n({actuals[peak_idx]/1000:.0f}K tokens)',
                 xy=(menit[peak_idx], actuals[peak_idx]),
                 xytext=(menit[peak_idx]+0.5, actuals[peak_idx]*0.9),
                 fontsize=10, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # ── Panel 2: Residual Error per model ────────────────────────
    ax2 = axes[1]
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    for model_name, err in errors.items():
        if model_name == 'Aktual': continue
        ax2.plot(menit, err, linewidth=1.4, color=colors.get(model_name,'purple'),
                 label=model_name, alpha=0.85)

    ax2.fill_between(menit,
                     [min(list(errors[m][i] for m in errors if m != 'Aktual')) for i in range(n)],
                     [max(list(errors[m][i] for m in errors if m != 'Aktual')) for i in range(n)],
                     alpha=0.05, color='gray', label='All Models Range')

    ax2.set_xlabel('Minute (Chronological Test Set)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Panel B – Residuals per Minute (Zero = Perfect, Negative = Over-Predict)',
                  fontsize=13)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, 'code_residual_analysis.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Residual plot disimpan: {out}")


if __name__ == '__main__':
    run_supplement()
