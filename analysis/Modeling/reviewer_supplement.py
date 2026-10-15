# -*- coding: utf-8 -*-
"""
reviewer_supplement.py
======================
Script to address IEEE reviewer comments:
1. Workload characteristics & volatility profiling (Duration, N, Mean, Std, CV, PAR, ACF, Spikes).
2. Additional evaluation metrics (WAPE, NRMSE) across all models.
3. Fully pairwise Diebold-Mariano statistical significance test across all 5 models.
"""

import os, pickle, random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Input

# Lock seed
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP_DIR = os.path.join(BASE_DIR, 'analysis', 'Preprocessing', 'preprocessed')
DATASETS = ['conv', 'code', 'multimodal']

def inverse_target(preds, X_ref, scaler):
    dummy = np.zeros((len(preds), X_ref.shape[1] + 1))
    dummy[:, 0] = np.array(preds).flatten()
    dummy[:, 1:] = X_ref
    return scaler.inverse_transform(dummy)[:, 0]

def load_data(ds):
    train = np.load(os.path.join(PREP_DIR, f"{ds}_train_scaled.npy"))
    test  = np.load(os.path.join(PREP_DIR, f"{ds}_test_scaled.npy"))
    with open(os.path.join(PREP_DIR, f"{ds}_scaler.pkl"), 'rb') as f:
        sc = pickle.load(f)['scaler']
    y_train, X_train = train[:, 0], train[:, 1:]
    y_test,  X_test  = test[:, 0],  test[:, 1:]
    y_test_real = inverse_target(y_test, X_test, sc)
    return X_train, y_train, X_test, y_test, y_test_real, sc

def dm_test(e1, e2, h=1):
    """Diebold-Mariano test with Harvey et al. (1997) correction."""
    T = len(e1)
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    gamma = [np.cov(d[lag:], d[:T-lag])[0, 1] if lag > 0 else np.var(d, ddof=1)
             for lag in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d)
    harvey_adj = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat_adj = dm_stat * harvey_adj
    p_val = 2 * (1 - stats.t.cdf(abs(dm_stat_adj), df=T-1))
    return dm_stat_adj, p_val

def compute_acf(series, lag):
    if len(series) <= lag:
        return np.nan
    y = series - np.mean(series)
    denom = np.sum(y**2)
    if denom == 0:
        return 0.0
    numer = np.sum(y[:-lag] * y[lag:])
    return numer / denom

print("=====================================================================")
print("PART 1: WORKLOAD PROFILING & VOLATILITY CHARACTERISTICS")
print("=====================================================================")

profile_rows = []
for ds in DATASETS:
    resampled_file = os.path.join(PREP_DIR, f"{ds}_resampled.csv")
    df = pd.read_csv(resampled_file)
    # The column may be 'TotalTokens' or the series values
    tokens = df['TotalTokens'].values if 'TotalTokens' in df.columns else df.iloc[:, 1].values
    
    total_n = len(tokens)
    split_info_file = os.path.join(PREP_DIR, f"{ds}_split_info.txt")
    with open(split_info_file, 'r') as f:
        info_txt = f.read()
    
    # Statistics
    mean_val = np.mean(tokens)
    median_val = np.median(tokens)
    std_val = np.std(tokens, ddof=1)
    min_val = np.min(tokens)
    max_val = np.max(tokens)
    cv_val = std_val / mean_val if mean_val > 0 else np.nan
    par_val = max_val / mean_val if mean_val > 0 else np.nan
    
    # ACF
    acf_1 = compute_acf(tokens, 1)
    acf_5 = compute_acf(tokens, 5)
    acf_15 = compute_acf(tokens, 15)
    acf_60 = compute_acf(tokens, 60)
    
    # Spikes > 95th percentile
    p95 = np.percentile(tokens, 95)
    spikes_count = np.sum(tokens > p95)
    spikes_pct = (spikes_count / total_n) * 100
    
    profile_rows.append({
        'Dataset': ds.upper(),
        'Duration': f"~{total_n} min (~{total_n/60:.1f} hrs)" if total_n < 1440 else f"7 days ({total_n} min)",
        'Total N': total_n,
        'Mean': mean_val,
        'Median': median_val,
        'Std Dev': std_val,
        'Min': min_val,
        'Max': max_val,
        'CV (std/mean)': cv_val,
        'PAR (max/mean)': par_val,
        'ACF Lag-1': acf_1,
        'ACF Lag-5': acf_5,
        'ACF Lag-15': acf_15,
        'ACF Lag-60': acf_60,
        'Spike Thresh (>95th)': p95,
        'Spikes Count': spikes_count
    })

df_profile = pd.DataFrame(profile_rows)

def df_to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                row_str.append(f"{val:,.4f}" if abs(val) < 1000 else f"{val:,.2f}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

print("\n--- TABLE I: WORKLOAD DESCRIPTIVE & VOLATILITY CHARACTERISTICS ---")
md_profile_cols = ['Dataset', 'Duration', 'Total N', 'Mean', 'Std Dev', 'CV (std/mean)', 'PAR (max/mean)', 'ACF Lag-1', 'ACF Lag-5', 'Spike Thresh (>95th)']
print(df_to_markdown(df_profile[md_profile_cols]))

print("\n\n=====================================================================")
print("PART 2: MODEL TRAINING, EVALUATION (MAE, RMSE, WAPE, NRMSE)")
print("=====================================================================")

perf_rows = []
all_errors = {}

for ds in DATASETS:
    print(f"\nProcessing {ds.upper()}...")
    X_train, y_train, X_test, y_test, y_test_real, sc = load_data(ds)
    all_errors[ds] = {}
    
    # 1. Naive Baseline
    feat_csv = os.path.join(PREP_DIR, f"{ds}_features.csv")
    df_feat = pd.read_csv(feat_csv)
    n_test = len(y_test_real)
    pred_nb = df_feat['Lag_1'].iloc[-n_test:].values
    err_nb = y_test_real - pred_nb
    all_errors[ds]['Naive'] = err_nb
    
    # 2. LightGBM
    lgb = LGBMRegressor(random_state=42, verbose=-1)
    lgb.fit(X_train, y_train)
    pred_lgb = inverse_target(lgb.predict(X_test), X_test, sc)
    err_lgb = y_test_real - pred_lgb
    all_errors[ds]['LightGBM'] = err_lgb
    
    # 3. CatBoost
    cat = CatBoostRegressor(random_seed=42, verbose=0)
    cat.fit(X_train, y_train)
    pred_cat = inverse_target(cat.predict(X_test), X_test, sc)
    err_cat = y_test_real - pred_cat
    all_errors[ds]['CatBoost'] = err_cat
    
    # 4. LSTM
    epochs = 50 if ds == 'multimodal' else 100
    X_tr3 = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_te3 = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])
    in_shape = (1, X_train.shape[1])
    
    lstm = Sequential([Input(shape=in_shape),
                       LSTM(64, activation='relu'),
                       Dense(32, activation='relu'),
                       Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_tr3, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
    pred_lstm = inverse_target(lstm.predict(X_te3, verbose=0), X_test, sc)
    err_lstm = y_test_real - pred_lstm
    all_errors[ds]['LSTM'] = err_lstm
    
    # 5. GRU
    gru = Sequential([Input(shape=in_shape),
                      GRU(64, activation='relu'),
                      Dense(32, activation='relu'),
                      Dense(1)])
    gru.compile(optimizer='adam', loss='mse')
    gru.fit(X_tr3, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
    pred_gru  = inverse_target(gru.predict(X_te3, verbose=0), X_test, sc)
    err_gru = y_test_real - pred_gru
    all_errors[ds]['GRU'] = err_gru
    
    # Calculate metrics
    models_dict = {
        'Naive Baseline': pred_nb,
        'LightGBM': pred_lgb,
        'CatBoost': pred_cat,
        'LSTM': pred_lstm,
        'GRU': pred_gru
    }
    
    y_mean = np.mean(y_test_real)
    sum_y = np.sum(y_test_real)
    
    for m_name, preds in models_dict.items():
        mae = mean_absolute_error(y_test_real, preds)
        rmse = np.sqrt(mean_squared_error(y_test_real, preds))
        wape = (np.sum(np.abs(y_test_real - preds)) / sum_y) * 100
        nrmse = (rmse / y_mean) * 100
        
        perf_rows.append({
            'Dataset': ds.upper(),
            'Model': m_name,
            'MAE': mae,
            'RMSE': rmse,
            'WAPE (%)': wape,
            'NRMSE (%)': nrmse
        })

df_perf = pd.DataFrame(perf_rows)
print("\n--- TABLE II: PREDICTIVE PERFORMANCE (MAE, RMSE, WAPE, NRMSE) ---")
print(df_to_markdown(df_perf))

print("\n\n=====================================================================")
print("PART 3: FULLY PAIRWISE DIEBOLD-MARIANO TEST")
print("=====================================================================")

model_names = ['Naive', 'LightGBM', 'CatBoost', 'LSTM', 'GRU']
dm_pairwise_rows = []

for ds in DATASETS:
    err_dict = all_errors[ds]
    # All pairs (5 choose 2 = 10 pairs)
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            e1 = err_dict[m1]
            e2 = err_dict[m2]
            
            # DM test: e1 is Model 1, e2 is Model 2
            # d = e1^2 - e2^2
            # If d_bar < 0 => Model 1 has smaller squared error => Model 1 is better
            # If d_bar > 0 => Model 2 has smaller squared error => Model 2 is better
            dm_stat, p_val = dm_test(e1, e2)
            
            winner = 'Statistically Tied'
            if p_val < 0.05:
                if dm_stat < 0:
                    winner = f"{m1} (p={p_val:.4f})"
                else:
                    winner = f"{m2} (p={p_val:.4f})"
            
            dm_pairwise_rows.append({
                'Dataset': ds.upper(),
                'Model 1': m1,
                'Model 2': m2,
                'DM Statistic': dm_stat,
                'p-value': p_val,
                'Significant Winner (p < 0.05)': winner
            })

df_dm_pairwise = pd.DataFrame(dm_pairwise_rows)
print("\n--- TABLE III: FULLY PAIRWISE DIEBOLD-MARIANO TEST (10 PAIRS PER DATASET) ---")
print(df_to_markdown(df_dm_pairwise))

# Export to CSV for reference
df_profile.to_csv(os.path.join(PREP_DIR, 'reviewer_workload_characteristics.csv'), index=False)
df_perf.to_csv(os.path.join(PREP_DIR, 'reviewer_performance_with_wape_nrmse.csv'), index=False)
df_dm_pairwise.to_csv(os.path.join(PREP_DIR, 'reviewer_dm_fully_pairwise.csv'), index=False)
print("\nAll reviewer supplement tables exported to CSV in:", PREP_DIR)
