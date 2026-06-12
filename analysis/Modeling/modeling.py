# -*- coding: utf-8 -*-
"""
modeling.py
===========
Skrip untuk melatih dan mengevaluasi algoritma ML & Deep Learning
untuk prediksi TotalTokens pada dataset telemetri Azure.
"""

import os
import numpy as np
import pickle
import pandas as pd

# Matriks metrik evaluasi
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model Machine Learning
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Model Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

# Mengunci random seed agar hasil eksperimen bisa direproduksi
np.random.seed(42)
tf.random.set_seed(42)

# Konfigurasi Direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR = os.path.join(BASE_DIR, 'analysis', 'Preprocessing', 'preprocessed')
DATASETS = ['conv', 'code', 'multimodal']

# Dictionary untuk menyimpan hasil
results = []

def inverse_transform_target(predictions, x_features, scaler):
    """
    Scaler di-fit pada [Target, Fitur_1, Fitur_2, ...].
    Untuk mengembalikan prediksi target ke skala asli, kita harus merakit
    matriks berdimensi (baris, total_kolom) lagi.
    """
    dummy_matrix = np.zeros((len(predictions), x_features.shape[1] + 1))
    dummy_matrix[:, 0] = predictions.flatten()
    dummy_matrix[:, 1:] = x_features
    return scaler.inverse_transform(dummy_matrix)[:, 0]

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_models():
    for ds in DATASETS:
        print(f"\n{'='*50}\nMemproses Dataset: {ds.upper()}\n{'='*50}")
        
        # Load numpy arrays
        train_path = os.path.join(PREP_DIR, f"{ds}_train_scaled.npy")
        test_path = os.path.join(PREP_DIR, f"{ds}_test_scaled.npy")
        scaler_path = os.path.join(PREP_DIR, f"{ds}_scaler.pkl")
        
        if not os.path.exists(train_path):
            print(f"Data preprocessed untuk {ds} tidak ditemukan!")
            continue
            
        train_scaled = np.load(train_path)
        test_scaled = np.load(test_path)
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            
        # Target adalah kolom ke-0, Fitur adalah sisa kolom
        y_train = train_scaled[:, 0]
        X_train = train_scaled[:, 1:]
        
        y_test = test_scaled[:, 0]
        X_test = test_scaled[:, 1:]
        
        # Kembalikan y_test ke skala original (untuk perhitungan error yang nyata)
        y_test_real = inverse_transform_target(y_test, X_test, scaler)
        
        # ----------------------------------------------------
        # 1. LIGHTGBM
        # ----------------------------------------------------
        print(">> Training LightGBM...")
        lgb = LGBMRegressor(random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        pred_lgb = lgb.predict(X_test)
        real_pred_lgb = inverse_transform_target(pred_lgb, X_test, scaler)
        mae_lgb = mean_absolute_error(y_test_real, real_pred_lgb)
        rmse_lgb = np.sqrt(mean_squared_error(y_test_real, real_pred_lgb))
        results.append((ds.upper(), 'LightGBM', mae_lgb, rmse_lgb))
        
        # ----------------------------------------------------
        # 2. CATBOOST
        # ----------------------------------------------------
        print(">> Training CatBoost...")
        cat = CatBoostRegressor(iterations=500, silent=True, random_seed=42)
        cat.fit(X_train, y_train)
        pred_cat = cat.predict(X_test)
        real_pred_cat = inverse_transform_target(pred_cat, X_test, scaler)
        mae_cat = mean_absolute_error(y_test_real, real_pred_cat)
        rmse_cat = np.sqrt(mean_squared_error(y_test_real, real_pred_cat))
        results.append((ds.upper(), 'CatBoost', mae_cat, rmse_cat))
        
        # ----------------------------------------------------
        # DEEP LEARNING PREPARATION
        # Reshape array dari 2D (samples, features) menjadi 3D (samples, timesteps=1, features)
        # ----------------------------------------------------
        X_train_dl = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        dl_input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])
        
        epochs_setting = 50 if ds == 'multimodal' else 100 # Multimodal sangat besar
        
        # ----------------------------------------------------
        # 3. LSTM
        # ----------------------------------------------------
        print(f">> Training LSTM (Epochs: {epochs_setting})...")
        lstm = build_lstm_model(dl_input_shape)
        lstm.fit(X_train_dl, y_train, epochs=epochs_setting, batch_size=32, verbose=0, shuffle=False)
        pred_lstm = lstm.predict(X_test_dl, verbose=0)
        real_pred_lstm = inverse_transform_target(pred_lstm, X_test, scaler)
        mae_lstm = mean_absolute_error(y_test_real, real_pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_real, real_pred_lstm))
        results.append((ds.upper(), 'LSTM', mae_lstm, rmse_lstm))
        
        # ----------------------------------------------------
        # 4. GRU
        # ----------------------------------------------------
        print(f">> Training GRU (Epochs: {epochs_setting})...")
        gru = build_gru_model(dl_input_shape)
        gru.fit(X_train_dl, y_train, epochs=epochs_setting, batch_size=32, verbose=0, shuffle=False)
        pred_gru = gru.predict(X_test_dl, verbose=0)
        real_pred_gru = inverse_transform_target(pred_gru, X_test, scaler)
        mae_gru = mean_absolute_error(y_test_real, real_pred_gru)
        rmse_gru = np.sqrt(mean_squared_error(y_test_real, real_pred_gru))
        results.append((ds.upper(), 'GRU', mae_gru, rmse_gru))

    # Print Final Markdown Table
    print("\n\n")
    print("### Tabel Komparasi Performa Akhir (TotalTokens / Menit)")
    print("| Dataset | Model | MAE (Token) | RMSE (Token) |")
    print("| :--- | :--- | :--- | :--- |")
    for row in results:
        print(f"| **{row[0]}** | {row[1]} | {row[2]:,.2f} | {row[3]:,.2f} |")

if __name__ == '__main__':
    evaluate_models()
