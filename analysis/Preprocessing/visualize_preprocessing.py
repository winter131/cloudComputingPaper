# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR = os.path.join(BASE_DIR, 'analysis', 'preprocessed')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

def plot_train_test_split(dataset_name):
    print(f"Visualisasi Train/Test Split untuk {dataset_name}...")
    
    # Load features
    csv_path = os.path.join(PREP_DIR, f'{dataset_name}_features.csv')
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan.")
        return
        
    df = pd.read_csv(csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    split_idx = int(len(df) * 0.8)
    
    plt.figure(figsize=(15, 6))
    
    # Plot Train
    plt.plot(df['TIMESTAMP'].iloc[:split_idx], df['TotalTokens'].iloc[:split_idx], 
             label='Train Set (80%)', color='royalblue', linewidth=1.5)
    
    # Plot Test
    plt.plot(df['TIMESTAMP'].iloc[split_idx:], df['TotalTokens'].iloc[split_idx:], 
             label='Test Set (20%)', color='crimson', linewidth=1.5)
    
    # Garis pemisah
    plt.axvline(x=df['TIMESTAMP'].iloc[split_idx], color='black', linestyle='--', alpha=0.7, label='Split Point')
    
    plt.title(f'Chronological Train-Test Split (Dataset: {dataset_name.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Waktu', fontsize=12)
    plt.ylabel('Total Token per Menit', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(ANALYSIS_DIR, f'{dataset_name}_traintest_split.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_feature_correlation(dataset_name):
    print(f"Visualisasi Korelasi Fitur untuk {dataset_name}...")
    
    csv_path = os.path.join(PREP_DIR, f'{dataset_name}_features.csv')
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    # Fitur yang kita mau lihat korelasinya
    cols = ['TotalTokens', 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3']
    
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    
    plt.title(f'Korelasi Fitur Lag vs Target (Dataset: {dataset_name.upper()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(ANALYSIS_DIR, f'{dataset_name}_correlation.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    for ds in ['conv', 'code', 'multimodal']:
        plot_train_test_split(ds)
        plot_feature_correlation(ds)
        
    print("Semua visualisasi preprocessing selesai dibuat di folder analysis!")
