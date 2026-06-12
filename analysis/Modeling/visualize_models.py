# -*- coding: utf-8 -*-
"""
visualize_models.py
===================
Script untuk memvisualisasikan hasil perbandingan (MAE & RMSE) dari 
model Machine Learning & Deep Learning (TERMASUK Naive Baseline).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Konfigurasi visualisasi
sns.set_theme(style="whitegrid", context="talk")

# Data hasil eksekusi model (Terkunci Seed Global)
data = {
    'Dataset': ['CONV', 'CONV', 'CONV', 'CONV', 'CONV',
                'CODE', 'CODE', 'CODE', 'CODE', 'CODE',
                'MULTIMODAL', 'MULTIMODAL', 'MULTIMODAL', 'MULTIMODAL', 'MULTIMODAL'],
    'Model': ['Naive Baseline', 'LightGBM', 'CatBoost', 'LSTM', 'GRU',
              'Naive Baseline', 'LightGBM', 'CatBoost', 'LSTM', 'GRU',
              'Naive Baseline', 'LightGBM', 'CatBoost', 'LSTM', 'GRU'],
    'MAE': [58091.72, 131623.16, 150513.62, 113968.24, 96955.66,
            183912.45, 500167.74, 665869.10, 467517.16, 602260.92,
            203505.46, 180545.40, 179195.51, 176296.13, 176038.49],
    'RMSE': [88904.94, 156776.73, 179482.35, 139275.57, 126423.96,
             240293.70, 538493.41, 755269.52, 513970.39, 656365.92,
             291512.13, 259277.91, 256986.28, 261676.70, 260482.15]
}

df = pd.DataFrame(data)

# Direktori Output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_metric(metric_name, title_text, filename):
    plt.figure(figsize=(14, 8))
    
    # Membuat Grouped Bar Chart
    # Warna: Naive (Abu), LightGBM (Biru), CatBoost (Oranye), LSTM (Hijau), GRU (Merah)
    ax = sns.barplot(data=df, x='Dataset', y=metric_name, hue='Model', 
                     palette=['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     edgecolor='black', linewidth=1)
    
    # Anotasi nilai di atas batang
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height/1000:.0f}K', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, 
                        xytext=(0, 5), textcoords='offset points', fontweight='bold')

    plt.title(title_text, fontsize=18, fontweight='bold', pad=20)
    plt.ylabel(f'{metric_name} (Total Tokens)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset Scenario', fontsize=14, fontweight='bold')
    plt.legend(title='Predictive Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Membentuk visualisasi komparasi performa model...")
    plot_metric('MAE', 'Final Predictive Accuracy Comparison: Mean Absolute Error (Lower is Better)', 'model_comparison_MAE.png')
    plot_metric('RMSE', 'Final Spike Sensitivity Comparison: Root Mean Squared Error (Lower is Better)', 'model_comparison_RMSE.png')
    print(f"Visualisasi berhasil disimpan di direktori: {OUTPUT_DIR}")
