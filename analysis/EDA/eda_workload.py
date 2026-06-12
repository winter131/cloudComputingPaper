import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

def load_and_preprocess(csv_path, label):
    df = pd.read_csv(csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['TotalTokens'] = df['ContextTokens'] + df['GeneratedTokens']
    df['WorkloadType'] = label
    
    # Extract components
    df['Hour'] = df['TIMESTAMP'].dt.hour
    df['Minute'] = df['TIMESTAMP'].dt.floor('Min') # Round to minute for time series
    
    return df

def run_comparative_eda(conv_csv, code_csv):
    print("Memuat Dataset Opsi 2 (Studi Komparasi)...")
    df_conv = load_and_preprocess(conv_csv, 'Conversational (Chat)')
    df_code = load_and_preprocess(code_csv, 'Coding (Copilot)')
    
    # Gabungkan data untuk visualisasi komparatif
    df_all = pd.concat([df_conv, df_code], ignore_index=True)
    
    print("\n" + "="*50)
    print("STATISTIK DESKRIPTIF PERBANDINGAN")
    print("="*50)
    
    # Menampilkan ringkasan statistik komparatif
    stats = df_all.groupby('WorkloadType')[['ContextTokens', 'GeneratedTokens', 'TotalTokens']].agg(['mean', 'max']).round(2)
    print(stats)
    
    # =========================================================
    # 1. Grafik Tren "Jam Sibuk" (24 Jam Skala Penuh)
    # =========================================================
    hourly_trend = df_all.groupby(['Hour', 'WorkloadType'])['TotalTokens'].sum().reset_index()
    
    # Membuat index 0-23 agar grafik tetap menampilkan rentang 24 jam penuh
    all_hours = pd.DataFrame({'Hour': range(24)})
    
    plt.figure(figsize=(14, 6))
    # Barplot berdampingan (hue)
    ax1 = sns.barplot(data=hourly_trend, x='Hour', y='TotalTokens', hue='WorkloadType', palette='Set1')
    
    plt.title('Perbandingan Tren Jam Sibuk (Total Token per Jam) - Skala 24 Jam', fontsize=16, fontweight='bold')
    plt.xlabel('Jam (0 - 23)', fontsize=12)
    plt.ylabel('Total Beban Token', fontsize=12)
    
    # Force X-axis to show all 24 hours even if data is empty
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24))
    
    # Format sumbu Y ke numerik
    current_values = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.legend(title='Jenis Workload')
    plt.tight_layout()
    output_path_trend = os.path.join(os.path.dirname(conv_csv), '..', 'analysis', 'hourly_trend_comparison.png')
    plt.savefig(output_path_trend, dpi=300)
    print(f"\nGrafik komparasi jam sibuk (24 Jam) disimpan di: {output_path_trend}")
    
    # =========================================================
    # 2. Grafik Fluktuasi Menit (Detail Kepadatan Trafik)
    # =========================================================
    # Karena dataset aslinya hanya berdurasi sekitar 1 jam, tren per menit jauh lebih informatif
    minute_trend = df_all.groupby(['Minute', 'WorkloadType'])['TotalTokens'].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=minute_trend, x='Minute', y='TotalTokens', hue='WorkloadType', palette='Set1', linewidth=2)
    
    plt.title('Fluktuasi Kepadatan Workload (Total Token per Menit)', fontsize=16, fontweight='bold')
    plt.xlabel('Waktu (Menit)', fontsize=12)
    plt.ylabel('Total Beban Token', fontsize=12)
    plt.xticks(rotation=45)
    
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.tight_layout()
    output_path_minute = os.path.join(os.path.dirname(conv_csv), '..', 'analysis', 'minute_fluctuation_comparison.png')
    plt.savefig(output_path_minute, dpi=300)
    print(f"Grafik fluktuasi per menit disimpan di: {output_path_minute}")

    # =========================================================
    # 3. Grafik Distribusi Token Histogram Komparatif
    # =========================================================
    plt.figure(figsize=(14, 6))
    
    # Fokus pada Total Tokens untuk membandingkan beban keseluruhan
    q_conv = df_conv['TotalTokens'].quantile(0.99)
    q_code = df_code['TotalTokens'].quantile(0.99)
    limit = max(q_conv, q_code) # Gunakan batas maksimal agar sejajar
    
    sns.kdeplot(data=df_conv[df_conv['TotalTokens'] < limit], x='TotalTokens', 
                 fill=True, color='red', label='Conversational (Chat)', alpha=0.5)
    sns.kdeplot(data=df_code[df_code['TotalTokens'] < limit], x='TotalTokens', 
                 fill=True, color='blue', label='Coding (Copilot)', alpha=0.5)
    
    plt.title('Perbandingan Distribusi Total Beban Token (Tanpa Outlier 1%)', fontsize=16, fontweight='bold')
    plt.xlabel('Jumlah Total Token per Request', fontsize=12)
    plt.ylabel('Kepadatan (Density)', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    output_path_dist = os.path.join(os.path.dirname(conv_csv), '..', 'analysis', 'distribution_comparison.png')
    plt.savefig(output_path_dist, dpi=300)
    print(f"Grafik distribusi komparatif disimpan di: {output_path_dist}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conv_csv = os.path.join(base_dir, 'data', 'AzureLLMInferenceTrace_conv.csv')
    code_csv = os.path.join(base_dir, 'data', 'AzureLLMInferenceTrace_code.csv')
    
    if os.path.exists(conv_csv) and os.path.exists(code_csv):
        run_comparative_eda(conv_csv, code_csv)
    else:
        print("Salah satu atau kedua file CSV tidak ditemukan di folder 'data'.")
