#!/usr/bin/env python3
"""
特徴量データの可視化

炎上期間を特定するために、主要な特徴量をプロットします。
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def visualize_features(csv_path: str, output_path: str = None):
    """
    特徴量データを可視化
    
    Args:
        csv_path: 特徴量CSVファイルのパス
        output_path: 出力画像のパス（Noneの場合は表示のみ）
    """
    # データ読み込み
    print(f"📊 データ読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # timestamp列を解析
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✓ {len(df)}件のレコードを読み込みました")
    print(f"  期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    
    # 統計情報表示
    print("\n📈 基本統計:")
    print(f"  投稿量 - 平均: {df['volume'].mean():.2f}, 最大: {df['volume'].max()}")
    print(f"  ネガティブ率 - 平均: {df['negative_rate'].mean():.2f}, 最大: {df['negative_rate'].max():.2f}")
    print(f"  批判的立場率 - 平均: {df['stance_against_rate'].mean():.2f}, 最大: {df['stance_against_rate'].max():.2f}")
    
    # 閾値超えの時間帯を表示
    print("\n⚠️  注目すべき時間帯:")
    
    # 投稿量が多い時間帯（平均+2σ以上）
    volume_threshold = df['volume'].mean() + 2 * df['volume'].std()
    high_volume = df[df['volume'] > volume_threshold]
    if len(high_volume) > 0:
        print(f"\n  📈 投稿量が多い時間帯 (volume > {volume_threshold:.1f}):")
        for _, row in high_volume.iterrows():
            print(f"    {row['timestamp']}: volume={row['volume']:.0f}, negative_rate={row['negative_rate']:.2f}")
    
    # ネガティブ率が高い時間帯（0.7以上）
    high_negative = df[df['negative_rate'] >= 0.7]
    if len(high_negative) > 0:
        print(f"\n  😡 ネガティブ率が高い時間帯 (negative_rate >= 0.7):")
        for _, row in high_negative.head(10).iterrows():
            print(f"    {row['timestamp']}: negative_rate={row['negative_rate']:.2f}, volume={row['volume']:.0f}")
    
    # 批判的立場率が高い時間帯（0.5以上）
    high_against = df[df['stance_against_rate'] >= 0.5]
    if len(high_against) > 0:
        print(f"\n  🔴 批判的立場率が高い時間帯 (stance_against_rate >= 0.5):")
        for _, row in high_against.head(10).iterrows():
            print(f"    {row['timestamp']}: against_rate={row['stance_against_rate']:.2f}, volume={row['volume']:.0f}")
    
    # 急増している時間帯（delta_volume > 0 かつ volume > 平均）
    rapid_increase = df[(df['delta_volume'] > 0) & (df['volume'] > df['volume'].mean())]
    if len(rapid_increase) > 0:
        print(f"\n  📊 急増している時間帯:")
        for _, row in rapid_increase.head(10).iterrows():
            print(f"    {row['timestamp']}: delta_volume={row['delta_volume']:.0f}, volume={row['volume']:.0f}")
    
    # グラフ作成
    print("\n🎨 グラフ作成中...")
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('炎上検知用 特徴量トレンド分析', fontsize=16, y=0.995)
    
    # 日本語フォント設定
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    # 1. 投稿量
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['volume'], color='steelblue', linewidth=1.5)
    ax1.fill_between(df['timestamp'], df['volume'], alpha=0.3, color='steelblue')
    ax1.axhline(y=df['volume'].mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax1.set_ylabel('Volume', fontsize=11)
    ax1.set_title('1. Volume (Tweet Count per Hour)', fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ネガティブ率
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['negative_rate'], color='orangered', linewidth=1.5)
    ax2.fill_between(df['timestamp'], df['negative_rate'], alpha=0.3, color='orangered')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    ax2.set_ylabel('Negative Rate', fontsize=11)
    ax2.set_title('2. Negative Rate (Sentiment)', fontsize=12, pad=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 批判的立場率
    ax3 = axes[2]
    ax3.plot(df['timestamp'], df['stance_against_rate'], color='crimson', linewidth=1.5)
    ax3.fill_between(df['timestamp'], df['stance_against_rate'], alpha=0.3, color='crimson')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax3.set_ylabel('Against Rate', fontsize=11)
    ax3.set_title('3. Stance AGAINST Rate', fontsize=12, pad=10)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 投稿変化量
    ax4 = axes[3]
    colors = ['green' if x > 0 else 'red' for x in df['delta_volume']]
    ax4.bar(df['timestamp'], df['delta_volume'], color=colors, alpha=0.6, width=0.03)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_ylabel('Delta Volume', fontsize=11)
    ax4.set_title('4. Delta Volume (Volume Change)', fontsize=12, pad=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 複合指標（炎上スコア）
    ax5 = axes[4]
    # 簡易的な炎上スコア = volume_norm * negative_rate * stance_against_rate
    volume_norm = df['volume'] / df['volume'].max() if df['volume'].max() > 0 else 0
    controversy_score = volume_norm * df['negative_rate'] * df['stance_against_rate']
    ax5.plot(df['timestamp'], controversy_score, color='purple', linewidth=1.5)
    ax5.fill_between(df['timestamp'], controversy_score, alpha=0.3, color='purple')
    ax5.set_ylabel('Controversy Score', fontsize=11)
    ax5.set_title('5. Controversy Score (Volume × Negative × Against)', fontsize=12, pad=10)
    ax5.set_xlabel('Timestamp', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ グラフを保存しました: {output_path}")
    else:
        print("✓ グラフを表示します")
        plt.show()
    
    plt.close()
    
    # 炎上期間の推奨
    print("\n" + "="*60)
    print("💡 炎上期間の推奨")
    print("="*60)
    
    # 複合指標で上位の期間を抽出
    df_scored = df.copy()
    df_scored['controversy_score'] = controversy_score
    df_scored = df_scored.sort_values('controversy_score', ascending=False)
    
    top_periods = df_scored[df_scored['controversy_score'] > 0].head(20)
    
    if len(top_periods) > 0:
        print("\n🔥 炎上スコアが高い時間帯（Top 20）:")
        for i, (_, row) in enumerate(top_periods.iterrows(), 1):
            print(f"  {i}. {row['timestamp']}")
            print(f"      score={row['controversy_score']:.3f}, volume={row['volume']:.0f}, "
                  f"negative={row['negative_rate']:.2f}, against={row['stance_against_rate']:.2f}")
        
        # 連続する期間をグループ化
        print("\n📅 推奨ラベリング期間（連続する高スコア時間帯）:")
        
        # 1時間以内に複数の高スコア時間帯がある場合、まとめる
        threshold = top_periods['controversy_score'].quantile(0.5)
        high_score_times = df[df_scored['controversy_score'] > threshold].sort_values('timestamp')
        
        if len(high_score_times) > 0:
            periods = []
            current_start = high_score_times.iloc[0]['timestamp']
            current_end = current_start
            
            for i in range(1, len(high_score_times)):
                prev_time = high_score_times.iloc[i-1]['timestamp']
                curr_time = high_score_times.iloc[i]['timestamp']
                
                # 2時間以内なら同じ期間とみなす
                if (curr_time - prev_time).total_seconds() <= 7200:
                    current_end = curr_time
                else:
                    periods.append((current_start, current_end))
                    current_start = curr_time
                    current_end = curr_time
            
            periods.append((current_start, current_end))
            
            for i, (start, end) in enumerate(periods, 1):
                print(f"\n  期間 {i}:")
                print(f"    start: \"{start}\"")
                print(f"    end: \"{end}\"")
                print(f"    label: \"炎上期間{i}\"")
    else:
        print("\n⚠️  炎上スコアが高い時間帯が見つかりませんでした。")
        print("     閾値を調整するか、データを確認してください。")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使い方: python visualize_features.py <feature_table.csv> [output.png]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_features(csv_path, output_path)
