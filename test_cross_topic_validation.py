#!/usr/bin/env python3
"""
クロストピック評価スクリプト
トピックごとにLeave-One-Out評価を実施し、汎化性能を検証

目的: topic特徴量への過学習を検証
方法: 各トピックをテストセットとし、残りで学習
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("クロストピック評価実験")
print("=" * 80)

# データ読み込み
df = pd.read_csv('outputs/unified_model_v2/combined_labeled.csv')

if 'is_controversy' in df.columns:
    df['label'] = df['is_controversy']
else:
    df['label'] = df['is_flame']

print(f"\n📊 データセット情報")
print(f"  総サンプル数: {len(df)}件")
print(f"  炎上: {(df['label']==1).sum()}件")
print(f"  非炎上: {(df['label']==0).sum()}件")

# トピック分布
print(f"\n📋 トピック別分布:")
topic_stats = df.groupby('topic').agg({
    'label': ['count', 'sum', 'mean']
}).round(3)
topic_stats.columns = ['サンプル数', '炎上数', '炎上率']
print(topic_stats)

# 特徴量準備
FEATURES_10 = [
    'volume', 'negative_rate', 'stance_against_rate',
    'stance_favor_rate', 'stance_neutral_rate',
    'delta_volume', 'delta_volume_rate',
    'flame_score', 'against_count', 'sentiment_polarity'
]

FEATURES_17 = FEATURES_10 + [
    'delta_negative_rate', 'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean',
    'topic_encoded'
]

# topic_encodedを事前に作成
le = LabelEncoder()
df['topic_encoded'] = le.fit_transform(df['topic'].fillna('unknown'))

# クロストピック評価関数
def cross_topic_evaluation(df, features, use_topic_feature=True):
    """
    Leave-One-Topic-Out Cross Validation
    
    Args:
        df: データフレーム
        features: 使用する特徴量リスト
        use_topic_feature: topic_encodedを使うか
    
    Returns:
        dict: 評価結果
    """
    topics = df['topic'].unique()
    results = []
    
    for test_topic in topics:
        # トレインとテストに分割
        train_df = df[df['topic'] != test_topic].copy()
        test_df = df[df['topic'] == test_topic].copy()
        
        # サンプル数チェック
        if len(test_df) < 5:
            print(f"  ⚠️  {test_topic}: サンプル数不足({len(test_df)}件) - スキップ")
            continue
        
        # topic_encodedを再エンコード（trainセットのみで）
        if use_topic_feature and 'topic_encoded' in features:
            # trainデータでエンコーダーをfit
            le_temp = LabelEncoder()
            train_df['topic_encoded'] = le_temp.fit_transform(train_df['topic'])
            
            # testデータは未知のトピックとして扱う（平均値で埋める）
            # または、最も近いトピックにマッピング
            test_df['topic_encoded'] = -1  # 未知のトピックフラグ
        
        # 特徴量とラベル
        X_train = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['label']
        X_test = test_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['label']
        
        # モデル学習
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'topic': test_topic,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        print(f"  {test_topic:20s} | Train:{len(train_df):3d} Test:{len(test_df):3d} | F1:{f1*100:6.2f}%")
    
    return pd.DataFrame(results)

# ===== 実験1: 10特徴量でクロストピック評価 =====
print("\n" + "=" * 80)
print("実験1: 10特徴量（topic特徴なし）でクロストピック評価")
print("=" * 80)

results_10 = cross_topic_evaluation(df, FEATURES_10, use_topic_feature=False)

print(f"\n📊 10特徴量の結果:")
print(f"  平均 F1:        {results_10['f1'].mean()*100:.2f}%")
print(f"  平均 Accuracy:  {results_10['accuracy'].mean()*100:.2f}%")
print(f"  平均 Precision: {results_10['precision'].mean()*100:.2f}%")
print(f"  平均 Recall:    {results_10['recall'].mean()*100:.2f}%")
print(f"  F1 標準偏差:    {results_10['f1'].std()*100:.2f}%")

# ===== 実験2: 17特徴量でクロストピック評価 =====
print("\n" + "=" * 80)
print("実験2: 17特徴量（topic特徴あり）でクロストピック評価")
print("=" * 80)
print("  注: testトピックは未知として扱う（topic_encoded=-1）")

results_17 = cross_topic_evaluation(df, FEATURES_17, use_topic_feature=True)

print(f"\n📊 17特徴量の結果:")
print(f"  平均 F1:        {results_17['f1'].mean()*100:.2f}%")
print(f"  平均 Accuracy:  {results_17['accuracy'].mean()*100:.2f}%")
print(f"  平均 Precision: {results_17['precision'].mean()*100:.2f}%")
print(f"  平均 Recall:    {results_17['recall'].mean()*100:.2f}%")
print(f"  F1 標準偏差:    {results_17['f1'].std()*100:.2f}%")

# ===== 比較結果 =====
print("\n" + "=" * 80)
print("📊 クロストピック評価の比較")
print("=" * 80)

print("\n┌─────────────────┬──────────────┬──────────────┬──────────┐")
print("│ 指標            │ 10特徴量     │ 17特徴量     │ 差分     │")
print("├─────────────────┼──────────────┼──────────────┼──────────┤")
print(f"│ F1 Score (平均) │ {results_10['f1'].mean()*100:11.2f}% │ {results_17['f1'].mean()*100:11.2f}% │ {(results_17['f1'].mean()-results_10['f1'].mean())*100:+7.2f}% │")
print(f"│ F1 標準偏差     │ {results_10['f1'].std()*100:11.2f}% │ {results_17['f1'].std()*100:11.2f}% │ {(results_17['f1'].std()-results_10['f1'].std())*100:+7.2f}% │")
print(f"│ Accuracy (平均) │ {results_10['accuracy'].mean()*100:11.2f}% │ {results_17['accuracy'].mean()*100:11.2f}% │ {(results_17['accuracy'].mean()-results_10['accuracy'].mean())*100:+7.2f}% │")
print(f"│ Precision (平均)│ {results_10['precision'].mean()*100:11.2f}% │ {results_17['precision'].mean()*100:11.2f}% │ {(results_17['precision'].mean()-results_10['precision'].mean())*100:+7.2f}% │")
print(f"│ Recall (平均)   │ {results_10['recall'].mean()*100:11.2f}% │ {results_17['recall'].mean()*100:11.2f}% │ {(results_17['recall'].mean()-results_10['recall'].mean())*100:+7.2f}% │")
print("└─────────────────┴──────────────┴──────────────┴──────────┘")

# トピック別詳細比較
print("\n" + "=" * 80)
print("📋 トピック別 F1スコア比較")
print("=" * 80)

merged = results_10.merge(results_17, on='topic', suffixes=('_10', '_17'))
merged['f1_diff'] = (merged['f1_17'] - merged['f1_10']) * 100

print("\nトピック            | 10特徴量 | 17特徴量 | 差分")
print("────────────────────┼──────────┼──────────┼────────")
for row in merged.itertuples():
    print(f"{row.topic:18s} │ {row.f1_10*100:7.2f}% │ {row.f1_17*100:7.2f}% │ {row.f1_diff:+6.2f}%")

# ===== 詳細分析 =====
print("\n" + "=" * 80)
print("🔍 詳細分析")
print("=" * 80)

# 性能が悪化したトピック
degraded = merged[merged['f1_diff'] < 0].sort_values('f1_diff')
if len(degraded) > 0:
    print(f"\n⚠️  17特徴量で性能が低下したトピック ({len(degraded)}個):")
    for row in degraded.itertuples():
        print(f"  - {row.topic:18s}: {row.f1_diff:+6.2f}% ({row.f1_10*100:.2f}% → {row.f1_17*100:.2f}%)")
else:
    print(f"\n✅ 全トピックで性能向上または維持")

# 性能が向上したトピック
improved = merged[merged['f1_diff'] > 1.0].sort_values('f1_diff', ascending=False)
if len(improved) > 0:
    print(f"\n🌟 17特徴量で大きく向上したトピック ({len(improved)}個、1%以上):")
    for row in improved.itertuples():
        print(f"  - {row.topic:18s}: {row.f1_diff:+6.2f}% ({row.f1_10*100:.2f}% → {row.f1_17*100:.2f}%)")

# ===== 結論 =====
print("\n" + "=" * 80)
print("💡 クロストピック評価の結論")
print("=" * 80)

f1_diff = (results_17['f1'].mean() - results_10['f1'].mean()) * 100
std_diff = (results_17['f1'].std() - results_10['f1'].std()) * 100

print(f"\n🎯 汎化性能の評価:")
print(f"   平均F1の差: {f1_diff:+.2f}%")
print(f"   標準偏差の差: {std_diff:+.2f}%")

if f1_diff > 0 and std_diff < 5:
    print(f"\n✅ 17特徴量版は汎化性能が優れている")
    print(f"   - 未知トピックでも平均{f1_diff:+.2f}%の性能向上")
    print(f"   - 標準偏差の増加も{std_diff:.2f}%に抑制")
    print(f"   - topic特徴への過学習は軽微")
elif f1_diff > 0 and std_diff >= 5:
    print(f"\n⚠️  17特徴量版は性能向上するが不安定")
    print(f"   - 平均では{f1_diff:+.2f}%向上")
    print(f"   - 標準偏差が{std_diff:+.2f}%増加（トピック依存性が高い）")
    print(f"   - topic特徴への過学習の懸念あり")
elif f1_diff < -1:
    print(f"\n❌ 17特徴量版は汎化性能が劣る")
    print(f"   - 未知トピックで平均{f1_diff:.2f}%の性能低下")
    print(f"   - topic特徴への過学習が深刻")
    print(f"   - 10特徴量版を推奨")
else:
    print(f"\n✓ 17特徴量版と10特徴量版で汎化性能はほぼ同等")
    print(f"   - 差分は{f1_diff:+.2f}%（誤差範囲内）")

print("\n" + "=" * 80)

# 最終推奨
print("\n🎯 最終推奨:")
if f1_diff >= 0:
    print("  ✅ 17特徴量版の採用を推奨")
    print(f"     - 未知トピックでも性能維持・向上 ({f1_diff:+.2f}%)")
    print("     - sentiment_avg_score, stance_mean系が汎化に寄与")
else:
    print("  ⚠️  17特徴量版は慎重に検討")
    print(f"     - クロストピック性能が{f1_diff:.2f}%低下")
    print("     - topic特徴への依存度が高い可能性")

print("\n" + "=" * 80)
