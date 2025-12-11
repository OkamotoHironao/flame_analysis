#!/usr/bin/env python3
"""
包括的評価スクリプト
6モデル × 5トピック × 2特徴量セット（10特徴 vs 16特徴）の完全比較

目的: 最適な特徴量セットとモデルの組み合わせを決定
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import time

print("=" * 100)
print("包括的評価実験: 6モデル × 5トピック × 2特徴量セット")
print("=" * 100)

# データ読み込み
df = pd.read_csv('outputs/unified_model_v2/combined_labeled.csv')

if 'is_controversy' in df.columns:
    df['label'] = df['is_controversy']
else:
    df['label'] = df['is_flame']

print(f"\n📊 データセット情報")
print(f"  総サンプル数: {len(df)}件")
print(f"  炎上: {(df['label']==1).sum()}件, 非炎上: {(df['label']==0).sum()}件")
print(f"  トピック数: {df['topic'].nunique()}個")

# 特徴量定義
FEATURES_10 = [
    'volume', 'negative_rate', 'stance_against_rate',
    'stance_favor_rate', 'stance_neutral_rate',
    'delta_volume', 'delta_volume_rate',
    'flame_score', 'against_count', 'sentiment_polarity'
]

FEATURES_16 = FEATURES_10 + [
    'delta_negative_rate', 'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean'
]

# モデル定義
def get_models():
    """6モデルを返す (model, use_scaling)"""
    return {
        'CatBoost': (CatBoostClassifier(
            iterations=100, depth=5, learning_rate=0.1,
            random_state=42, verbose=0
        ), False),
        
        'XGBoost': (XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ), False),
        
        'LightGBM': (LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            num_leaves=31, random_state=42, verbose=-1
        ), False),
        
        'Random Forest': (RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ), False),
        
        'SVM (RBF)': (SVC(
            kernel='rbf', C=10, gamma='scale', random_state=42
        ), True),
        
        'Logistic Regression': (LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ), True)
    }

# =================================================================
# 実験1: 標準評価（全トピック混在）
# =================================================================
print("\n" + "=" * 100)
print("実験1: 標準評価（全トピック混在、80/20分割）")
print("=" * 100)

results_standard = {}

for feature_name, features in [("10特徴量", FEATURES_10), ("16特徴量", FEATURES_16)]:
    print(f"\n{'='*50}")
    print(f"特徴量セット: {feature_name} ({len(features)}個)")
    print(f"{'='*50}")
    
    results_standard[feature_name] = {}
    
    X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, (model, use_scaling) in get_models().items():
        X_tr = X_train_scaled if use_scaling else X_train
        X_te = X_test_scaled if use_scaling else X_test
        
        # 訓練
        start_time = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - start_time
        
        # 予測
        y_pred = model.predict(X_te)
        
        # 評価
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # CV評価
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        
        results_standard[feature_name][model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'train_time': train_time
        }
        
        print(f"  {model_name:20s} | F1:{f1*100:6.2f}% | Acc:{acc*100:6.2f}% | "
              f"Prec:{prec*100:6.2f}% | Rec:{rec*100:6.2f}% | "
              f"CV:{cv_scores.mean()*100:6.2f}±{cv_scores.std()*100:4.2f}% | "
              f"Time:{train_time:5.2f}s")

# =================================================================
# 実験2: クロストピック評価（全モデル × 全トピック）
# =================================================================
print("\n" + "=" * 100)
print("実験2: クロストピック評価（Leave-One-Topic-Out）")
print("=" * 100)

def cross_topic_evaluation_all_models(df, features):
    """全モデルでクロストピック評価"""
    topics = df['topic'].unique()
    results = []
    
    for test_topic in topics:
        train_df = df[df['topic'] != test_topic]
        test_df = df[df['topic'] == test_topic]
        
        if len(test_df) < 5:
            continue
        
        X_train = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['label']
        X_test = test_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['label']
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        topic_results = {'topic': test_topic, 'n_test': len(test_df)}
        
        for model_name, (model, use_scaling) in get_models().items():
            X_tr = X_train_scaled if use_scaling else X_train
            X_te = X_test_scaled if use_scaling else X_test
            
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            
            f1 = f1_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            
            topic_results[f'{model_name}_f1'] = f1
            topic_results[f'{model_name}_acc'] = acc
        
        results.append(topic_results)
    
    return pd.DataFrame(results)

results_cross_topic = {}

for feature_name, features in [("10特徴量", FEATURES_10), ("16特徴量", FEATURES_16)]:
    print(f"\n{'='*50}")
    print(f"特徴量セット: {feature_name}")
    print(f"{'='*50}")
    
    results_df = cross_topic_evaluation_all_models(df, features)
    results_cross_topic[feature_name] = results_df
    
    # トピック別結果表示
    print("\nトピック別 F1スコア:")
    print("─" * 100)
    header = f"{'トピック':12s} | {'Test':4s}"
    for model_name in get_models().keys():
        header += f" | {model_name:12s}"
    print(header)
    print("─" * 100)
    
    for _, row in results_df.iterrows():
        line = f"{row['topic']:12s} | {int(row['n_test']):4d}"
        for model_name in get_models().keys():
            f1_val = row[f'{model_name}_f1'] * 100
            line += f" | {f1_val:11.2f}%"
        print(line)
    
    # 各モデルの平均F1
    print("\n各モデルの平均F1スコア（クロストピック）:")
    print("─" * 50)
    for model_name in get_models().keys():
        avg_f1 = results_df[f'{model_name}_f1'].mean() * 100
        std_f1 = results_df[f'{model_name}_f1'].std() * 100
        print(f"  {model_name:20s}: {avg_f1:6.2f}% ± {std_f1:5.2f}%")

# =================================================================
# 総合比較サマリー
# =================================================================
print("\n" + "=" * 100)
print("📊 総合比較サマリー")
print("=" * 100)

# 標準評価の比較
print("\n【標準評価（全トピック混在）】")
print("\n10特徴量:")
print("─" * 100)
print(f"{'モデル':20s} | {'F1':7s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':7s} | {'CV F1':12s}")
print("─" * 100)
for model_name in get_models().keys():
    r = results_standard["10特徴量"][model_name]
    print(f"{model_name:20s} | {r['f1']*100:6.2f}% | {r['accuracy']*100:7.2f}% | "
          f"{r['precision']*100:8.2f}% | {r['recall']*100:6.2f}% | "
          f"{r['cv_f1_mean']*100:5.2f}±{r['cv_f1_std']*100:4.2f}%")

print("\n16特徴量:")
print("─" * 100)
print(f"{'モデル':20s} | {'F1':7s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':7s} | {'CV F1':12s}")
print("─" * 100)
for model_name in get_models().keys():
    r = results_standard["16特徴量"][model_name]
    print(f"{model_name:20s} | {r['f1']*100:6.2f}% | {r['accuracy']*100:7.2f}% | "
          f"{r['precision']*100:8.2f}% | {r['recall']*100:6.2f}% | "
          f"{r['cv_f1_mean']*100:5.2f}±{r['cv_f1_std']*100:4.2f}%")

# モデル別の改善度
print("\n" + "=" * 100)
print("📈 モデル別改善度（10特徴量 → 16特徴量）")
print("=" * 100)
print(f"{'モデル':20s} | {'F1変化':10s} | {'Accuracy変化':12s} | {'CV F1変化':12s} | {'クロスF1変化':14s}")
print("─" * 100)

for model_name in get_models().keys():
    r10_std = results_standard["10特徴量"][model_name]
    r16_std = results_standard["16特徴量"][model_name]
    
    f1_diff = (r16_std['f1'] - r10_std['f1']) * 100
    acc_diff = (r16_std['accuracy'] - r10_std['accuracy']) * 100
    cv_diff = (r16_std['cv_f1_mean'] - r10_std['cv_f1_mean']) * 100
    
    # クロストピック差分
    cross10 = results_cross_topic["10特徴量"][f'{model_name}_f1'].mean()
    cross16 = results_cross_topic["16特徴量"][f'{model_name}_f1'].mean()
    cross_diff = (cross16 - cross10) * 100
    
    print(f"{model_name:20s} | {f1_diff:+9.2f}% | {acc_diff:+11.2f}% | "
          f"{cv_diff:+11.2f}% | {cross_diff:+13.2f}%")

# =================================================================
# 最終推奨
# =================================================================
print("\n" + "=" * 100)
print("💡 最終推奨")
print("=" * 100)

# 各特徴量セットでの最良モデル
print("\n🏆 各特徴量セットでの最良モデル:")

for feature_name in ["10特徴量", "16特徴量"]:
    print(f"\n【{feature_name}】")
    
    # 標準評価での最良
    best_std = max(results_standard[feature_name].items(), 
                   key=lambda x: x[1]['f1'])
    print(f"  標準評価:      {best_std[0]:20s} F1:{best_std[1]['f1']*100:.2f}%")
    
    # クロストピックでの最良
    cross_results = results_cross_topic[feature_name]
    model_avg_f1 = {}
    for model_name in get_models().keys():
        model_avg_f1[model_name] = cross_results[f'{model_name}_f1'].mean()
    best_cross = max(model_avg_f1.items(), key=lambda x: x[1])
    print(f"  クロストピック: {best_cross[0]:20s} F1:{best_cross[1]*100:.2f}%")

# 総合推奨
print("\n🎯 総合推奨:")

# 16特徴量版で改善したモデル数をカウント
improved_count = 0
for model_name in get_models().keys():
    cross10 = results_cross_topic["10特徴量"][f'{model_name}_f1'].mean()
    cross16 = results_cross_topic["16特徴量"][f'{model_name}_f1'].mean()
    if cross16 > cross10:
        improved_count += 1

if improved_count >= 4:  # 6モデル中4以上改善
    print("  ✅ 16特徴量版を推奨")
    print(f"     - {improved_count}/6モデルでクロストピック性能が向上")
    print("     - sentiment_avg_score, stance_mean系が汎化に寄与")
    
    # 最良の組み合わせ
    best_combo = None
    best_f1 = 0
    for model_name in get_models().keys():
        std_f1 = results_standard["16特徴量"][model_name]['f1']
        cross_f1 = results_cross_topic["16特徴量"][f'{model_name}_f1'].mean()
        avg_f1 = (std_f1 + cross_f1) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_combo = (model_name, std_f1, cross_f1)
    
    print(f"\n  🌟 最良の組み合わせ: {best_combo[0]} + 16特徴量")
    print(f"     標準評価F1:      {best_combo[1]*100:.2f}%")
    print(f"     クロストピックF1: {best_combo[2]*100:.2f}%")
    print(f"     平均F1:          {best_f1*100:.2f}%")
    
else:
    print("  ⚠️  10特徴量版を推奨")
    print(f"     - {6-improved_count}/6モデルでクロストピック性能が低下")
    print("     - 追加特徴量の効果が限定的")

print("\n" + "=" * 100)
