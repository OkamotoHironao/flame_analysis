#!/usr/bin/env python3
"""
14特徴量版の検証
topic_encodedを除外し、sentiment/stance系の追加特徴のみ使用

目的: 汎化性能と精度のバランス検証
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("14特徴量版の検証実験")
print("topic_encodedを除外、sentiment/stance系のみ追加")
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

# 特徴量定義
FEATURES_10 = [
    'volume', 'negative_rate', 'stance_against_rate',
    'stance_favor_rate', 'stance_neutral_rate',
    'delta_volume', 'delta_volume_rate',
    'flame_score', 'against_count', 'sentiment_polarity'
]

FEATURES_14 = FEATURES_10 + [
    'delta_negative_rate', 'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean'
]

print(f"\n📋 使用特徴量 ({len(FEATURES_14)}個):")
print("\n【既存10特徴量】")
for i, f in enumerate(FEATURES_10, 1):
    print(f"  {i:2d}. {f}")

print("\n【追加4特徴量】")
for i, f in enumerate(FEATURES_14[10:], 11):
    print(f"  {i:2d}. {f}")

# =================================================================
# 実験1: 標準評価（train/test split）
# =================================================================
print("\n" + "=" * 80)
print("実験1: 標準評価（80/20分割、同一トピック内での評価）")
print("=" * 80)

results_standard = {}

for name, features in [("10特徴量", FEATURES_10), ("14特徴量", FEATURES_14)]:
    print(f"\n🤖 {name}版の評価")
    
    X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # CatBoost
    model = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # CV評価
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    
    results_standard[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'feature_importance': model.feature_importances_,
        'features': features
    }
    
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  CV F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =================================================================
# 実験2: クロストピック評価
# =================================================================
print("\n" + "=" * 80)
print("実験2: クロストピック評価（未知トピックでの汎化性能）")
print("=" * 80)

def cross_topic_evaluation(df, features):
    """Leave-One-Topic-Out Cross Validation"""
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
        
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'topic': test_topic,
            'n_test': len(test_df),
            'f1': f1,
            'accuracy': acc
        })
        
        print(f"  {test_topic:20s} | Test:{len(test_df):3d} | F1:{f1*100:6.2f}% | Acc:{acc*100:6.2f}%")
    
    return pd.DataFrame(results)

results_cross_topic = {}

for name, features in [("10特徴量", FEATURES_10), ("14特徴量", FEATURES_14)]:
    print(f"\n🔄 {name}版のクロストピック評価")
    results_df = cross_topic_evaluation(df, features)
    results_cross_topic[name] = results_df
    
    print(f"\n  平均 F1:       {results_df['f1'].mean()*100:.2f}%")
    print(f"  F1 標準偏差:   {results_df['f1'].std()*100:.2f}%")
    print(f"  平均 Accuracy: {results_df['accuracy'].mean()*100:.2f}%")

# =================================================================
# 比較結果サマリー
# =================================================================
print("\n" + "=" * 80)
print("📊 総合比較サマリー")
print("=" * 80)

print("\n【標準評価（同一トピック内）】")
print("┌─────────────────┬──────────────┬──────────────┬──────────┐")
print("│ 指標            │ 10特徴量     │ 14特徴量     │ 差分     │")
print("├─────────────────┼──────────────┼──────────────┼──────────┤")
r10 = results_standard["10特徴量"]
r14 = results_standard["14特徴量"]
print(f"│ F1 Score        │ {r10['f1']*100:11.2f}% │ {r14['f1']*100:11.2f}% │ {(r14['f1']-r10['f1'])*100:+7.2f}% │")
print(f"│ Accuracy        │ {r10['accuracy']*100:11.2f}% │ {r14['accuracy']*100:11.2f}% │ {(r14['accuracy']-r10['accuracy'])*100:+7.2f}% │")
print(f"│ Precision       │ {r10['precision']*100:11.2f}% │ {r14['precision']*100:11.2f}% │ {(r14['precision']-r10['precision'])*100:+7.2f}% │")
print(f"│ Recall          │ {r10['recall']*100:11.2f}% │ {r14['recall']*100:11.2f}% │ {(r14['recall']-r10['recall'])*100:+7.2f}% │")
print(f"│ CV F1 (平均)    │ {r10['cv_f1_mean']*100:11.2f}% │ {r14['cv_f1_mean']*100:11.2f}% │ {(r14['cv_f1_mean']-r10['cv_f1_mean'])*100:+7.2f}% │")
print("└─────────────────┴──────────────┴──────────────┴──────────┘")

print("\n【クロストピック評価（未知トピック）】")
print("┌─────────────────┬──────────────┬──────────────┬──────────┐")
print("│ 指標            │ 10特徴量     │ 14特徴量     │ 差分     │")
print("├─────────────────┼──────────────┼──────────────┼──────────┤")
c10 = results_cross_topic["10特徴量"]
c14 = results_cross_topic["14特徴量"]
print(f"│ F1 (平均)       │ {c10['f1'].mean()*100:11.2f}% │ {c14['f1'].mean()*100:11.2f}% │ {(c14['f1'].mean()-c10['f1'].mean())*100:+7.2f}% │")
print(f"│ F1 標準偏差     │ {c10['f1'].std()*100:11.2f}% │ {c14['f1'].std()*100:11.2f}% │ {(c14['f1'].std()-c10['f1'].std())*100:+7.2f}% │")
print(f"│ Accuracy (平均) │ {c10['accuracy'].mean()*100:11.2f}% │ {c14['accuracy'].mean()*100:11.2f}% │ {(c14['accuracy'].mean()-c10['accuracy'].mean())*100:+7.2f}% │")
print("└─────────────────┴──────────────┴──────────────┴──────────┘")

# =================================================================
# 特徴量重要度分析
# =================================================================
print("\n" + "=" * 80)
print("🔝 14特徴量版の特徴量重要度 TOP10")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': FEATURES_14,
    'importance': results_standard["14特徴量"]['feature_importance']
}).sort_values('importance', ascending=False)

print("\n順位 | 特徴量                    | 重要度")
print("─────┼──────────────────────────┼────────")
for i, row in enumerate(importance_df.head(10).itertuples(), 1):
    print(f" {i:2d}  │ {row.feature:24s} │ {row.importance:6.2f}")

# 追加特徴量の重要度
added_features = ['delta_negative_rate', 'delta_against_rate', 
                  'sentiment_avg_score', 'stance_against_mean', 
                  'stance_favor_mean', 'stance_neutral_mean']
added_importance = importance_df[importance_df['feature'].isin(added_features)]

print("\n" + "=" * 80)
print("📌 追加4特徴量の分析")
print("=" * 80)
print("\n特徴量                    | 重要度 | 全体順位")
print("──────────────────────────┼────────┼──────")
for row in added_importance.itertuples():
    rank = importance_df.index.get_loc(row.Index) + 1
    print(f"{row.feature:24s} │ {row.importance:6.2f} │ {rank:2d}位")

total_added = added_importance['importance'].sum()
total_all = importance_df['importance'].sum()
print(f"\n追加4特徴量の重要度合計: {total_added:.2f} ({total_added/total_all*100:.1f}%)")

# =================================================================
# 最終結論
# =================================================================
print("\n" + "=" * 80)
print("💡 最終結論")
print("=" * 80)

std_improvement = (r14['f1'] - r10['f1']) * 100
cross_improvement = (c14['f1'].mean() - c10['f1'].mean()) * 100

print(f"\n📊 性能評価:")
print(f"  標準評価（同一トピック内）:")
print(f"    F1スコア: {r10['f1']*100:.2f}% → {r14['f1']*100:.2f}% ({std_improvement:+.2f}%)")
print(f"  クロストピック評価（未知トピック）:")
print(f"    F1スコア: {c10['f1'].mean()*100:.2f}% → {c14['f1'].mean()*100:.2f}% ({cross_improvement:+.2f}%)")

if std_improvement > 0.5 and cross_improvement >= -1.0:
    print(f"\n✅ 14特徴量版を推奨")
    print(f"   理由:")
    print(f"   - 標準評価で{std_improvement:+.2f}%向上")
    print(f"   - クロストピック評価で{cross_improvement:+.2f}%（許容範囲）")
    print(f"   - sentiment_avg_score, stance_mean系が有効")
    print(f"   - topicへの過学習を回避")
    recommendation = "14特徴量版"
elif std_improvement > 0 and cross_improvement < -2.0:
    print(f"\n⚠️  14特徴量版は効果限定的")
    print(f"   標準評価: {std_improvement:+.2f}%向上")
    print(f"   クロストピック: {cross_improvement:+.2f}%低下")
    print(f"   汎化性能とのトレードオフを検討")
    recommendation = "状況次第"
else:
    print(f"\n✓ 10特徴量版を推奨")
    print(f"   理由: 14特徴量版の改善が{std_improvement:.2f}%と小幅")
    print(f"   シンプルさを優先")
    recommendation = "10特徴量版"

print(f"\n🎯 推奨: {recommendation}")

if recommendation == "14特徴量版":
    print(f"\n📝 次のステップ:")
    print(f"  1. compare_all_models.pyを14特徴量版に更新")
    print(f"  2. 6モデル全てで再実験")
    print(f"  3. プレゼン資料を「14特徴量」に更新")
    print(f"  4. 重要特徴量TOP5を発表資料に反映")

print("\n" + "=" * 80)
