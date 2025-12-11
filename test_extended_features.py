#!/usr/bin/env python3
"""
特徴量追加の検証スクリプト
現在の10特徴量 vs 拡張版（全特徴量使用）で性能比較
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
print("=" * 80)
print("特徴量追加検証実験")
print("=" * 80)

df = pd.read_csv('outputs/unified_model_v2/combined_labeled.csv')

# ラベル準備
if 'is_controversy' in df.columns:
    df['label'] = df['is_controversy']
else:
    df['label'] = df['is_flame']

print(f"\n📊 データセット情報")
print(f"  総サンプル数: {len(df)}件")
print(f"  炎上: {(df['label']==1).sum()}件")
print(f"  非炎上: {(df['label']==0).sum()}件")

# ===== 実験1: 現在の10特徴量 =====
print("\n" + "=" * 80)
print("実験1: 現在の10特徴量")
print("=" * 80)

FEATURES_CURRENT = [
    'volume', 'negative_rate', 'stance_against_rate',
    'stance_favor_rate', 'stance_neutral_rate',
    'delta_volume', 'delta_volume_rate',
    'flame_score', 'against_count', 'sentiment_polarity'
]

print(f"使用特徴量 ({len(FEATURES_CURRENT)}個):")
for i, f in enumerate(FEATURES_CURRENT, 1):
    print(f"  {i}. {f}")

X_current = df[FEATURES_CURRENT].fillna(0).replace([np.inf, -np.inf], 0)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_current, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost (現在)
print("\n🤖 CatBoost (現在の10特徴量)")
model_current = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
model_current.fit(X_train, y_train)
y_pred_current = model_current.predict(X_test)

acc_current = accuracy_score(y_test, y_pred_current)
prec_current = precision_score(y_test, y_pred_current)
rec_current = recall_score(y_test, y_pred_current)
f1_current = f1_score(y_test, y_pred_current)

print(f"  Accuracy:  {acc_current:.4f} ({acc_current*100:.2f}%)")
print(f"  Precision: {prec_current:.4f} ({prec_current*100:.2f}%)")
print(f"  Recall:    {rec_current:.4f} ({rec_current*100:.2f}%)")
print(f"  F1 Score:  {f1_current:.4f} ({f1_current*100:.2f}%)")

# CV評価
cv_scores_current = cross_val_score(model_current, X_current, y, cv=5, scoring='f1')
print(f"  CV F1:     {cv_scores_current.mean():.4f} ± {cv_scores_current.std():.4f}")

# ===== 実験2: 全特徴量（拡張版）=====
print("\n" + "=" * 80)
print("実験2: 拡張版（未使用特徴量を追加）")
print("=" * 80)

# 利用可能な全数値特徴量
FEATURES_EXTENDED = [
    # 現在使用中
    'volume', 'negative_rate', 'stance_against_rate',
    'stance_favor_rate', 'stance_neutral_rate',
    'delta_volume', 'delta_volume_rate',
    'flame_score', 'against_count', 'sentiment_polarity',
    # 追加
    'delta_negative_rate', 'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean'
]

# topicをカテゴリ変数として追加
df_extended = df.copy()
if 'topic' in df_extended.columns:
    le = LabelEncoder()
    df_extended['topic_encoded'] = le.fit_transform(df_extended['topic'].fillna('unknown'))
    FEATURES_EXTENDED.append('topic_encoded')

print(f"使用特徴量 ({len(FEATURES_EXTENDED)}個):")
for i, f in enumerate(FEATURES_EXTENDED, 1):
    print(f"  {i}. {f}")

X_extended = df_extended[FEATURES_EXTENDED].fillna(0).replace([np.inf, -np.inf], 0)

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_extended, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost (拡張版)
print("\n🤖 CatBoost (拡張版)")
model_extended = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
model_extended.fit(X_train_ext, y_train_ext)
y_pred_extended = model_extended.predict(X_test_ext)

acc_extended = accuracy_score(y_test_ext, y_pred_extended)
prec_extended = precision_score(y_test_ext, y_pred_extended)
rec_extended = recall_score(y_test_ext, y_pred_extended)
f1_extended = f1_score(y_test_ext, y_pred_extended)

print(f"  Accuracy:  {acc_extended:.4f} ({acc_extended*100:.2f}%)")
print(f"  Precision: {prec_extended:.4f} ({prec_extended*100:.2f}%)")
print(f"  Recall:    {rec_extended:.4f} ({rec_extended*100:.2f}%)")
print(f"  F1 Score:  {f1_extended:.4f} ({f1_extended*100:.2f}%)")

# CV評価
cv_scores_extended = cross_val_score(model_extended, X_extended, y, cv=5, scoring='f1')
print(f"  CV F1:     {cv_scores_extended.mean():.4f} ± {cv_scores_extended.std():.4f}")

# ===== 比較結果 =====
print("\n" + "=" * 80)
print("📊 比較結果サマリー")
print("=" * 80)

print("\n┌─────────────────┬──────────────┬──────────────┬──────────┐")
print("│ 指標            │ 現在(10特徴) │ 拡張(17特徴) │ 差分     │")
print("├─────────────────┼──────────────┼──────────────┼──────────┤")
print(f"│ Accuracy        │ {acc_current*100:11.2f}% │ {acc_extended*100:11.2f}% │ {(acc_extended-acc_current)*100:+7.2f}% │")
print(f"│ Precision       │ {prec_current*100:11.2f}% │ {prec_extended*100:11.2f}% │ {(prec_extended-prec_current)*100:+7.2f}% │")
print(f"│ Recall          │ {rec_current*100:11.2f}% │ {rec_extended*100:11.2f}% │ {(rec_extended-rec_current)*100:+7.2f}% │")
print(f"│ F1 Score        │ {f1_current*100:11.2f}% │ {f1_extended*100:11.2f}% │ {(f1_extended-f1_current)*100:+7.2f}% │")
print(f"│ CV F1 (平均)    │ {cv_scores_current.mean()*100:11.2f}% │ {cv_scores_extended.mean()*100:11.2f}% │ {(cv_scores_extended.mean()-cv_scores_current.mean())*100:+7.2f}% │")
print("└─────────────────┴──────────────┴──────────────┴──────────┘")

# 特徴量重要度の比較
print("\n" + "=" * 80)
print("🔝 拡張版の特徴量重要度 TOP10")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': FEATURES_EXTENDED,
    'importance': model_extended.feature_importances_
}).sort_values('importance', ascending=False)

print("\n順位 | 特徴量                    | 重要度")
print("─────┼──────────────────────────┼────────")
for i, row in enumerate(importance_df.head(10).itertuples(), 1):
    print(f" {i:2d}  │ {row.feature:24s} │ {row.importance:6.2f}")

# 追加した特徴量の重要度
print("\n" + "=" * 80)
print("📌 追加特徴量の重要度分析")
print("=" * 80)

added_features = [
    'delta_negative_rate', 'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean',
    'topic_encoded'
]

added_importance = importance_df[importance_df['feature'].isin(added_features)]
print("\n特徴量                    | 重要度 | 順位")
print("──────────────────────────┼────────┼──────")
for row in added_importance.itertuples():
    rank = importance_df[importance_df['feature'] == row.feature].index[0] + 1
    print(f"{row.feature:24s} │ {row.importance:6.2f} │ {rank:2d}位")

total_added_importance = added_importance['importance'].sum()
total_importance = importance_df['importance'].sum()
print(f"\n追加特徴量の重要度合計: {total_added_importance:.2f} ({total_added_importance/total_importance*100:.1f}%)")

# ===== 結論 =====
print("\n" + "=" * 80)
print("💡 結論と推奨")
print("=" * 80)

f1_improvement = (f1_extended - f1_current) * 100
cv_improvement = (cv_scores_extended.mean() - cv_scores_current.mean()) * 100

if f1_improvement > 0.5:
    print(f"\n✅ 特徴量追加を推奨")
    print(f"   理由: F1スコアが {f1_improvement:+.2f}% 向上")
    print(f"   CV F1も {cv_improvement:+.2f}% 改善")
elif f1_improvement > 0:
    print(f"\n⚠️  特徴量追加の効果は限定的")
    print(f"   F1スコア向上: わずか {f1_improvement:+.2f}%")
    print(f"   モデルの複雑さとのトレードオフを検討")
else:
    print(f"\n❌ 特徴量追加は非推奨")
    print(f"   理由: F1スコアが {f1_improvement:.2f}% 低下")
    print(f"   現在の10特徴量で十分")

# 追加特徴量の有用性
high_importance_added = added_importance[added_importance['importance'] > 2.0]
if len(high_importance_added) > 0:
    print(f"\n🌟 特に有用な追加特徴量:")
    for row in high_importance_added.itertuples():
        print(f"   - {row.feature}: {row.importance:.2f}")
else:
    print(f"\n⚠️  重要度2.0以上の追加特徴量なし（既存特徴量で十分カバー）")

print("\n" + "=" * 80)
