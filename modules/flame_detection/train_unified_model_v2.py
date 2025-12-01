#!/usr/bin/env python3
"""
複数トピック統合学習スクリプト v2 (改善版)

改善点:
1. 複合特徴量の追加（volume × negative_rate など）
2. 閾値調整でRecall向上
3. 特徴量の正規化オプション
4. SMOTE によるオーバーサンプリング（オプション）

Usage:
    python train_unified_model_v2.py
    python train_unified_model_v2.py --threshold 0.3
    python train_unified_model_v2.py --use-smote
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
import xgboost as xgb
import joblib

# 基本特徴量
BASE_FEATURE_COLUMNS = [
    'volume',
    'delta_volume',
    'negative_rate',
    'delta_negative_rate',
    'stance_favor_rate',
    'stance_against_rate',
    'stance_neutral_rate',
]


def discover_topics(base_dir):
    """利用可能なトピックを検出"""
    topics = []
    outputs_dir = Path(base_dir) / "outputs"
    
    if outputs_dir.exists():
        for topic_dir in outputs_dir.iterdir():
            if topic_dir.is_dir():
                labeled_csv = topic_dir / f"{topic_dir.name}_labeled.csv"
                if labeled_csv.exists():
                    topics.append(topic_dir.name)
    
    return sorted(topics)


def load_topic_data(topic_name, base_dir):
    """トピックのラベル付きデータを読み込む"""
    csv_path = Path(base_dir) / "outputs" / topic_name / f"{topic_name}_labeled.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️ ファイルなし: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df['topic'] = topic_name
    
    return df


def add_composite_features(df):
    """複合特徴量を追加（改善ポイント①）"""
    df = df.copy()
    
    # 1. 炎上スコア = volume × negative_rate（両方高いと炎上の可能性高）
    df['flame_score'] = df['volume'] * df['negative_rate']
    
    # 2. ネガティブ投稿の絶対数
    df['negative_count'] = df['volume'] * df['negative_rate']
    
    # 3. 批判的投稿の絶対数（AGAINST × volume）
    df['against_count'] = df['volume'] * df['stance_against_rate']
    
    # 4. ネガティブ率の対数変換（分布の正規化）
    df['negative_rate_log'] = np.log1p(df['negative_rate'] * 100)
    
    # 5. 投稿量の対数変換（外れ値の影響軽減）
    df['volume_log'] = np.log1p(df['volume'])
    
    # 6. 感情極性（ネガティブ率 - ポジティブ率の代わりに、stance使用）
    df['sentiment_polarity'] = df['stance_against_rate'] - df['stance_favor_rate']
    
    # 7. 投稿量が閾値以上かどうか（バイナリ特徴量）
    df['is_high_volume'] = (df['volume'] >= 50).astype(int)
    
    # 8. ネガティブ率が閾値以上かどうか
    df['is_high_negative'] = (df['negative_rate'] >= 0.2).astype(int)
    
    # 9. 両方高い場合のフラグ
    df['is_both_high'] = ((df['volume'] >= 50) & (df['negative_rate'] >= 0.2)).astype(int)
    
    return df


# 拡張特徴量リスト
EXTENDED_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    'flame_score',
    'negative_count',
    'against_count',
    'negative_rate_log',
    'volume_log',
    'sentiment_polarity',
    'is_high_volume',
    'is_high_negative',
    'is_both_high',
]


def prepare_features(df, feature_columns):
    """特徴量を準備"""
    available_cols = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️ 欠損特徴量: {missing_cols}")
    
    X = df[available_cols].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available_cols


def train_model(X_train, y_train, X_test, y_test, class_weight='balanced', params=None):
    """XGBoostモデルを学習"""
    
    # クラス重み計算
    if class_weight == 'balanced':
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    else:
        scale_pos_weight = 1.0
    
    print(f"  クラス重み (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # デフォルトパラメータ
    default_params = {
        'n_estimators': 100,
        'max_depth': 3,  # 4→3に変更（過学習防止）
        'learning_rate': 0.05,  # 0.1→0.05に変更
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'min_child_weight': 3,  # 追加（過学習防止）
        'subsample': 0.8,  # 追加
        'colsample_bytree': 0.8,  # 追加
    }
    
    if params:
        default_params.update(params)
    
    model = xgb.XGBClassifier(**default_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model


def find_optimal_threshold(model, X_test, y_test, target_recall=0.8):
    """最適な閾値を探索（改善ポイント②）"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # 目標Recall以上で最大Precisionの閾値を探す
    best_threshold = 0.5
    best_f1 = 0
    
    for i, threshold in enumerate(thresholds):
        if i < len(recall_vals) - 1:
            r = recall_vals[i]
            p = precision_vals[i]
            
            if r >= target_recall and p > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """モデルを評価"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if len(np.unique(y_test)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics['roc_auc'] = None
    
    return metrics, y_pred, y_pred_proba


def cross_validate_model(X, y, n_splits=5, threshold=0.5):
    """クロスバリデーション"""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
    }
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='複数トピック統合学習スクリプト v2 (改善版)',
    )
    
    parser.add_argument('--topics', type=str, default=None,
                        help='使用するトピック（カンマ区切り）')
    parser.add_argument('--output', type=str, default='outputs/unified_model_v2/',
                        help='出力ディレクトリ')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='テストデータの割合')
    parser.add_argument('--cv', type=int, default=5,
                        help='クロスバリデーションの分割数')
    parser.add_argument('--threshold', type=float, default=None,
                        help='分類閾値（省略時は自動探索）')
    parser.add_argument('--target-recall', type=float, default=0.8,
                        help='目標Recall（閾値自動探索時）')
    parser.add_argument('--use-extended-features', action='store_true', default=True,
                        help='拡張特徴量を使用')
    parser.add_argument('--use-smote', action='store_true',
                        help='SMOTEでオーバーサンプリング')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔥 複数トピック統合学習 v2 (改善版)")
    print("=" * 60)
    print("\n📝 改善点:")
    print("  1. 複合特徴量の追加 (flame_score, negative_count等)")
    print("  2. 閾値自動調整 (Recall向上)")
    print("  3. 正則化パラメータ調整 (過学習防止)")
    
    # トピック検出
    available_topics = discover_topics(base_dir)
    print(f"\n📂 利用可能なトピック: {available_topics}")
    
    if args.topics:
        target_topics = [t.strip() for t in args.topics.split(',')]
    else:
        target_topics = available_topics
    
    print(f"📊 使用トピック: {target_topics}")
    
    # データ統合
    print(f"\n{'='*60}")
    print("📥 データ読み込み・統合")
    print("=" * 60)
    
    all_dfs = []
    topic_stats = {}
    
    for topic in target_topics:
        print(f"\n  📁 {topic}")
        df = load_topic_data(topic, base_dir)
        
        if df is not None:
            n_total = len(df)
            n_pos = (df['is_controversy'] == 1).sum()
            n_neg = (df['is_controversy'] == 0).sum()
            
            print(f"    レコード数: {n_total}")
            print(f"    炎上(1): {n_pos} ({n_pos/n_total*100:.1f}%)")
            print(f"    非炎上(0): {n_neg} ({n_neg/n_total*100:.1f}%)")
            
            topic_stats[topic] = {'total': n_total, 'positive': n_pos, 'negative': n_neg}
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ 有効なデータがありません")
        sys.exit(1)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 複合特徴量を追加
    if args.use_extended_features:
        print(f"\n🔧 複合特徴量を追加中...")
        combined_df = add_composite_features(combined_df)
        feature_columns = EXTENDED_FEATURE_COLUMNS
    else:
        feature_columns = BASE_FEATURE_COLUMNS
    
    print(f"\n✓ 統合完了: {len(combined_df)}件")
    total_pos = (combined_df['is_controversy'] == 1).sum()
    total_neg = (combined_df['is_controversy'] == 0).sum()
    print(f"  炎上(1): {total_pos} ({total_pos/len(combined_df)*100:.1f}%)")
    print(f"  非炎上(0): {total_neg} ({total_neg/len(combined_df)*100:.1f}%)")
    
    # 特徴量準備
    print(f"\n{'='*60}")
    print("🔧 特徴量準備")
    print("=" * 60)
    
    X, used_features = prepare_features(combined_df, feature_columns)
    y = combined_df['is_controversy'].values
    
    print(f"  使用特徴量: {len(used_features)}個")
    for f in used_features:
        print(f"    - {f}")
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTEオーバーサンプリング
    if args.use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            print(f"\n📈 SMOTEでオーバーサンプリング中...")
            smote = SMOTE(random_state=42)
            X_scaled, y = smote.fit_resample(X_scaled, y)
            print(f"  リサンプリング後: {len(y)}件")
            print(f"  炎上(1): {(y==1).sum()}, 非炎上(0): {(y==0).sum()}")
        except ImportError:
            print("  ⚠️ imbalanced-learn未インストール。SMOTEをスキップ")
    
    # データ分割
    print(f"\n{'='*60}")
    print("📊 データ分割")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )
    
    print(f"  訓練データ: {len(X_train)}件")
    print(f"    炎上(1): {(y_train==1).sum()}")
    print(f"    非炎上(0): {(y_train==0).sum()}")
    print(f"  テストデータ: {len(X_test)}件")
    print(f"    炎上(1): {(y_test==1).sum()}")
    print(f"    非炎上(0): {(y_test==0).sum()}")
    
    # モデル学習
    print(f"\n{'='*60}")
    print("🤖 モデル学習")
    print("=" * 60)
    
    model = train_model(X_train, y_train, X_test, y_test, class_weight='balanced')
    
    # 閾値調整
    print(f"\n{'='*60}")
    print("🎯 閾値調整")
    print("=" * 60)
    
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  指定閾値を使用: {threshold}")
    else:
        threshold = find_optimal_threshold(model, X_test, y_test, target_recall=args.target_recall)
        print(f"  目標Recall: {args.target_recall*100:.0f}%")
        print(f"  最適閾値: {threshold:.3f}")
    
    # 評価（デフォルト閾値0.5）
    print(f"\n{'='*60}")
    print("📈 モデル評価")
    print("=" * 60)
    
    print(f"\n  【閾値 0.5 での評価】")
    metrics_05, y_pred_05, _ = evaluate_model(model, X_test, y_test, threshold=0.5)
    print(f"    Accuracy:  {metrics_05['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics_05['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_05['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_05['f1']*100:.2f}%")
    
    cm_05 = confusion_matrix(y_test, y_pred_05)
    print(f"\n    混同行列:")
    print(f"                予測:非炎上  予測:炎上")
    print(f"      実際:非炎上    {cm_05[0,0]:4d}      {cm_05[0,1]:4d}")
    print(f"      実際:炎上      {cm_05[1,0]:4d}      {cm_05[1,1]:4d}")
    
    # 評価（最適閾値）
    print(f"\n  【閾値 {threshold:.3f} での評価】")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, threshold=threshold)
    print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics['precision']*100:.2f}%")
    print(f"    Recall:    {metrics['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics['f1']*100:.2f}%")
    if metrics['roc_auc']:
        print(f"    ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    混同行列:")
    print(f"                予測:非炎上  予測:炎上")
    print(f"      実際:非炎上    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"      実際:炎上      {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # クロスバリデーション
    print(f"\n  {args.cv}-Fold クロスバリデーション:")
    cv_scores = cross_validate_model(X_scaled, y, n_splits=args.cv)
    
    print(f"    Accuracy: {cv_scores['accuracy'].mean()*100:.2f}% (±{cv_scores['accuracy'].std()*100:.2f}%)")
    print(f"    F1 Score: {cv_scores['f1'].mean()*100:.2f}% (±{cv_scores['f1'].std()*100:.2f}%)")
    print(f"    ROC-AUC:  {cv_scores['roc_auc'].mean()*100:.2f}% (±{cv_scores['roc_auc'].std()*100:.2f}%)")
    
    # 特徴量重要度
    print(f"\n{'='*60}")
    print("📊 特徴量重要度")
    print("=" * 60)
    
    feature_importance = dict(zip(used_features, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(sorted_importance, 1):
        bar = '█' * int(imp * 50)
        print(f"  {i:2d}. {feat:25s} {imp*100:5.2f}% {bar}")
    
    # 見逃しサンプルの分析
    print(f"\n{'='*60}")
    print("🔍 見逃しサンプル分析")
    print("=" * 60)
    
    missed_mask = (y_test == 1) & (y_pred == 0)
    if missed_mask.sum() > 0:
        print(f"  見逃し件数: {missed_mask.sum()}件")
        # 見逃したサンプルの特徴を表示
        X_test_df = pd.DataFrame(X_test, columns=used_features)
        missed_samples = X_test_df[missed_mask]
        print(f"  見逃しサンプルの特徴（正規化後）:")
        for col in ['volume', 'negative_rate', 'flame_score']:
            if col in missed_samples.columns:
                print(f"    {col}: {missed_samples[col].values}")
    else:
        print("  ✓ 見逃しなし！")
    
    # モデル保存
    print(f"\n{'='*60}")
    print("💾 モデル保存")
    print("=" * 60)
    
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ モデル: {model_path}")
    
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ スケーラー: {scaler_path}")
    
    metadata = {
        'version': 'v2',
        'created_at': datetime.now().isoformat(),
        'topics': target_topics,
        'topic_stats': topic_stats,
        'features': used_features,
        'threshold': threshold,
        'metrics': {
            'threshold_05': metrics_05,
            'threshold_optimal': metrics,
            'cv_accuracy_mean': float(cv_scores['accuracy'].mean()),
            'cv_accuracy_std': float(cv_scores['accuracy'].std()),
            'cv_f1_mean': float(cv_scores['f1'].mean()),
            'cv_f1_std': float(cv_scores['f1'].std()),
            'cv_roc_auc_mean': float(cv_scores['roc_auc'].mean()),
            'cv_roc_auc_std': float(cv_scores['roc_auc'].std()),
        },
        'feature_importance': {k: float(v) for k, v in sorted_importance},
        'improvements': [
            '複合特徴量 (flame_score, negative_count等)',
            '閾値自動調整',
            '正則化パラメータ調整',
        ]
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ メタデータ: {metadata_path}")
    
    combined_path = output_dir / "combined_labeled.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"  ✓ 統合データ: {combined_path}")
    
    print(f"\n{'='*60}")
    print("✅ 統合学習 v2 完了！")
    print("=" * 60)
    
    print(f"\n📊 最終結果 (閾値={threshold:.3f}):")
    print(f"  Recall:    {metrics_05['recall']*100:.2f}% → {metrics['recall']*100:.2f}%")
    print(f"  Precision: {metrics_05['precision']*100:.2f}% → {metrics['precision']*100:.2f}%")
    print(f"  F1 Score:  {metrics_05['f1']*100:.2f}% → {metrics['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
