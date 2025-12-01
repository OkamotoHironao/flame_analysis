#!/usr/bin/env python3
"""
複数トピック統合学習スクリプト

複数のトピックからラベル付きデータを統合し、
汎化した炎上検知モデルを学習する

Usage:
    python train_unified_model.py
    python train_unified_model.py --topics 松本人志,三苫,寿司ペロ
    python train_unified_model.py --output outputs/unified_model/
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
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib

# 使用する特徴量（トピック非依存）
FEATURE_COLUMNS = [
    # 投稿量関連
    'volume',
    'delta_volume',
    'volume_ratio',
    
    # 感情関連（BERT）
    'positive_rate',
    'neutral_rate', 
    'negative_rate',
    'delta_positive_rate',
    'delta_neutral_rate',
    'delta_negative_rate',
    
    # 立場関連
    'stance_favor_rate',
    'stance_against_rate',
    'stance_neutral_rate',
    
    # 感情確率（平均）
    'avg_score',
]


def discover_topics(base_dir):
    """利用可能なトピック（ラベル付きデータがあるもの）を検出"""
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
    
    # トピック列を追加（分析用）
    df['topic'] = topic_name
    
    return df


def prepare_features(df, feature_columns):
    """特徴量を準備（欠損値処理など）"""
    # 必要な列のみ抽出
    available_cols = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️ 欠損特徴量: {missing_cols}")
    
    X = df[available_cols].copy()
    
    # 欠損値を0で埋める（または中央値など）
    X = X.fillna(0)
    
    # 無限大を置換
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available_cols


def train_model(X_train, y_train, X_test, y_test, class_weight=None):
    """XGBoostモデルを学習"""
    
    # クラス重みの計算
    if class_weight == 'balanced':
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    else:
        scale_pos_weight = 1.0
    
    print(f"  クラス重み (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # モデル定義
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 学習
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model


def evaluate_model(model, X_test, y_test, topic_info=None):
    """モデルを評価"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # ROC-AUCは両クラスが必要
    if len(np.unique(y_test)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics['roc_auc'] = None
    
    return metrics, y_pred, y_pred_proba


def cross_validate_model(X, y, n_splits=5):
    """クロスバリデーションで評価"""
    # クラス重み計算
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
    }
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='複数トピック統合学習スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python train_unified_model.py
  python train_unified_model.py --topics 松本人志,三苫
  python train_unified_model.py --output outputs/unified_model/
        """
    )
    
    parser.add_argument(
        '--topics',
        type=str,
        default=None,
        help='使用するトピック（カンマ区切り）。省略時は全て使用'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/unified_model/',
        help='出力ディレクトリ'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='テストデータの割合（デフォルト: 0.2）'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='クロスバリデーションの分割数（デフォルト: 5）'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔥 複数トピック統合学習")
    print("=" * 60)
    
    # トピック検出
    available_topics = discover_topics(base_dir)
    print(f"\n📂 利用可能なトピック: {available_topics}")
    
    if args.topics:
        target_topics = [t.strip() for t in args.topics.split(',')]
        # 存在チェック
        invalid = [t for t in target_topics if t not in available_topics]
        if invalid:
            print(f"❌ 存在しないトピック: {invalid}")
            sys.exit(1)
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
            
            topic_stats[topic] = {
                'total': n_total,
                'positive': n_pos,
                'negative': n_neg
            }
            
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ 有効なデータがありません")
        sys.exit(1)
    
    # 統合
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ 統合完了: {len(combined_df)}件")
    
    # 全体のラベル分布
    total_pos = (combined_df['is_controversy'] == 1).sum()
    total_neg = (combined_df['is_controversy'] == 0).sum()
    print(f"  炎上(1): {total_pos} ({total_pos/len(combined_df)*100:.1f}%)")
    print(f"  非炎上(0): {total_neg} ({total_neg/len(combined_df)*100:.1f}%)")
    
    if total_pos == 0 or total_neg == 0:
        print("❌ エラー: 両クラスのサンプルが必要です")
        print("   炎上(1)と非炎上(0)の両方を含むデータを使用してください")
        sys.exit(1)
    
    # 特徴量準備
    print(f"\n{'='*60}")
    print("🔧 特徴量準備")
    print("=" * 60)
    
    X, used_features = prepare_features(combined_df, FEATURE_COLUMNS)
    y = combined_df['is_controversy'].values
    
    print(f"  使用特徴量: {len(used_features)}個")
    for f in used_features:
        print(f"    - {f}")
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    # 評価
    print(f"\n{'='*60}")
    print("📈 モデル評価")
    print("=" * 60)
    
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    print(f"\n  テストデータ評価:")
    print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics['precision']*100:.2f}%")
    print(f"    Recall:    {metrics['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics['f1']*100:.2f}%")
    if metrics['roc_auc']:
        print(f"    ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  混同行列:")
    print(f"              予測:非炎上  予測:炎上")
    print(f"    実際:非炎上    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"    実際:炎上      {cm[1,0]:4d}      {cm[1,1]:4d}")
    
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
    
    # トピック別評価
    print(f"\n{'='*60}")
    print("📊 トピック別評価")
    print("=" * 60)
    
    # テストデータのインデックスを復元
    test_indices = combined_df.index[len(X_train):]
    
    for topic in target_topics:
        topic_mask = combined_df.loc[test_indices, 'topic'] == topic
        if topic_mask.sum() > 0:
            topic_y_test = y_test[topic_mask.values[:len(y_test)]]
            topic_y_pred = y_pred[topic_mask.values[:len(y_pred)]]
            
            if len(topic_y_test) > 0:
                acc = accuracy_score(topic_y_test, topic_y_pred)
                print(f"  {topic}: Accuracy {acc*100:.1f}% ({len(topic_y_test)}件)")
    
    # モデル保存
    print(f"\n{'='*60}")
    print("💾 モデル保存")
    print("=" * 60)
    
    # モデル
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ モデル: {model_path}")
    
    # スケーラー
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ スケーラー: {scaler_path}")
    
    # メタデータ
    metadata = {
        'created_at': datetime.now().isoformat(),
        'topics': target_topics,
        'topic_stats': topic_stats,
        'features': used_features,
        'metrics': {
            'test': metrics,
            'cv_accuracy_mean': float(cv_scores['accuracy'].mean()),
            'cv_accuracy_std': float(cv_scores['accuracy'].std()),
            'cv_f1_mean': float(cv_scores['f1'].mean()),
            'cv_f1_std': float(cv_scores['f1'].std()),
            'cv_roc_auc_mean': float(cv_scores['roc_auc'].mean()),
            'cv_roc_auc_std': float(cv_scores['roc_auc'].std()),
        },
        'feature_importance': {k: float(v) for k, v in sorted_importance},
        'model_params': model.get_params(),
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ メタデータ: {metadata_path}")
    
    # 統合データも保存
    combined_path = output_dir / "combined_labeled.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"  ✓ 統合データ: {combined_path}")
    
    print(f"\n{'='*60}")
    print("✅ 統合学習完了！")
    print("=" * 60)
    
    print(f"\n📊 最終結果:")
    print(f"  使用トピック: {', '.join(target_topics)}")
    print(f"  総サンプル数: {len(combined_df)}")
    print(f"  テスト Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  テスト F1 Score: {metrics['f1']*100:.2f}%")
    print(f"  CV Accuracy: {cv_scores['accuracy'].mean()*100:.2f}% (±{cv_scores['accuracy'].std()*100:.2f}%)")
    
    print(f"\n💡 次のステップ:")
    print(f"  1. 新しいトピックでモデルをテスト")
    print(f"  2. SHAP分析で予測理由を可視化")
    print(f"  3. 閾値調整でPrecision/Recallのバランス調整")


if __name__ == '__main__':
    main()
