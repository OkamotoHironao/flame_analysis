#!/usr/bin/env python3
"""
感情分析手法の比較実験スクリプト

比較対象:
1. BERT: koheiduck/bert-japanese-finetuned-sentiment
2. 辞書ベース: PN辞書を使用
3. BERT + 辞書: 両方の特徴量を使用

Usage:
    python compare_sentiment_methods.py
    python compare_sentiment_methods.py --topics 松本人志,三苫,寿司ペロ
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import xgboost as xgb
import joblib

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent
FLAME_DIR = BASE_DIR / "modules" / "flame_detection"
OUTPUTS_DIR = FLAME_DIR / "outputs"


# ========================================
# 感情分析手法の比較用特徴量セット
# ========================================
SENTIMENT_FEATURE_SETS = {
    'bert_only': {
        'name': 'BERT のみ',
        'features': [
            'volume', 'delta_volume',
            'negative_rate', 'delta_negative_rate',
            'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
            'flame_score', 'against_count',
            'negative_rate_log', 'volume_log',
            'sentiment_polarity',
            'is_high_volume', 'is_high_negative', 'is_both_high',
        ],
        'description': 'BERTベースの感情分析のみを使用'
    },
    'dict_only': {
        'name': '辞書ベース のみ',
        'features': [
            'volume', 'delta_volume',
            'dict_negative_rate', 'delta_dict_negative_rate',
            'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
            'dict_flame_score', 'against_count',
            'dict_negative_rate_log', 'volume_log',
            'sentiment_polarity',
            'is_high_volume', 'is_dict_high_negative', 'is_both_dict_high',
        ],
        'description': 'PN辞書ベースの感情分析のみを使用'
    },
    'bert_and_dict': {
        'name': 'BERT + 辞書',
        'features': [
            'volume', 'delta_volume',
            # BERT
            'negative_rate', 'delta_negative_rate',
            'flame_score', 'negative_rate_log',
            'is_high_negative',
            # 辞書
            'dict_negative_rate', 'delta_dict_negative_rate',
            'dict_flame_score', 'dict_negative_rate_log',
            'is_dict_high_negative',
            # 共通
            'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
            'against_count', 'volume_log',
            'sentiment_polarity',
            'is_high_volume', 'is_both_high', 'is_both_dict_high',
        ],
        'description': 'BERTと辞書の両方を特徴量として使用'
    },
}

# ========================================
# 立場検出の有無の比較用特徴量セット
# ========================================
STANCE_FEATURE_SETS = {
    'with_stance': {
        'name': 'Stance あり',
        'features': [
            'volume', 'delta_volume',
            'negative_rate', 'delta_negative_rate',
            'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
            'flame_score', 'against_count',
            'negative_rate_log', 'volume_log',
            'sentiment_polarity',
            'is_high_volume', 'is_high_negative', 'is_both_high',
        ],
        'description': '感情分析 + 立場検出（賛成/反対/中立）を使用'
    },
    'without_stance': {
        'name': 'Stance なし',
        'features': [
            'volume', 'delta_volume',
            'negative_rate', 'delta_negative_rate',
            'flame_score',
            'negative_rate_log', 'volume_log',
            'is_high_volume', 'is_high_negative', 'is_both_high',
        ],
        'description': '感情分析のみを使用（立場検出なし）'
    },
}

# 後方互換性のため
FEATURE_SETS = SENTIMENT_FEATURE_SETS


def discover_topics():
    """利用可能なトピックを検出"""
    topics = []
    
    if OUTPUTS_DIR.exists():
        for topic_dir in OUTPUTS_DIR.iterdir():
            if topic_dir.is_dir() and topic_dir.name != 'unified_model_v2':
                labeled_csv = topic_dir / f"{topic_dir.name}_labeled.csv"
                if labeled_csv.exists():
                    topics.append(topic_dir.name)
    
    return sorted(topics)


def load_topic_data(topic_name):
    """トピックのラベル付きデータを読み込む"""
    csv_path = OUTPUTS_DIR / topic_name / f"{topic_name}_labeled.csv"
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    df['topic'] = topic_name
    
    return df


def add_dictionary_features(df, topic_name):
    """辞書ベースの特徴量を追加（事前計算が必要）"""
    # 辞書感情分析結果のパスを確認
    dict_sentiment_path = BASE_DIR / "data" / "processed" / f"{topic_name}_dict_sentiment_1h.csv"
    
    if dict_sentiment_path.exists():
        dict_df = pd.read_csv(dict_sentiment_path)
        
        # タイムスタンプでマージ
        df = df.merge(
            dict_df[['timestamp', 'dict_negative_rate', 'dict_positive_rate']],
            on='timestamp',
            how='left'
        )
    else:
        # 辞書データがない場合はダミー値
        df['dict_negative_rate'] = 0.0
        df['dict_positive_rate'] = 0.0
    
    return df


def add_composite_features(df, use_dict=False):
    """複合特徴量を追加"""
    df = df.copy()
    
    # BERT関連
    if 'negative_rate' in df.columns:
        df['flame_score'] = df['volume'] * df['negative_rate']
        df['negative_rate_log'] = np.log1p(df['negative_rate'] * 100)
        df['is_high_negative'] = (df['negative_rate'] >= 0.2).astype(int)
        df['is_both_high'] = ((df['volume'] >= 50) & (df['negative_rate'] >= 0.2)).astype(int)
    
    # 差分
    if 'negative_rate' in df.columns:
        df['delta_negative_rate'] = df['negative_rate'].diff().fillna(0)
    
    # 辞書関連
    if 'dict_negative_rate' in df.columns:
        df['dict_flame_score'] = df['volume'] * df['dict_negative_rate']
        df['dict_negative_rate_log'] = np.log1p(df['dict_negative_rate'] * 100)
        df['is_dict_high_negative'] = (df['dict_negative_rate'] >= 0.2).astype(int)
        df['is_both_dict_high'] = ((df['volume'] >= 50) & (df['dict_negative_rate'] >= 0.2)).astype(int)
        df['delta_dict_negative_rate'] = df['dict_negative_rate'].diff().fillna(0)
    
    # 共通
    df['against_count'] = df['volume'] * df.get('stance_against_rate', 0)
    df['volume_log'] = np.log1p(df['volume'])
    df['sentiment_polarity'] = df.get('stance_against_rate', 0) - df.get('stance_favor_rate', 0)
    df['is_high_volume'] = (df['volume'] >= 50).astype(int)
    df['delta_volume'] = df['volume'].diff().fillna(0)
    
    return df


def prepare_features(df, feature_columns):
    """特徴量を準備"""
    available_cols = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️ 欠損特徴量: {missing_cols[:5]}...")
    
    X = df[available_cols].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available_cols


def train_and_evaluate(X, y, method_name, n_splits=5):
    """モデルを学習して評価"""
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
    
    # クロスバリデーション
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    results = {
        'method': method_name,
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_auc_mean': cv_roc_auc.mean(),
        'cv_roc_auc_std': cv_roc_auc.std(),
        'n_features': X.shape[1],
        'n_samples': len(y),
    }
    
    return results


def run_comparison(topics=None, output_dir=None):
    """比較実験を実行"""
    print("=" * 70)
    print("🔬 感情分析手法の比較実験")
    print("=" * 70)
    
    # トピック検出
    available_topics = discover_topics()
    print(f"\n📂 利用可能なトピック: {available_topics}")
    
    if topics:
        target_topics = [t.strip() for t in topics.split(',')]
    else:
        target_topics = available_topics
    
    print(f"📊 使用トピック: {target_topics}")
    
    # データ統合
    print(f"\n{'='*60}")
    print("📥 データ読み込み・統合")
    print(f"{'='*60}")
    
    all_data = []
    for topic in target_topics:
        df = load_topic_data(topic)
        if df is not None:
            # 辞書特徴量を追加
            df = add_dictionary_features(df, topic)
            all_data.append(df)
            print(f"  ✓ {topic}: {len(df)}件")
    
    if not all_data:
        print("❌ データがありません")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 統合データ: {len(combined_df)}件")
    
    # 複合特徴量を追加
    combined_df = add_composite_features(combined_df, use_dict=True)
    
    # 正解ラベル
    y = combined_df['is_controversy'].values
    print(f"   炎上(1): {(y==1).sum()}件, 非炎上(0): {(y==0).sum()}件")
    
    # 比較実験
    print(f"\n{'='*60}")
    print("🔬 比較実験開始")
    print(f"{'='*60}")
    
    results = []
    
    for method_key, method_info in FEATURE_SETS.items():
        print(f"\n📌 {method_info['name']}")
        print(f"   {method_info['description']}")
        
        X, used_features = prepare_features(combined_df, method_info['features'])
        print(f"   使用特徴量: {len(used_features)}個")
        
        result = train_and_evaluate(X, y, method_info['name'])
        result['method_key'] = method_key
        result['features_used'] = used_features
        results.append(result)
        
        print(f"   Accuracy: {result['cv_accuracy_mean']*100:.1f}% (±{result['cv_accuracy_std']*100:.1f})")
        print(f"   F1 Score: {result['cv_f1_mean']*100:.1f}% (±{result['cv_f1_std']*100:.1f})")
        print(f"   ROC-AUC:  {result['cv_roc_auc_mean']*100:.1f}% (±{result['cv_roc_auc_std']*100:.1f})")
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("📊 比較結果サマリー")
    print(f"{'='*70}")
    
    print(f"\n{'手法':<20} {'Accuracy':<15} {'F1 Score':<15} {'ROC-AUC':<15}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['method']:<20} {r['cv_accuracy_mean']*100:>6.1f}% (±{r['cv_accuracy_std']*100:.1f})  "
              f"{r['cv_f1_mean']*100:>6.1f}% (±{r['cv_f1_std']*100:.1f})  "
              f"{r['cv_roc_auc_mean']*100:>6.1f}% (±{r['cv_roc_auc_std']*100:.1f})")
    
    # 最良手法
    best = max(results, key=lambda x: x['cv_f1_mean'])
    print(f"\n🏆 最良手法: {best['method']} (F1: {best['cv_f1_mean']*100:.1f}%)")
    
    # 結果を保存
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "comparison_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON保存
    output_json = output_dir / f"sentiment_comparison_{timestamp}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        # features_usedをシリアライズ可能に変換
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy['features_used'] = list(r_copy['features_used'])
            results_serializable.append(r_copy)
        
        json.dump({
            'timestamp': timestamp,
            'comparison_type': 'sentiment',
            'topics': target_topics,
            'n_samples': len(combined_df),
            'results': results_serializable,
            'best_method': best['method'],
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果保存: {output_json}")
    
    # CSV保存
    output_csv = output_dir / f"sentiment_comparison_{timestamp}.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"💾 CSV保存: {output_csv}")
    
    return results


def run_stance_comparison(topics=None, output_dir=None):
    """立場検出の有無の比較実験を実行"""
    print("=" * 70)
    print("🔬 立場検出の有無の比較実験")
    print("=" * 70)
    
    # トピック検出
    available_topics = discover_topics()
    print(f"\n📂 利用可能なトピック: {available_topics}")
    
    if topics:
        target_topics = [t.strip() for t in topics.split(',')]
    else:
        target_topics = available_topics
    
    print(f"📊 使用トピック: {target_topics}")
    
    # データ統合
    print(f"\n{'='*60}")
    print("📥 データ読み込み・統合")
    print(f"{'='*60}")
    
    all_data = []
    for topic in target_topics:
        df = load_topic_data(topic)
        if df is not None:
            all_data.append(df)
            print(f"  ✓ {topic}: {len(df)}件")
    
    if not all_data:
        print("❌ データがありません")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 統合データ: {len(combined_df)}件")
    
    # 複合特徴量を追加
    combined_df = add_composite_features(combined_df, use_dict=False)
    
    # 正解ラベル
    y = combined_df['is_controversy'].values
    print(f"   炎上(1): {(y==1).sum()}件, 非炎上(0): {(y==0).sum()}件")
    
    # 比較実験
    print(f"\n{'='*60}")
    print("🔬 比較実験開始")
    print(f"{'='*60}")
    
    results = []
    
    for method_key, method_info in STANCE_FEATURE_SETS.items():
        print(f"\n📌 {method_info['name']}")
        print(f"   {method_info['description']}")
        
        X, used_features = prepare_features(combined_df, method_info['features'])
        print(f"   使用特徴量: {len(used_features)}個")
        
        result = train_and_evaluate(X, y, method_info['name'])
        result['method_key'] = method_key
        result['features_used'] = used_features
        results.append(result)
        
        print(f"   Accuracy: {result['cv_accuracy_mean']*100:.1f}% (±{result['cv_accuracy_std']*100:.1f})")
        print(f"   F1 Score: {result['cv_f1_mean']*100:.1f}% (±{result['cv_f1_std']*100:.1f})")
        print(f"   ROC-AUC:  {result['cv_roc_auc_mean']*100:.1f}% (±{result['cv_roc_auc_std']*100:.1f})")
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("📊 比較結果サマリー")
    print(f"{'='*70}")
    
    print(f"\n{'手法':<20} {'Accuracy':<15} {'F1 Score':<15} {'ROC-AUC':<15}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['method']:<20} {r['cv_accuracy_mean']*100:>6.1f}% (±{r['cv_accuracy_std']*100:.1f})  "
              f"{r['cv_f1_mean']*100:>6.1f}% (±{r['cv_f1_std']*100:.1f})  "
              f"{r['cv_roc_auc_mean']*100:>6.1f}% (±{r['cv_roc_auc_std']*100:.1f})")
    
    # 差分を計算
    with_stance = next((r for r in results if r['method_key'] == 'with_stance'), None)
    without_stance = next((r for r in results if r['method_key'] == 'without_stance'), None)
    
    if with_stance and without_stance:
        diff_f1 = (with_stance['cv_f1_mean'] - without_stance['cv_f1_mean']) * 100
        diff_auc = (with_stance['cv_roc_auc_mean'] - without_stance['cv_roc_auc_mean']) * 100
        print(f"\n📈 立場検出の効果:")
        print(f"   F1 Score:  {'+' if diff_f1 >= 0 else ''}{diff_f1:.1f}%")
        print(f"   ROC-AUC:   {'+' if diff_auc >= 0 else ''}{diff_auc:.1f}%")
    
    # 最良手法
    best = max(results, key=lambda x: x['cv_f1_mean'])
    print(f"\n🏆 最良手法: {best['method']} (F1: {best['cv_f1_mean']*100:.1f}%)")
    
    # 結果を保存
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "comparison_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON保存
    output_json = output_dir / f"stance_comparison_{timestamp}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy['features_used'] = list(r_copy['features_used'])
            results_serializable.append(r_copy)
        
        json.dump({
            'timestamp': timestamp,
            'comparison_type': 'stance',
            'topics': target_topics,
            'n_samples': len(combined_df),
            'results': results_serializable,
            'best_method': best['method'],
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果保存: {output_json}")
    
    # CSV保存
    output_csv = output_dir / f"stance_comparison_{timestamp}.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"💾 CSV保存: {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='手法比較実験')
    parser.add_argument('--topics', type=str, default=None, help='使用するトピック（カンマ区切り）')
    parser.add_argument('--output', type=str, default=None, help='出力ディレクトリ')
    parser.add_argument('--type', type=str, default='sentiment', 
                        choices=['sentiment', 'stance', 'all'],
                        help='比較タイプ: sentiment(感情分析), stance(立場検出), all(両方)')
    
    args = parser.parse_args()
    
    if args.type == 'sentiment':
        run_comparison(topics=args.topics, output_dir=args.output)
    elif args.type == 'stance':
        run_stance_comparison(topics=args.topics, output_dir=args.output)
    elif args.type == 'all':
        run_comparison(topics=args.topics, output_dir=args.output)
        print("\n" + "="*70 + "\n")
        run_stance_comparison(topics=args.topics, output_dir=args.output)


if __name__ == '__main__':
    main()
