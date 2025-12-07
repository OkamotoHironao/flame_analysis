#!/usr/bin/env python3
"""
エンゲージメント特徴量の効果検証
エンゲージメント指標を含む場合と含まない場合でモデル性能を比較
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'IPAGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

from modules.flame_detection.train_unified_model_v2 import (
    load_topic_data,
    add_composite_features,
    prepare_features,
    BASE_FEATURE_COLUMNS,
    EXTENDED_FEATURE_COLUMNS,
    ENGAGEMENT_FEATURE_COLUMNS
)


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_name, feature_cols):
    """
    モデルを訓練して評価
    
    Args:
        X_train, y_train: 訓練データ
        X_test, y_test: テストデータ
        feature_name: 特徴量セットの名前
        feature_cols: 特徴量の列名リスト
    
    Returns:
        dict: 評価結果
    """
    print(f"\n{'='*60}")
    print(f"🔧 {feature_name} のモデル訓練")
    print(f"{'='*60}")
    print(f"  特徴量数: {len(feature_cols)}")
    print(f"  訓練サンプル数: {len(X_train)}")
    print(f"  テストサンプル数: {len(X_test)}")
    
    # LightGBMモデル
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 評価結果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # クロスバリデーション
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    print(f"\n🔄 5-Fold Cross Validation:")
    print(f"  F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 特徴量重要度 (Top 10):")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:6.4f}")
    
    return {
        'model': model,
        'feature_name': feature_name,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_pred': y_pred,
        'y_test': y_test
    }


def visualize_comparison(results_list, output_dir):
    """
    比較結果を可視化
    
    Args:
        results_list: 評価結果のリスト
        output_dir: 出力ディレクトリ
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. メトリクス比較（棒グラフ）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('エンゲージメント特徴量の効果比較', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        names = [r['feature_name'] for r in results_list]
        values = [r['metrics'][metric] for r in results_list]
        
        bars = ax.bar(names, values, color=['#3498db', '#e74c3c'])
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ メトリクス比較グラフを保存: {output_dir / 'engagement_metrics_comparison.png'}")
    plt.close()
    
    # 2. 特徴量重要度比較（エンゲージメント有りのみ）
    engagement_result = next(r for r in results_list if 'エンゲージメント有り' in r['feature_name'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = engagement_result['feature_importance'].head(15)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='#2ecc71')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('重要度', fontsize=12)
    ax.set_title('特徴量重要度 (エンゲージメント有り)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ 特徴量重要度グラフを保存: {output_dir / 'engagement_feature_importance.png'}")
    plt.close()
    
    # 3. 混同行列比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('混同行列の比較', fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results_list):
        ax = axes[idx]
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={'label': 'サンプル数'})
        ax.set_title(result['feature_name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('予測ラベル', fontsize=10)
        ax.set_ylabel('真のラベル', fontsize=10)
        ax.set_xticklabels(['非炎上', '炎上'])
        ax.set_yticklabels(['非炎上', '炎上'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ 混同行列グラフを保存: {output_dir / 'engagement_confusion_matrix.png'}")
    plt.close()


def main():
    """メイン処理"""
    print("🔥 エンゲージメント特徴量の効果検証")
    print("="*60)
    
    # データ読み込み
    data_dir = project_root / "data" / "processed"
    
    # 全トピックのデータを読み込み
    topics = [
        "WBC", "エアライダー", "フワちゃん", "マリオカートワールド",
        "みそきん", "広陵", "三苫", "寿司ペロ", "松本人志", "大谷翔平_mvp_2021"
    ]
    
    print(f"\n📂 データ読み込み中...")
    all_data = []
    for topic in topics:
        feature_path = data_dir / f"{topic}_feature_table.csv"
        label_path = data_dir / f"{topic}_labeled.csv"
        
        if not feature_path.exists() or not label_path.exists():
            print(f"  ⚠️ {topic}: データが見つかりません（スキップ）")
            continue
        
        df = load_topic_data(str(feature_path), str(label_path), topic)
        all_data.append(df)
        print(f"  ✓ {topic}: {len(df)}件")
    
    if not all_data:
        print("❌ データが見つかりません")
        return
    
    # 全データを結合
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ 合計 {len(df_all)}件のデータを読み込みました")
    print(f"  炎上: {(df_all['label']==1).sum()}件")
    print(f"  非炎上: {(df_all['label']==0).sum()}件")
    
    # 複合特徴量を追加
    df_all = add_composite_features(df_all)
    
    # 訓練/テストデータ分割
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=df_all['label']
    )
    
    y_train = train_df['label']
    y_test = test_df['label']
    
    # 比較実験
    results = []
    
    # 1. エンゲージメント無し
    print(f"\n{'='*60}")
    print("実験1: エンゲージメント特徴量なし")
    print(f"{'='*60}")
    
    X_train_base, cols_base = prepare_features(train_df, EXTENDED_FEATURE_COLUMNS)
    X_test_base, _ = prepare_features(test_df, EXTENDED_FEATURE_COLUMNS)
    
    result_base = train_and_evaluate(
        X_train_base, y_train, X_test_base, y_test,
        "エンゲージメント無し (10特徴量)", cols_base
    )
    results.append(result_base)
    
    # 2. エンゲージメント有り
    print(f"\n{'='*60}")
    print("実験2: エンゲージメント特徴量あり")
    print(f"{'='*60}")
    
    X_train_eng, cols_eng = prepare_features(train_df, ENGAGEMENT_FEATURE_COLUMNS)
    X_test_eng, _ = prepare_features(test_df, ENGAGEMENT_FEATURE_COLUMNS)
    
    result_eng = train_and_evaluate(
        X_train_eng, y_train, X_test_eng, y_test,
        "エンゲージメント有り (15特徴量)", cols_eng
    )
    results.append(result_eng)
    
    # 結果の可視化
    output_dir = project_root / "outputs" / "engagement_comparison"
    visualize_comparison(results, output_dir)
    
    # 比較結果をJSON保存
    comparison_data = {
        'without_engagement': {
            'feature_count': len(result_base['feature_cols']),
            'features': result_base['feature_cols'],
            'metrics': result_base['metrics'],
            'cv_f1_mean': float(result_base['cv_f1_mean']),
            'cv_f1_std': float(result_base['cv_f1_std'])
        },
        'with_engagement': {
            'feature_count': len(result_eng['feature_cols']),
            'features': result_eng['feature_cols'],
            'metrics': result_eng['metrics'],
            'cv_f1_mean': float(result_eng['cv_f1_mean']),
            'cv_f1_std': float(result_eng['cv_f1_std'])
        },
        'improvement': {
            'f1_delta': result_eng['metrics']['f1'] - result_base['metrics']['f1'],
            'f1_improvement_pct': (result_eng['metrics']['f1'] - result_base['metrics']['f1']) / result_base['metrics']['f1'] * 100,
            'accuracy_delta': result_eng['metrics']['accuracy'] - result_base['metrics']['accuracy']
        }
    }
    
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 比較結果をJSON保存: {json_path}")
    
    # サマリー表示
    print(f"\n{'='*60}")
    print("📊 最終結果サマリー")
    print(f"{'='*60}")
    print(f"\n1️⃣  エンゲージメント無し:")
    print(f"  F1 Score: {result_base['metrics']['f1']:.4f}")
    print(f"  CV F1:    {result_base['cv_f1_mean']:.4f} ± {result_base['cv_f1_std']:.4f}")
    
    print(f"\n2️⃣  エンゲージメント有り:")
    print(f"  F1 Score: {result_eng['metrics']['f1']:.4f}")
    print(f"  CV F1:    {result_eng['cv_f1_mean']:.4f} ± {result_eng['cv_f1_std']:.4f}")
    
    f1_improvement = result_eng['metrics']['f1'] - result_base['metrics']['f1']
    f1_improvement_pct = f1_improvement / result_base['metrics']['f1'] * 100
    
    print(f"\n🎯 改善度:")
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement_pct:+.2f}%)")
    
    if f1_improvement > 0:
        print(f"\n✅ エンゲージメント特徴量により性能が向上しました！")
    elif f1_improvement < 0:
        print(f"\n⚠️ エンゲージメント特徴量による性能向上は見られませんでした。")
    else:
        print(f"\n➡️ エンゲージメント特徴量による性能変化はありませんでした。")
    
    print(f"\n📁 出力ディレクトリ: {output_dir}")
    print(f"  - engagement_metrics_comparison.png")
    print(f"  - engagement_feature_importance.png")
    print(f"  - engagement_confusion_matrix.png")
    print(f"  - comparison_results.json")


if __name__ == "__main__":
    main()
