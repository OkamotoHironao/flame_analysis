#!/usr/bin/env python3
"""
エンゲージメント特徴量の効果検証（シンプル版）
既存データ(エンゲージメント無し)を使用して検証
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'IPAGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_name, feature_cols):
    """モデルを訓練して評価"""
    print(f"\n{'='*60}")
    print(f"🔧 {feature_name}")
    print(f"{'='*60}")
    print(f"  特徴量数: {len(feature_cols)}")
    print(f"  訓練: {len(X_train)}件, テスト: {len(X_test)}件")
    
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
    y_pred = model.predict(X_test)
    
    # 評価
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
    
    print(f"\n🔄 5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 特徴量重要度 (Top 10):")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']*100:5.1f}%")
    
    return {
        'model': model,
        'feature_name': feature_name,
        'feature_cols': feature_cols,
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_pred': y_pred,
        'y_test': y_test
    }


def visualize_comparison(results_list, output_dir):
    """比較結果を可視化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # メトリクス比較
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
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'engagement_metrics_comparison.png'}")
    plt.close()
    
    # 特徴量重要度
    engagement_result = next((r for r in results_list if 'エンゲージメント有り' in r['feature_name']), results_list[0])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = engagement_result['feature_importance'].head(15)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='#2ecc71')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('重要度', fontsize=12)
    ax.set_title(f'特徴量重要度 ({engagement_result["feature_name"]})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'feature_importance.png'}")
    plt.close()


def main():
    """メイン処理"""
    print("🔥 エンゲージメント特徴量の効果検証")
    print("="*60)
    
    # 既存の統一モデルデータを読み込み
    data_path = project_root / "outputs" / "unified_model_v2" / "combined_labeled.csv"
    
    if not data_path.exists():
        print(f"❌ データが見つかりません: {data_path}")
        return
    
    print(f"\n📂 データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    
    # is_controversyをlabelに変換
    if 'is_controversy' in df.columns:
        df['label'] = df['is_controversy']
    
    print(f"✓ {len(df)}件のデータを読み込みました")
    print(f"  炎上: {(df['label']==1).sum()}件")
    print(f"  非炎上: {(df['label']==0).sum()}件")
    
    # 訓練/テスト分割
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    y_train = train_df['label']
    y_test = test_df['label']
    
    results = []
    
    # ===============================================
    # 実験1: 基本特徴量のみ (7個)
    # ===============================================
    BASE_FEATURES = [
        'volume', 'negative_rate', 'stance_against_rate',
        'stance_favor_rate', 'stance_neutral_rate',
        'delta_volume', 'delta_volume_rate'
    ]
    
    X_train_base = train_df[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_base = test_df[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    
    result_base = train_and_evaluate(
        X_train_base, y_train, X_test_base, y_test,
        "基本特徴量 (7個)", BASE_FEATURES
    )
    results.append(result_base)
    
    # ===============================================
    # 実験2: 複合特徴量追加 (10個)
    # ===============================================
    EXTENDED_FEATURES = BASE_FEATURES + [
        'flame_score', 'against_count', 'sentiment_polarity'
    ]
    
    X_train_ext = train_df[EXTENDED_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_ext = test_df[EXTENDED_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    
    result_ext = train_and_evaluate(
        X_train_ext, y_train, X_test_ext, y_test,
        "複合特徴量追加 (10個)", EXTENDED_FEATURES
    )
    results.append(result_ext)
    
    # ===============================================
    # 実験3: 仮想エンゲージメント特徴量 (15個)
    # ===============================================
    # 注: 既存データにエンゲージメントデータがないため、
    # flame_scoreから推定した仮想値を使用
    print(f"\n{'='*60}")
    print("⚠️  注: エンゲージメントデータが欠損のため、仮想データで補完")
    print(f"{'='*60}")
    
    df_virtual = df.copy()
    # 仮想エンゲージメント = flame_score を基準にランダム生成
    np.random.seed(42)
    df_virtual['avg_engagement'] = df['flame_score'] * np.random.uniform(0.5, 1.5, len(df))
    df_virtual['total_engagement'] = df_virtual['avg_engagement'] * df['volume']
    df_virtual['engagement_rate'] = df_virtual['total_engagement'] / (df['volume'] + 1)
    df_virtual['flame_engagement_score'] = df_virtual['total_engagement'] * df['negative_rate']
    df_virtual['against_engagement_score'] = df_virtual['total_engagement'] * df['stance_against_rate']
    
    train_virtual, test_virtual = train_test_split(df_virtual, test_size=0.2, random_state=42, stratify=df_virtual['label'])
    
    ENGAGEMENT_FEATURES = EXTENDED_FEATURES + [
        'avg_engagement', 'total_engagement', 'engagement_rate',
        'flame_engagement_score', 'against_engagement_score'
    ]
    
    X_train_eng = train_virtual[ENGAGEMENT_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_eng = test_virtual[ENGAGEMENT_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    y_train_eng = train_virtual['label']
    y_test_eng = test_virtual['label']
    
    result_eng = train_and_evaluate(
        X_train_eng, y_train_eng, X_test_eng, y_test_eng,
        "エンゲージメント有り (15個・仮想)", ENGAGEMENT_FEATURES
    )
    results.append(result_eng)
    
    # 可視化
    output_dir = project_root / "outputs" / "engagement_comparison"
    visualize_comparison(results, output_dir)
    
    # JSON保存
    comparison_data = {
        'baseline': {
            'feature_count': len(result_base['feature_cols']),
            'metrics': result_base['metrics'],
            'cv_f1': f"{result_base['cv_f1_mean']:.4f} ± {result_base['cv_f1_std']:.4f}"
        },
        'extended': {
            'feature_count': len(result_ext['feature_cols']),
            'metrics': result_ext['metrics'],
            'cv_f1': f"{result_ext['cv_f1_mean']:.4f} ± {result_ext['cv_f1_std']:.4f}"
        },
        'with_engagement_virtual': {
            'feature_count': len(result_eng['feature_cols']),
            'metrics': result_eng['metrics'],
            'cv_f1': f"{result_eng['cv_f1_mean']:.4f} ± {result_eng['cv_f1_std']:.4f}",
            'note': '仮想エンゲージメントデータを使用'
        }
    }
    
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {json_path}")
    
    # サマリー
    print(f"\n{'='*60}")
    print("📊 最終結果サマリー")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}️⃣  {result['feature_name']}")
        print(f"  F1 Score: {result['metrics']['f1']:.4f}")
        print(f"  CV F1:    {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}")
    
    improvement = result_eng['metrics']['f1'] - result_ext['metrics']['f1']
    improvement_pct = improvement / result_ext['metrics']['f1'] * 100
    
    print(f"\n🎯 改善度 (複合→エンゲージメント):")
    print(f"  F1 Score: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    print(f"\n📁 出力: {output_dir}")


if __name__ == "__main__":
    main()
