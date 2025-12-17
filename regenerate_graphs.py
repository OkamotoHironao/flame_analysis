#!/usr/bin/env python3
"""
プレゼンサイト用のグラフ画像を最新データで再生成
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_all_models_comparison():
    """全モデル性能比較グラフ"""
    print("📊 全モデル性能比較グラフを生成中...")
    
    # データ読み込み
    comparison_path = Path('outputs/unified_models_comparison/summary.json')
    with open(comparison_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データ整理
    models = []
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    
    for name, result in data['results'].items():
        models.append(name)
        metrics = result['metrics']
        f1_scores.append(metrics['f1'] * 100)
        accuracies.append(metrics['accuracy'] * 100)
        precisions.append(metrics['precision'] * 100)
        recalls.append(metrics['recall'] * 100)
    
    # F1スコアでソート
    sorted_indices = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('All Models Performance Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.6
    
    # F1 Score
    ax = axes[0, 0]
    bars = ax.bar(x, f1_scores, width, color='#1f77b4')
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(90, 100)
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 1]
    bars = ax.bar(x, accuracies, width, color='#ff7f0e')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(90, 100)
    for i, (bar, score) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Precision
    ax = axes[1, 0]
    bars = ax.bar(x, precisions, width, color='#2ca02c')
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(85, 102)
    for i, (bar, score) in enumerate(zip(bars, precisions)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Recall
    ax = axes[1, 1]
    bars = ax.bar(x, recalls, width, color='#d62728')
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(85, 102)
    for i, (bar, score) in enumerate(zip(bars, recalls)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = Path('outputs/all_models_comparison/all_models_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")


def generate_f1_ranking():
    """F1スコアランキンググラフ"""
    print("📊 F1スコアランキンググラフを生成中...")
    
    # データ読み込み
    comparison_path = Path('outputs/unified_models_comparison/summary.json')
    with open(comparison_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データ整理
    models = []
    f1_scores = []
    
    for name, result in data['results'].items():
        models.append(name)
        f1_scores.append(result['metrics']['f1'] * 100)
    
    # F1スコアでソート
    sorted_indices = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.barh(range(len(models)), f1_scores, color=colors[:len(models)])
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Ranking', fontsize=16, fontweight='bold')
    ax.set_xlim(92, 98)
    
    # 数値表示
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    # 保存
    output_path = Path('outputs/all_models_comparison/f1_ranking.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")


def generate_feature_importance():
    """特徴量重要度グラフ（6モデル平均）"""
    print("📊 特徴量重要度グラフを生成中...")
    
    # データ読み込み
    unified_dir = Path('outputs/unified_models_comparison')
    models = ['CatBoost', 'XGBoost', 'LightGBM', 'Random_Forest', 'Logistic_Regression', 'SVM_RBF']
    
    all_importances = defaultdict(list)
    
    for model_name in models:
        metadata_file = unified_dir / model_name / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            fi = metadata.get('feature_importance')
            if fi and isinstance(fi, dict):
                for feature, importance in fi.items():
                    all_importances[feature].append(importance)
    
    # 平均を計算
    averaged_importance = {}
    for feature, values in all_importances.items():
        averaged_importance[feature] = np.mean(values)
    
    # TOP10
    sorted_features = sorted(averaged_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#06A77D', '#004E89', '#FF6B35', '#9D4EDD', '#E63946', 
              '#06A77D', '#004E89', '#FF6B35', '#9D4EDD', '#E63946']
    
    bars = ax.barh(range(len(features)), importances, color=colors[:len(features)])
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Importance (Average of 6 Models)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Feature Importance', fontsize=16, fontweight='bold')
    
    # 数値表示
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    # 保存
    output_path = Path('outputs/all_models_comparison/feature_importance_top_model.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")


def main():
    print("=" * 70)
    print("🎨 プレゼンサイト用グラフ画像の再生成")
    print("=" * 70)
    print()
    
    # 各グラフを生成
    generate_all_models_comparison()
    generate_f1_ranking()
    generate_feature_importance()
    
    print()
    print("=" * 70)
    print("✅ すべてのグラフ画像を再生成しました！")
    print("=" * 70)
    print()
    print("生成されたファイル:")
    print("  - outputs/all_models_comparison/all_models_comparison.png")
    print("  - outputs/all_models_comparison/f1_ranking.png")
    print("  - outputs/all_models_comparison/feature_importance_top_model.png")
    print()
    print("プレゼンサイトをリフレッシュして確認してください。")


if __name__ == '__main__':
    main()
