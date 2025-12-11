#!/usr/bin/env python3
"""
全モデル比較スクリプト (CatBoost含む6モデル)
XGBoost, Random Forest, LightGBM, CatBoost, SVM, Logistic Regression
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
import json

# モデル
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # バックエンド設定

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_and_evaluate(X_train, y_train, X_test, y_test, model, model_name):
    """モデルを訓練して評価"""
    print(f"\n{'='*60}")
    print(f"🔧 {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 訓練
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  訓練時間:  {train_time:.2f}秒")
    
    # クロスバリデーション
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    print(f"  CV F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 特徴量重要度 (可能な場合)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    
    return {
        'model_name': model_name,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'train_time': train_time,
        'feature_importance': feature_importance,
        'y_pred': y_pred
    }


def visualize_comparison(results, feature_names, output_dir):
    """比較結果を可視化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # メトリクス比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('全モデル性能比較 (6モデル)', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        names = [r['model_name'] for r in results]
        values = [r['metrics'][metric] for r in results]
        
        bars = ax.bar(names, values, color=colors)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim(0.8, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ {output_dir / 'all_models_comparison.png'}")
    plt.close()
    
    # F1スコアランキング
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
    names = [r['model_name'] for r in sorted_results]
    f1_scores = [r['metrics']['f1'] for r in sorted_results]
    
    bars = ax.barh(range(len(names)), f1_scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('モデル別F1スコアランキング', fontsize=14, fontweight='bold')
    ax.set_xlim(0.8, 1.0)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
               f'{score:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')
        
        # 1位に王冠
        if i == 0:
            ax.text(0.81, bar.get_y() + bar.get_height()/2., '👑',
                   ha='left', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_ranking.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_dir / 'f1_ranking.png'}")
    plt.close()
    
    # 特徴量重要度 (トップモデル)
    top_model = sorted_results[0]
    if top_model['feature_importance'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': top_model['feature_importance']
        }).sort_values('importance', ascending=False).head(10)
        
        ax.barh(range(len(importance_df)), importance_df['importance'], color='#2ecc71')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('重要度', fontsize=12)
        ax.set_title(f'特徴量重要度 ({top_model["model_name"]} - 最高性能モデル)', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_top_model.png', dpi=300, bbox_inches='tight')
        print(f"✓ {output_dir / 'feature_importance_top_model.png'}")
        plt.close()


def main():
    """メイン処理"""
    print("🔥 全モデル比較実験 (6モデル)")
    print("="*60)
    
    # データ読み込み
    data_path = project_root / "outputs" / "unified_model_v2" / "combined_labeled.csv"
    
    if not data_path.exists():
        print(f"❌ データが見つかりません: {data_path}")
        return
    
    print(f"\n📂 データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    
    if 'is_controversy' in df.columns:
        df['label'] = df['is_controversy']
    
    print(f"✓ {len(df)}件のデータを読み込みました")
    print(f"  炎上: {(df['label']==1).sum()}件")
    print(f"  非炎上: {(df['label']==0).sum()}件")
    
    # 特徴量（16特徴量版）
    FEATURES = [
        # 既存10特徴量
        'volume', 'negative_rate', 'stance_against_rate',
        'stance_favor_rate', 'stance_neutral_rate',
        'delta_volume', 'delta_volume_rate',
        'flame_score', 'against_count', 'sentiment_polarity',
        # 追加6特徴量（汎化性能向上のため）
        'delta_negative_rate', 'delta_against_rate',
        'sentiment_avg_score',
        'stance_against_mean', 'stance_favor_mean', 'stance_neutral_mean'
    ]
    
    X = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['label']
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n訓練データ: {len(X_train)}件")
    print(f"テストデータ: {len(X_test)}件")
    
    # スケーリング (SVM用)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル定義
    models = {
        'LightGBM': (LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1
        ), False),  # スケーリング不要
        
        'XGBoost': (XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ), False),
        
        'CatBoost': (CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        ), False),
        
        'Random Forest': (RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ), False),
        
        'SVM (RBF)': (SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        ), True),  # スケーリング必要
        
        'Logistic Regression': (LogisticRegression(
            max_iter=1000,
            random_state=42
        ), True)
    }
    
    # 各モデルを評価
    results = []
    
    for model_name, (model, needs_scaling) in models.items():
        if needs_scaling:
            result = train_and_evaluate(
                X_train_scaled, y_train, X_test_scaled, y_test, 
                model, model_name
            )
        else:
            result = train_and_evaluate(
                X_train, y_train, X_test, y_test, 
                model, model_name
            )
        results.append(result)
    
    # 可視化
    output_dir = project_root / "outputs" / "all_models_comparison"
    visualize_comparison(results, FEATURES, output_dir)
    
    # JSON保存
    comparison_data = {}
    top_model = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)[0]
    
    for result in results:
        comparison_data[result['model_name']] = {
            'metrics': result['metrics'],
            'cv_f1': f"{result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}",
            'train_time': f"{result['train_time']:.2f}秒"
        }
    
    # 特徴量重要度を追加（トップモデルのみ）
    if top_model['feature_importance'] is not None:
        importance_data = []
        for feat, imp in zip(FEATURES, top_model['feature_importance']):
            importance_data.append({
                'feature': feat,
                'importance': float(imp)
            })
        # 重要度順にソート
        importance_data = sorted(importance_data, key=lambda x: x['importance'], reverse=True)
        comparison_data['_feature_importance'] = {
            'top_model': top_model['model_name'],
            'features': importance_data
        }
    
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {json_path}")
    
    # サマリー
    print(f"\n{'='*60}")
    print("📊 最終結果サマリー")
    print(f"{'='*60}")
    
    sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        rank_emoji = '🏆' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}️⃣'
        print(f"\n{rank_emoji} {result['model_name']}")
        print(f"  F1 Score:  {result['metrics']['f1']:.4f}")
        print(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"  CV F1:     {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}")
        print(f"  訓練時間:  {result['train_time']:.2f}秒")
    
    # 勝者発表
    winner = sorted_results[0]
    print(f"\n{'='*60}")
    print(f"🎉 最高性能モデル: {winner['model_name']}")
    print(f"   F1 Score: {winner['metrics']['f1']:.4f}")
    print(f"{'='*60}")
    
    print(f"\n📁 出力: {output_dir}")


if __name__ == "__main__":
    main()
