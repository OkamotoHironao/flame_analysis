#!/usr/bin/env python3
"""
全アルゴリズムで統合モデルを訓練

6つのアルゴリズムそれぞれで統合モデルを作成し、性能を比較
- CatBoost
- XGBoost
- LightGBM
- Random Forest
- SVM (RBF)
- Logistic Regression

Usage:
    python train_all_unified_models.py
"""

import sys
from pathlib import Path
import json
import time
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
from sklearn.inspection import permutation_importance
import joblib

# モデル
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 16特徴量（compare_all_models.pyと同じ）
FEATURES = [
    'volume',
    'negative_rate',
    'stance_against_rate',
    'stance_favor_rate',
    'stance_neutral_rate',
    'delta_volume',
    'delta_volume_rate',
    'flame_score',
    'against_count',
    'sentiment_polarity',
    'delta_negative_rate',
    'delta_against_rate',
    'sentiment_avg_score',
    'stance_against_mean',
    'stance_favor_mean',
    'stance_neutral_mean',
]


def load_unified_data():
    """統合データを読み込み"""
    data_path = Path("outputs/unified_model_v2/combined_labeled.csv")
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    print(f"✓ データ読み込み完了: {len(df)}件")
    print(f"  炎上: {(df['is_controversy'] == 1).sum()}件")
    print(f"  非炎上: {(df['is_controversy'] == 0).sum()}件")
    
    # 特徴量とラベル
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ 欠損特徴量: {missing_features}")
    
    X = df[available_features]
    y = df['is_controversy']
    
    print(f"✓ 使用特徴量: {len(available_features)}個")
    
    return X, y


def get_model_configs():
    """各モデルの設定を返す"""
    configs = {
        'CatBoost': {
            'model': CatBoostClassifier(
                iterations=100,
                depth=3,
                learning_rate=0.1,
                loss_function='Logloss',
                random_seed=42,
                verbose=False
            ),
            'needs_scaling': False
        },
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            'needs_scaling': False
        },
        'LightGBM': {
            'model': LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'needs_scaling': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'needs_scaling': False
        },
        'SVM (RBF)': {
            'model': SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'needs_scaling': True
        },
        'Logistic Regression': {
            'model': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'needs_scaling': True
        }
    }
    
    return configs


def find_optimal_threshold(model, X_test, y_test, scaler=None):
    """最適な閾値を探索"""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # F1スコア最大の閾値を探す
    best_threshold = 0.5
    best_f1 = 0
    
    for i, threshold in enumerate(thresholds):
        if i < len(recall_vals) - 1:
            r = recall_vals[i]
            p = precision_vals[i]
            
            if p > 0 and r > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    return best_threshold


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, needs_scaling=False):
    """モデルを訓練して評価"""
    print(f"\n{'='*70}")
    print(f"🔧 {model_name}")
    print(f"{'='*70}")
    
    scaler = None
    if needs_scaling:
        print("  📊 特徴量を標準化中...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # 訓練
    print(f"  🎓 訓練中...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"  ✓ 訓練完了 ({train_time:.2f}秒)")
    
    # 閾値最適化
    print(f"  🎯 最適閾値を探索中...")
    threshold = find_optimal_threshold(model, X_test, y_test, scaler if needs_scaling else None)
    print(f"  ✓ 最適閾値: {threshold:.4f}")
    
    # 予測
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 評価
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None
    }
    
    print(f"\n  📊 評価結果:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"    Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"    Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"    F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    if metrics['roc_auc']:
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # クロスバリデーション
    print(f"\n  🔄 5-Fold クロスバリデーション中...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {
        'accuracy': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy'),
        'f1': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    }
    
    print(f"    CV Accuracy: {cv_scores['accuracy'].mean():.4f} ± {cv_scores['accuracy'].std():.4f}")
    print(f"    CV F1:       {cv_scores['f1'].mean():.4f} ± {cv_scores['f1'].std():.4f}")
    print(f"    CV ROC-AUC:  {cv_scores['roc_auc'].mean():.4f} ± {cv_scores['roc_auc'].std():.4f}")
    
    # 特徴量重要度
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # Logistic Regressionの係数
        feature_importance = dict(zip(X_train.columns, np.abs(model.coef_[0])))
    else:
        # SVMなど、直接的な特徴量重要度がない場合はpermutation importanceを計算
        print(f"  🔍 Permutation Importance計算中...")
        perm_result = permutation_importance(
            model, X_test_scaled, y_test, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        feature_importance = dict(zip(X_train.columns, perm_result.importances_mean))
        print(f"  ✓ Permutation Importance計算完了")
    
    return {
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'metrics': metrics,
        'cv_scores': {
            'accuracy_mean': cv_scores['accuracy'].mean(),
            'accuracy_std': cv_scores['accuracy'].std(),
            'f1_mean': cv_scores['f1'].mean(),
            'f1_std': cv_scores['f1'].std(),
            'roc_auc_mean': cv_scores['roc_auc'].mean(),
            'roc_auc_std': cv_scores['roc_auc'].std()
        },
        'train_time': train_time,
        'feature_importance': feature_importance
    }


def save_model(model_name, result, X_train, output_dir):
    """モデルを保存"""
    model_dir = Path(output_dir) / model_name.replace(' ', '_').replace('(', '').replace(')', '')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル保存
    joblib.dump(result['model'], model_dir / 'model.pkl')
    
    # スケーラー保存（ある場合）
    if result['scaler'] is not None:
        joblib.dump(result['scaler'], model_dir / 'scaler.pkl')
    
    # メタデータ保存（float32をfloatに変換）
    def convert_to_python_types(obj):
        """numpy/pandas型をPython標準型に変換"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    metadata = {
        'model_name': model_name,
        'created_at': datetime.now().isoformat(),
        'threshold': float(result['threshold']),
        'features': list(X_train.columns),
        'metrics': {
            'test': convert_to_python_types(result['metrics']),
            'cv': convert_to_python_types(result['cv_scores'])
        },
        'train_time': float(result['train_time']),
        'feature_importance': convert_to_python_types(result['feature_importance']) if result['feature_importance'] else None
    }
    
    with open(model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ モデル保存: {model_dir}")


def main():
    print("=" * 70)
    print("🔥 全アルゴリズム統合モデル訓練")
    print("=" * 70)
    
    # データ読み込み
    print("\n📥 データ読み込み")
    X, y = load_unified_data()
    
    if X is None:
        return
    
    # データ分割
    print("\n✂️ データ分割")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  訓練: {len(X_train)}件, テスト: {len(X_test)}件")
    
    # モデル設定取得
    configs = get_model_configs()
    
    # 各モデルで訓練
    results = {}
    output_dir = Path("outputs/unified_models_comparison")
    
    for model_name, config in configs.items():
        result = train_and_evaluate_model(
            model_name,
            config['model'],
            X_train,
            y_train,
            X_test,
            y_test,
            needs_scaling=config['needs_scaling']
        )
        
        results[model_name] = result
        
        # モデル保存
        save_model(model_name, result, X_train, output_dir)
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("📊 全モデル性能サマリー")
    print(f"{'='*70}\n")
    
    # F1スコアでソート
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)
    
    print(f"{'順位':<6} {'モデル':<25} {'F1':<10} {'Accuracy':<10} {'訓練時間':<10}")
    print("-" * 70)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        icon = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}位"
        print(f"{icon:<6} {model_name:<25} {result['metrics']['f1']:.4f}    {result['metrics']['accuracy']:.4f}    {result['train_time']:.2f}秒")
    
    # 総合レポート保存
    def convert_to_python_types(obj):
        """numpy/pandas型をPython標準型に変換"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    summary = {
        'created_at': datetime.now().isoformat(),
        'total_models': len(results),
        'best_model': sorted_results[0][0],
        'best_f1': float(sorted_results[0][1]['metrics']['f1']),
        'results': {
            name: {
                'metrics': convert_to_python_types(res['metrics']),
                'cv_scores': convert_to_python_types(res['cv_scores']),
                'threshold': float(res['threshold']),
                'train_time': float(res['train_time'])
            }
            for name, res in results.items()
        }
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 総合レポート保存: {output_dir / 'summary.json'}")
    print(f"\n🎉 全モデル訓練完了！")


if __name__ == '__main__':
    main()
