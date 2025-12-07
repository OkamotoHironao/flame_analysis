#!/usr/bin/env python3
"""
機械学習モデル比較スクリプト
XGBoost, Random Forest, LightGBM, CatBoost, SVM, Logistic Regression を比較
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# モデル
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# パス設定
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"


def load_all_labeled_data():
    """全トピックのラベル付きデータを読み込み"""
    all_data = []
    
    for topic_dir in OUTPUTS_DIR.iterdir():
        if topic_dir.is_dir():
            labeled_csv = topic_dir / f"{topic_dir.name}_labeled.csv"
            if labeled_csv.exists():
                df = pd.read_csv(labeled_csv)
                df['topic'] = topic_dir.name
                all_data.append(df)
                print(f"  ✓ {topic_dir.name}: {len(df)}件")
    
    if not all_data:
        raise ValueError("ラベル付きデータが見つかりません")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  合計: {len(combined)}件")
    
    return combined


def prepare_features(df):
    """特徴量を準備"""
    feature_columns = [
        'volume', 'delta_volume', 'negative_rate', 'delta_negative_rate',
        'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
        'flame_score', 'against_count', 'sentiment_polarity'
    ]
    
    # 存在する特徴量のみ使用
    available_features = [col for col in feature_columns if col in df.columns]
    
    # 複合特徴量を計算（存在しない場合）
    if 'flame_score' not in df.columns and 'volume' in df.columns and 'negative_rate' in df.columns:
        df['flame_score'] = df['volume'] * df['negative_rate']
        available_features.append('flame_score')
    
    if 'against_count' not in df.columns and 'volume' in df.columns and 'stance_against_rate' in df.columns:
        df['against_count'] = df['volume'] * df['stance_against_rate']
        available_features.append('against_count')
    
    if 'sentiment_polarity' not in df.columns and 'stance_against_rate' in df.columns and 'stance_favor_rate' in df.columns:
        df['sentiment_polarity'] = df['stance_against_rate'] - df['stance_favor_rate']
        available_features.append('sentiment_polarity')
    
    # 重複を除去
    available_features = list(dict.fromkeys(available_features))
    
    X = df[available_features].fillna(0)
    y = df['is_controversy']
    
    return X, y, available_features


def compare_models(X, y):
    """モデルを比較"""
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # モデル定義
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    # 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    print("\n" + "=" * 70)
    print("📊 機械学習モデル比較（5-Fold Cross Validation）")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n🔄 {name} を評価中...")
        
        start_time = time.time()
        
        # クロスバリデーション
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        roc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        train_time = time.time() - start_time
        
        result = {
            'model': name,
            'accuracy_mean': acc_scores.mean(),
            'accuracy_std': acc_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'roc_auc_mean': roc_scores.mean(),
            'roc_auc_std': roc_scores.std(),
            'train_time': train_time
        }
        results.append(result)
        
        print(f"   Accuracy: {acc_scores.mean()*100:.2f}% (±{acc_scores.std()*100:.2f}%)")
        print(f"   F1 Score: {f1_scores.mean()*100:.2f}% (±{f1_scores.std()*100:.2f}%)")
        print(f"   ROC-AUC:  {roc_scores.mean()*100:.2f}% (±{roc_scores.std()*100:.2f}%)")
        print(f"   学習時間: {train_time:.2f}秒")
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("🔬 機械学習モデル比較実験")
    print("=" * 70)
    
    # データ読み込み
    print("\n📂 ラベル付きデータ読み込み中...")
    df = load_all_labeled_data()
    
    # 特徴量準備
    print("\n🔧 特徴量準備中...")
    X, y, features = prepare_features(df)
    print(f"   特徴量数: {len(features)}")
    print(f"   サンプル数: {len(y)} (炎上: {y.sum()}, 非炎上: {len(y) - y.sum()})")
    
    # モデル比較
    results_df = compare_models(X, y)
    
    # 結果表示
    print("\n" + "=" * 70)
    print("📊 比較結果サマリー")
    print("=" * 70)
    
    # ソート（F1スコア降順）
    results_df = results_df.sort_values('f1_mean', ascending=False)
    
    print("\n" + results_df.to_string(index=False))
    
    # ベストモデル
    best_model = results_df.iloc[0]
    print(f"\n🏆 ベストモデル: {best_model['model']}")
    print(f"   F1 Score: {best_model['f1_mean']*100:.2f}%")
    print(f"   ROC-AUC: {best_model['roc_auc_mean']*100:.2f}%")
    
    # 結果保存
    output_dir = OUTPUTS_DIR / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)
    print(f"\n💾 結果保存: {output_dir / 'comparison_results.csv'}")
    
    return results_df


if __name__ == "__main__":
    main()
