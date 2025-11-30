#!/usr/bin/env python3
"""
炎上検知分類モデル学習スクリプト

ラベル付き特徴量データから is_controversy を予測する2値分類モデルを学習します。
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(csv_path: str) -> pd.DataFrame:
    """
    ラベル付き特徴量CSVを読み込み
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        pd.DataFrame: 読み込んだデータ
    """
    print(f"📖 データ読み込み中: {csv_path}")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {csv_path}")
    
    df = pd.read_csv(csv_path, comment='#')
    print(f"✓ {len(df)}件のレコードを読み込みました")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    特徴量とラベルを準備
    
    Args:
        df: 元のデータフレーム
        
    Returns:
        X: 特徴量データフレーム
        y: ラベルシリーズ
        feature_names: 特徴量名のリスト
    """
    print("\n🔧 特徴量準備中...")
    
    # ラベル列の確認
    if 'is_controversy' not in df.columns:
        raise ValueError("'is_controversy' 列が見つかりません")
    
    # 除外する列（timestamp, ラベル列など）
    exclude_cols = ['timestamp', 'is_controversy', 'created_at', 'date', 'datetime']
    
    # 数値列のみを特徴量として使用
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ラベル列を除外
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("使用可能な数値特徴量が見つかりません")
    
    print(f"✓ 使用する特徴量（{len(feature_cols)}個）:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    X = df[feature_cols].copy()
    y = df['is_controversy'].copy()
    
    # 欠損値の確認
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n⚠️  欠損値を検出: {missing_count}件")
        print("  → 0で埋めます")
        X = X.fillna(0)
    
    # クラス分布の確認
    print(f"\n📊 クラス分布:")
    print(f"  非炎上 (0): {(y == 0).sum()}件 ({(y == 0).mean() * 100:.1f}%)")
    print(f"  炎上 (1): {(y == 1).sum()}件 ({(y == 1).mean() * 100:.1f}%)")
    
    return X, y, feature_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'xgboost'
) -> Any:
    """
    分類モデルを学習
    
    Args:
        X_train: 訓練用特徴量
        y_train: 訓練用ラベル
        model_type: モデルタイプ ('xgboost' or 'randomforest')
        
    Returns:
        学習済みモデル
    """
    print(f"\n🤖 モデル学習中（{model_type}）...")
    
    # クラス不均衡対策: scale_pos_weight を計算
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"  クラス重み調整: scale_pos_weight={scale_pos_weight:.2f}")
    
    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == 'randomforest':
        # RandomForestの場合はclass_weightで調整
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")
    
    model.fit(X_train, y_train)
    print("✓ 学習完了")
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> Dict[str, float]:
    """
    モデルを評価
    
    Args:
        model: 学習済みモデル
        X_test: テスト用特徴量
        y_test: テスト用ラベル
        output_dir: 出力ディレクトリ
        
    Returns:
        評価指標の辞書
    """
    print("\n📊 モデル評価中...")
    
    # 予測
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 評価指標の計算
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    # 結果表示
    print("\n" + "="*60)
    print("📈 評価結果")
    print("="*60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print("="*60)
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 混同行列:")
    print(cm)
    print(f"\n  True Negative (正しく非炎上と予測): {cm[0, 0]}")
    print(f"  False Positive (誤って炎上と予測): {cm[0, 1]}")
    print(f"  False Negative (誤って非炎上と予測): {cm[1, 0]}")
    print(f"  True Positive (正しく炎上と予測): {cm[1, 1]}")
    
    # 詳細レポート
    print("\n📋 分類レポート:")
    print(classification_report(
        y_test, y_pred,
        target_names=['非炎上', '炎上'],
        zero_division=0
    ))
    
    # 評価結果をテキストファイルに保存
    eval_path = output_dir / 'evaluation.txt'
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("炎上検知モデル 評価結果\n")
        f.write("="*60 + "\n\n")
        
        f.write("【評価指標】\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"  AUC:       {metrics['auc']:.4f}\n\n")
        
        f.write("【混同行列】\n")
        f.write(f"  True Negative:  {cm[0, 0]}\n")
        f.write(f"  False Positive: {cm[0, 1]}\n")
        f.write(f"  False Negative: {cm[1, 0]}\n")
        f.write(f"  True Positive:  {cm[1, 1]}\n\n")
        
        f.write("【分類レポート】\n")
        f.write(classification_report(
            y_test, y_pred,
            target_names=['非炎上', '炎上'],
            zero_division=0
        ))
    
    print(f"\n✓ 評価結果を保存: {eval_path}")
    
    # 混同行列とROC曲線を可視化
    visualize_evaluation(y_test, y_pred, y_pred_proba, cm, output_dir)
    
    return metrics


def visualize_evaluation(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    cm: np.ndarray,
    output_dir: Path
):
    """
    評価結果を可視化
    
    Args:
        y_test: テストラベル
        y_pred: 予測ラベル
        y_pred_proba: 予測確率
        cm: 混同行列
        output_dir: 出力ディレクトリ
    """
    print("\n🎨 評価結果を可視化中...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 混同行列
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Controversy', 'Controversy'],
        yticklabels=['Non-Controversy', 'Controversy'],
        ax=ax1
    )
    ax1.set_title('Confusion Matrix', fontsize=14, pad=10)
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # 2. ROC曲線
    ax2 = axes[1]
    if len(np.unique(y_test)) > 1:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate', fontsize=11)
        ax2.set_ylabel('True Positive Rate', fontsize=11)
        ax2.set_title('ROC Curve', fontsize=14, pad=10)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'AUC計算不可\n(クラスが1種類のみ)',
                ha='center', va='center', fontsize=12)
        ax2.set_title('ROC Curve', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    eval_fig_path = output_dir / 'evaluation_metrics.png'
    plt.savefig(eval_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ 評価グラフを保存: {eval_fig_path}")
    plt.close()


def visualize_feature_importance(
    model: Any,
    feature_names: list,
    output_dir: Path,
    top_n: int = 20
):
    """
    特徴量重要度を可視化
    
    Args:
        model: 学習済みモデル
        feature_names: 特徴量名のリスト
        output_dir: 出力ディレクトリ
        top_n: 表示する上位N個の特徴量
    """
    print("\n📊 特徴量重要度を可視化中...")
    
    # 特徴量重要度を取得
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("⚠️  このモデルは特徴量重要度をサポートしていません")
        return
    
    # DataFrameに変換してソート
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 上位N個を表示
    print(f"\n🏆 特徴量重要度 Top {min(top_n, len(importance_df))}:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 可視化
    plt.figure(figsize=(10, max(6, len(importance_df.head(top_n)) * 0.4)))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance (Top {len(top_features)})', fontsize=14, pad=15)
    plt.gca().invert_yaxis()  # 上位を上に表示
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    importance_path = output_dir / 'feature_importance.png'
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    print(f"✓ 特徴量重要度グラフを保存: {importance_path}")
    plt.close()
    
    # CSV保存
    importance_csv = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_csv, index=False, encoding='utf-8')
    print(f"✓ 特徴量重要度CSVを保存: {importance_csv}")


def save_model(model: Any, output_dir: Path):
    """
    学習済みモデルを保存
    
    Args:
        model: 学習済みモデル
        output_dir: 出力ディレクトリ
    """
    print("\n💾 モデル保存中...")
    
    model_path = output_dir / 'model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ モデルを保存: {model_path}")
    
    # モデル情報も保存
    info_path = output_dir / 'model_info.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("学習済みモデル情報\n")
        f.write("="*60 + "\n\n")
        f.write(f"モデルタイプ: {type(model).__name__}\n")
        f.write(f"保存パス: {model_path}\n\n")
        f.write("【モデルパラメータ】\n")
        f.write(str(model.get_params()) + "\n")
    
    print(f"✓ モデル情報を保存: {info_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='炎上検知分類モデル学習スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python train_classifier.py labeled.csv output/
  python train_classifier.py labeled.csv output/ --model randomforest
  python train_classifier.py labeled.csv output/ --test-size 0.3
        """
    )
    
    parser.add_argument(
        'input_csv',
        type=str,
        help='ラベル付き特徴量CSVファイル'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='出力ディレクトリ'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'randomforest'],
        default='xgboost',
        help='使用するモデル (default: xgboost)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='テストデータの割合 (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='乱数シード (default: 42)'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔥 炎上検知分類モデル学習")
    print("="*60)
    print(f"  入力CSV: {args.input_csv}")
    print(f"  出力先: {args.output_dir}")
    print(f"  モデル: {args.model}")
    print(f"  テスト割合: {args.test_size}")
    print("="*60)
    
    try:
        # 1. データ読み込み
        df = load_data(args.input_csv)
        
        # 2. 特徴量準備
        X, y, feature_names = prepare_features(df)
        
        # 3. 学習・テスト分割
        print(f"\n✂️  データ分割中（train: {1-args.test_size:.0%}, test: {args.test_size:.0%}）...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y  # クラス比率を保持
        )
        print(f"✓ 訓練データ: {len(X_train)}件")
        print(f"✓ テストデータ: {len(X_test)}件")
        
        # 4. モデル学習
        model = train_model(X_train, y_train, model_type=args.model)
        
        # 5. モデル評価
        metrics = evaluate_model(model, X_test, y_test, output_dir)
        
        # 6. 特徴量重要度の可視化
        visualize_feature_importance(model, feature_names, output_dir)
        
        # 7. モデル保存
        save_model(model, output_dir)
        
        print("\n" + "="*60)
        print("✅ すべての処理が正常に完了しました！")
        print("="*60)
        print(f"\n📂 出力ファイル:")
        print(f"  - {output_dir / 'model.pkl'}")
        print(f"  - {output_dir / 'evaluation.txt'}")
        print(f"  - {output_dir / 'evaluation_metrics.png'}")
        print(f"  - {output_dir / 'feature_importance.png'}")
        print(f"  - {output_dir / 'feature_importance.csv'}")
        print(f"  - {output_dir / 'model_info.txt'}")
        print()
        
    except Exception as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
