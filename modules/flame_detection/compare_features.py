#!/usr/bin/env python3
"""
特徴量の比較実験

現在の特徴量セット vs 新しい特徴量を追加したセットを比較
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ベースディレクトリ
BASE_DIR = Path(__file__).parent.parent.parent


# ===== 特徴量定義 =====

# 現在の基本特徴量
BASE_FEATURES = [
    'volume',
    'delta_volume',
    'negative_rate',
    'delta_negative_rate',
    'stance_favor_rate',
    'stance_against_rate',
    'stance_neutral_rate',
]

# 現在の拡張特徴量
CURRENT_EXTENDED = BASE_FEATURES + [
    'flame_score',
    'against_count',
    'negative_rate_log',
    'volume_log',
    'sentiment_polarity',
    'is_high_volume',
    'is_high_negative',
    'is_both_high',
]

# 新しく追加する特徴量
NEW_FEATURES = [
    'negative_engagement',   # ネガティブ × エンゲージメント
    'volume_spike_ratio',    # 通常時からの乖離率
    'polarization',          # 意見の二極化度
    'negative_momentum',     # ネガティブ拡大の勢い
]

# 重要度0の特徴量を削除した最適化版
OPTIMIZED_FEATURES = BASE_FEATURES + [
    'flame_score',
    'against_count',
    'sentiment_polarity',
    # 重要度0を削除: negative_rate_log, volume_log, is_high_volume, is_high_negative, is_both_high
]

# 最適化 + 有効な新特徴量を追加
OPTIMIZED_WITH_NEW = OPTIMIZED_FEATURES + [
    'negative_engagement',   # 重要度3位
    'negative_momentum',     # 重要度8位
    'polarization',          # 重要度13位
    # volume_spike_ratioは重要度0なので除外
]


def load_standardized_data(topic_name):
    """標準化データを読み込み"""
    csv_path = BASE_DIR / "data" / "standardized" / f"{topic_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_labeled_data(topic_name):
    """ラベル付き時系列データを読み込み"""
    csv_path = BASE_DIR / "modules" / "flame_detection" / "outputs" / topic_name / f"{topic_name}_labeled.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['topic'] = topic_name
        return df
    return None


def calculate_engagement_features(topic_name):
    """元データからエンゲージメント関連の特徴量を計算"""
    # 標準化データ読み込み
    raw_df = load_standardized_data(topic_name)
    if raw_df is None:
        return None
    
    # タイムスタンプをパース（タイムゾーン対応）
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # タイムゾーン情報がない場合はJSTとして扱う
    if raw_df['timestamp'].dt.tz is None:
        raw_df['timestamp'] = raw_df['timestamp'].dt.tz_localize('Asia/Tokyo')
    
    raw_df['hour'] = raw_df['timestamp'].dt.floor('h')
    
    # エンゲージメントを計算
    raw_df['engagement'] = raw_df['like_count'].fillna(0) + raw_df['retweet_count'].fillna(0)
    
    # 時間窓ごとに集計
    hourly = raw_df.groupby('hour').agg({
        'engagement': ['sum', 'mean'],
        'like_count': 'sum',
        'retweet_count': 'sum',
    }).reset_index()
    
    hourly.columns = ['timestamp', 'total_engagement', 'avg_engagement', 'total_likes', 'total_retweets']
    # タイムゾーン付きでフォーマット（ラベル付きデータと一致させる）
    hourly['timestamp'] = hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S+09:00')
    
    return hourly


def add_current_composite_features(df):
    """現在の複合特徴量を追加"""
    df = df.copy()
    
    # 1. 炎上スコア
    df['flame_score'] = df['volume'] * df['negative_rate']
    
    # 2. 批判的投稿の絶対数
    df['against_count'] = df['volume'] * df['stance_against_rate']
    
    # 3. ネガティブ率の対数変換
    df['negative_rate_log'] = np.log1p(df['negative_rate'] * 100)
    
    # 4. 投稿量の対数変換
    df['volume_log'] = np.log1p(df['volume'])
    
    # 5. 感情極性
    df['sentiment_polarity'] = df['stance_against_rate'] - df['stance_favor_rate']
    
    # 6. 投稿量が閾値以上か
    df['is_high_volume'] = (df['volume'] >= 50).astype(int)
    
    # 7. ネガティブ率が閾値以上か
    df['is_high_negative'] = (df['negative_rate'] >= 0.2).astype(int)
    
    # 8. 両方高い場合
    df['is_both_high'] = ((df['volume'] >= 50) & (df['negative_rate'] >= 0.2)).astype(int)
    
    return df


def add_new_features(df):
    """新しい特徴量を追加"""
    df = df.copy()
    
    # 1. negative_engagement: ネガティブ × エンゲージメント
    if 'total_engagement' in df.columns:
        df['total_engagement'] = df['total_engagement'].fillna(0)
        df['negative_engagement'] = df['negative_rate'] * df['total_engagement']
    else:
        # エンゲージメントデータがない場合は volume で代用
        df['negative_engagement'] = df['negative_rate'] * df['volume']
    
    # 2. volume_spike_ratio: 通常時からの乖離率
    volume_mean = df['volume'].mean()
    volume_std = df['volume'].std()
    if volume_std > 0:
        df['volume_spike_ratio'] = (df['volume'] - volume_mean) / volume_std
    else:
        df['volume_spike_ratio'] = 0
    
    # 3. polarization: 意見の二極化度（賛成×反対）
    df['polarization'] = df['stance_favor_rate'] * df['stance_against_rate']
    
    # 4. negative_momentum: ネガティブ拡大の勢い
    # delta_negative_rate × volume（ネガティブが増えていて、かつ量が多い）
    df['negative_momentum'] = df['delta_negative_rate'] * df['volume']
    
    return df


def prepare_features(df, feature_columns):
    """特徴量を準備"""
    available_cols = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]
    
    if missing_cols:
        print(f"    ⚠️ 欠損特徴量: {missing_cols}")
    
    X = df[available_cols].copy()
    X = X.fillna(0)
    
    return X, available_cols


def evaluate_model(X, y, feature_name=""):
    """クロスバリデーションでモデルを評価"""
    # クラス重み計算
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []
    roc_aucs = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, y_proba))
    
    # 特徴量重要度（最後のモデルから）
    importance = dict(zip(X.columns, model.feature_importances_))
    
    return {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'roc_auc': np.mean(roc_aucs) if roc_aucs else 0,
        'roc_auc_std': np.std(roc_aucs) if roc_aucs else 0,
        'importance': importance,
    }


def main():
    print("=" * 70)
    print("🔬 特徴量比較実験")
    print("=" * 70)
    
    # トピック一覧を取得
    outputs_dir = BASE_DIR / "modules" / "flame_detection" / "outputs"
    topics = []
    for topic_dir in outputs_dir.iterdir():
        if topic_dir.is_dir():
            labeled_csv = topic_dir / f"{topic_dir.name}_labeled.csv"
            if labeled_csv.exists():
                topics.append(topic_dir.name)
    
    print(f"\n📊 対象トピック: {topics}")
    
    # 全トピックのデータを結合
    all_data = []
    engagement_data = {}
    
    for topic in topics:
        df = load_labeled_data(topic)
        if df is not None:
            all_data.append(df)
            # エンゲージメントデータも読み込み
            eng_df = calculate_engagement_features(topic)
            if eng_df is not None:
                engagement_data[topic] = eng_df
            print(f"  ✅ {topic}: {len(df)}件, 炎上率: {df['is_controversy'].mean()*100:.1f}%")
    
    if not all_data:
        print("❌ データがありません")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📈 合計: {len(combined_df)}件, 炎上: {combined_df['is_controversy'].sum()}件")
    
    # 現在の特徴量を追加
    combined_df = add_current_composite_features(combined_df)
    
    # エンゲージメントデータをマージ
    eng_frames = []
    for topic, eng_df in engagement_data.items():
        eng_df = eng_df.copy()
        eng_df['topic'] = topic
        eng_frames.append(eng_df)
    
    if eng_frames:
        all_engagement = pd.concat(eng_frames, ignore_index=True)
        combined_df = combined_df.merge(
            all_engagement[['timestamp', 'topic', 'total_engagement', 'avg_engagement']],
            on=['timestamp', 'topic'],
            how='left'
        )
    
    # 新しい特徴量を追加
    combined_df = add_new_features(combined_df)
    
    # ラベル
    y = combined_df['is_controversy']
    
    # 5つの特徴量セットで比較
    feature_sets = {
        '① 基本特徴量のみ (7個)': BASE_FEATURES,
        '② 現在の拡張特徴量 (15個)': CURRENT_EXTENDED,
        '③ 全部追加 (19個)': CURRENT_EXTENDED + NEW_FEATURES,
        '④ 最適化版 (10個)': OPTIMIZED_FEATURES,
        '⑤ 最適化+新特徴量 (13個)': OPTIMIZED_WITH_NEW,
    }
    
    print("\n" + "=" * 70)
    print("📊 比較結果")
    print("=" * 70)
    
    results = {}
    
    for name, features in feature_sets.items():
        print(f"\n🔹 {name}")
        X, used_features = prepare_features(combined_df, features)
        print(f"   使用特徴量: {len(used_features)}個")
        
        result = evaluate_model(X, y, name)
        results[name] = result
        
        print(f"   Accuracy: {result['accuracy']*100:.2f}% (±{result['accuracy_std']*100:.2f})")
        print(f"   F1 Score: {result['f1']*100:.2f}% (±{result['f1_std']*100:.2f})")
        print(f"   ROC-AUC:  {result['roc_auc']*100:.2f}% (±{result['roc_auc_std']*100:.2f})")
    
    # 比較サマリー
    print("\n" + "=" * 70)
    print("📈 比較サマリー")
    print("=" * 70)
    print(f"\n{'特徴量セット':<30} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    
    for name, result in results.items():
        short_name = name.split('(')[0].strip()
        print(f"{short_name:<30} {result['accuracy']*100:>9.2f}% {result['f1']*100:>9.2f}% {result['roc_auc']*100:>9.2f}%")
    
    # 新特徴量の重要度
    print("\n" + "=" * 70)
    print("🆕 特徴量重要度（最適化+新特徴量版）")
    print("=" * 70)
    
    new_features_result = results['⑤ 最適化+新特徴量 (13個)']
    importance = new_features_result['importance']
    
    # 全特徴量を重要度順にソート
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n全特徴量ランキング:")
    for i, (feat, imp) in enumerate(sorted_importance, 1):
        marker = "🆕" if feat in NEW_FEATURES else "  "
        print(f"  {i:2}. {marker} {feat:<25} {imp:.4f}")
    
    # 新特徴量のみ
    print("\n新特徴量のみ:")
    for feat in NEW_FEATURES:
        if feat in importance:
            print(f"  - {feat:<25} {importance[feat]:.4f}")
    
    # 改善度の計算
    print("\n" + "=" * 70)
    print("📊 改善度分析")
    print("=" * 70)
    
    baseline = results['② 現在の拡張特徴量 (15個)']
    optimized = results['④ 最適化版 (10個)']
    optimized_new = results['⑤ 最適化+新特徴量 (13個)']
    
    print(f"\n【現在 vs 最適化版】（重要度0の特徴量を削除）")
    acc_diff = (optimized['accuracy'] - baseline['accuracy']) * 100
    f1_diff = (optimized['f1'] - baseline['f1']) * 100
    print(f"  Accuracy: {baseline['accuracy']*100:.2f}% → {optimized['accuracy']*100:.2f}% ({'+' if acc_diff >= 0 else ''}{acc_diff:.2f}%)")
    print(f"  F1 Score: {baseline['f1']*100:.2f}% → {optimized['f1']*100:.2f}% ({'+' if f1_diff >= 0 else ''}{f1_diff:.2f}%)")
    print(f"  特徴量数: 15個 → 10個 (5個削減)")
    
    print(f"\n【最適化版 vs 最適化+新特徴量】")
    acc_diff = (optimized_new['accuracy'] - optimized['accuracy']) * 100
    f1_diff = (optimized_new['f1'] - optimized['f1']) * 100
    print(f"  Accuracy: {optimized['accuracy']*100:.2f}% → {optimized_new['accuracy']*100:.2f}% ({'+' if acc_diff >= 0 else ''}{acc_diff:.2f}%)")
    print(f"  F1 Score: {optimized['f1']*100:.2f}% → {optimized_new['f1']*100:.2f}% ({'+' if f1_diff >= 0 else ''}{f1_diff:.2f}%)")
    
    # 最良の特徴量セットを判定
    print("\n" + "=" * 70)
    print("🏆 推奨")
    print("=" * 70)
    
    best_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best = results[best_name]
    print(f"\n最良の特徴量セット: {best_name}")
    print(f"  Accuracy: {best['accuracy']*100:.2f}%")
    print(f"  F1 Score: {best['f1']*100:.2f}%")
    print(f"  ROC-AUC:  {best['roc_auc']*100:.2f}%")


if __name__ == "__main__":
    main()
