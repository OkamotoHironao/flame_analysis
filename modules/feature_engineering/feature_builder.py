#!/usr/bin/env python3
"""
特徴量統合スクリプト（Feature Builder）
感情分析と立場分類の結果を統合し、炎上判定用の特徴量テーブルを生成
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm


def extract_query_name(filepath):
    """
    ファイル名またはディレクトリパスからクエリ名を抽出
    
    Args:
        filepath: ファイルパス
        
    Returns:
        str: クエリ名（例: "松本人志"）
    """
    path = Path(filepath)
    
    # ディレクトリ構造から抽出（data/original/松本人志/ など）
    if 'original' in path.parts:
        original_idx = path.parts.index('original')
        if original_idx + 1 < len(path.parts):
            potential_query = path.parts[original_idx + 1]
            # ファイル名でなければクエリ名として採用
            if not potential_query.endswith('.csv'):
                return potential_query
    
    # outputs配下の場合（outputs/松本人志/ など）
    if 'outputs' in path.parts:
        outputs_idx = path.parts.index('outputs')
        if outputs_idx + 1 < len(path.parts):
            potential_query = path.parts[outputs_idx + 1]
            # ファイル名でなければクエリ名として採用
            if not potential_query.endswith('.csv'):
                return potential_query
    
    # ファイル名から抽出（松本人志_*.csv）
    filename = path.stem
    # サフィックスを除去
    for suffix in ['_sentiment_1h', '_sentiment_30m', '_sentiment_10m', '_stance', '_analyzed', '_bert', '_feature_table']:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break
    
    # 残りの部分からタイムスタンプ（数字のみ）を除去
    parts = filename.split('_')
    query_parts = [p for p in parts if not p.isdigit()]
    
    return '_'.join(query_parts) if query_parts else "unknown"


def load_sentiment_timeseries(csv_path):
    """
    時系列化済みの感情分析結果を読み込む
    
    Args:
        csv_path: 感情分析CSVのパス
        
    Returns:
        pd.DataFrame: 感情分析の時系列データ
    """
    print(f"\n📊 感情分析データ読み込み中: {csv_path}")
    
    df = pd.read_csv(csv_path, comment='#')
    
    # timestamp列をdatetimeに変換
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    else:
        raise ValueError("感情分析CSVに 'timestamp' または 'time' 列が必要です")
    
    # 必要な列を確認
    required_cols = ['timestamp', 'count', 'negative_rate']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"感情分析CSVに '{col}' 列が必要です")
    
    # avg_score列がない場合は0で埋める
    if 'avg_score' not in df.columns:
        df['avg_score'] = 0.0
    
    print(f"✓ {len(df)}件のタイムスタンプを読み込みました")
    print(f"  期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    
    return df[['timestamp', 'count', 'avg_score', 'negative_rate']]


def load_and_aggregate_stance(csv_path):
    """
    立場分類結果を読み込み、1時間ごとに集計
    
    Args:
        csv_path: 立場分類CSVのパス
        
    Returns:
        pd.DataFrame: 1時間ごとに集計された立場分類データ
    """
    print(f"\n🎯 立場分類データ読み込み中: {csv_path}")
    
    df = pd.read_csv(csv_path, comment='#')
    
    # 日時列の確認
    date_col = None
    for col in ['created_at', 'date', 'timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("立場分類CSVに 'created_at', 'date', または 'timestamp' 列が必要です")
    
    # datetime変換
    df['timestamp'] = pd.to_datetime(df[date_col])
    
    # 1時間単位に丸める（床切り）
    df['timestamp'] = df['timestamp'].dt.floor('h')
    
    print(f"✓ {len(df)}件のツイートを読み込みました")
    print(f"  期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    
    # 立場ラベルの確認
    if 'stance_label' not in df.columns:
        raise ValueError("立場分類CSVに 'stance_label' 列が必要です")
    
    print(f"\n📈 1時間ごとに集計中...")
    
    # 時間ごとに集計
    aggregated_data = []
    
    for timestamp, group in tqdm(df.groupby('timestamp'), desc="集計中"):
        total = len(group)
        
        # 立場の割合を計算
        stance_counts = group['stance_label'].value_counts()
        against_count = stance_counts.get('AGAINST', 0)
        favor_count = stance_counts.get('FAVOR', 0)
        neutral_count = stance_counts.get('NEUTRAL', 0)
        
        # 確率の平均を計算（列がある場合）
        against_mean = group['stance_against'].mean() if 'stance_against' in group.columns else 0.0
        favor_mean = group['stance_favor'].mean() if 'stance_favor' in group.columns else 0.0
        neutral_mean = group['stance_neutral'].mean() if 'stance_neutral' in group.columns else 0.0
        
        # エンゲージメント指標を集計
        avg_like = group['like_count'].mean() if 'like_count' in group.columns else 0.0
        avg_retweet = group['retweet_count'].mean() if 'retweet_count' in group.columns else 0.0
        avg_reply = group['reply_count'].mean() if 'reply_count' in group.columns else 0.0
        
        max_like = group['like_count'].max() if 'like_count' in group.columns else 0.0
        max_retweet = group['retweet_count'].max() if 'retweet_count' in group.columns else 0.0
        max_reply = group['reply_count'].max() if 'reply_count' in group.columns else 0.0
        
        total_like = group['like_count'].sum() if 'like_count' in group.columns else 0.0
        total_retweet = group['retweet_count'].sum() if 'retweet_count' in group.columns else 0.0
        total_reply = group['reply_count'].sum() if 'reply_count' in group.columns else 0.0
        
        aggregated_data.append({
            'timestamp': timestamp,
            'stance_against_rate': against_count / total if total > 0 else 0.0,
            'stance_favor_rate': favor_count / total if total > 0 else 0.0,
            'stance_neutral_rate': neutral_count / total if total > 0 else 0.0,
            'stance_against_mean': against_mean,
            'stance_favor_mean': favor_mean,
            'stance_neutral_mean': neutral_mean,
            'stance_count': total,
            'avg_like_count': avg_like,
            'avg_retweet_count': avg_retweet,
            'avg_reply_count': avg_reply,
            'max_like_count': max_like,
            'max_retweet_count': max_retweet,
            'max_reply_count': max_reply,
            'total_like_count': total_like,
            'total_retweet_count': total_retweet,
            'total_reply_count': total_reply
        })
    
    result = pd.DataFrame(aggregated_data)
    print(f"✓ {len(result)}個の時間枠に集計しました")
    
    return result


def create_features(sentiment_df, stance_df):
    """
    感情分析と立場分類のデータを統合し、特徴量を生成
    
    Args:
        sentiment_df: 感情分析の時系列データ
        stance_df: 立場分類の集計データ
        
    Returns:
        pd.DataFrame: 特徴量テーブル
    """
    print(f"\n🔧 特徴量生成中...")
    
    # timestamp をキーにして結合
    df = pd.merge(
        sentiment_df,
        stance_df,
        on='timestamp',
        how='outer'
    )
    
    # timestamp でソート
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 基本特徴量
    df['volume'] = df['count'].fillna(0)
    df['sentiment_avg_score'] = df['avg_score'].fillna(0)
    
    # 欠損値を0で埋める
    fill_cols = [
        'negative_rate', 
        'stance_against_rate', 
        'stance_favor_rate',
        'stance_neutral_rate',
        'stance_against_mean',
        'stance_favor_mean',
        'stance_neutral_mean'
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # エンゲージメント複合特徴量
    if 'avg_like_count' in df.columns:
        df['avg_engagement'] = df['avg_like_count'] + df['avg_retweet_count'] + df['avg_reply_count']
        df['total_engagement'] = df['total_like_count'] + df['total_retweet_count'] + df['total_reply_count']
        df['max_engagement'] = df[['max_like_count', 'max_retweet_count', 'max_reply_count']].max(axis=1)
        df['engagement_rate'] = (df['total_engagement'] / df['volume']).fillna(0).replace([float('inf'), float('-inf')], 0)
    else:
        df['avg_engagement'] = 0
        df['total_engagement'] = 0
        df['max_engagement'] = 0
        df['engagement_rate'] = 0
    
    # 差分特徴量（前時間との差）
    df['delta_volume'] = df['volume'].diff().fillna(0)
    df['delta_negative_rate'] = df['negative_rate'].diff().fillna(0)
    df['delta_against_rate'] = df['stance_against_rate'].diff().fillna(0)
    df['delta_engagement'] = df['total_engagement'].diff().fillna(0)
    
    # 変化率特徴量（前時間比）
    df['delta_volume_rate'] = (df['delta_volume'] / df['volume'].shift(1)).fillna(0)
    df['delta_volume_rate'] = df['delta_volume_rate'].replace([float('inf'), float('-inf')], 0)
    df['delta_engagement_rate'] = (df['delta_engagement'] / df['total_engagement'].shift(1)).fillna(0)
    df['delta_engagement_rate'] = df['delta_engagement_rate'].replace([float('inf'), float('-inf')], 0)
    
    # 最終的な特徴量列を選択
    feature_cols = [
        'timestamp',
        'volume',
        'negative_rate',
        'stance_against_rate',
        'stance_favor_rate',
        'stance_neutral_rate',
        'delta_volume',
        'delta_negative_rate',
        'delta_against_rate',
        'delta_volume_rate',
        'sentiment_avg_score',
        'stance_against_mean',
        'stance_favor_mean',
        'stance_neutral_mean',
        'avg_engagement',
        'total_engagement',
        'max_engagement',
        'engagement_rate',
        'delta_engagement',
        'delta_engagement_rate',
        'avg_like_count',
        'avg_retweet_count',
        'avg_reply_count'
    ]
    
    # 存在する列のみ選択
    existing_cols = [col for col in feature_cols if col in df.columns]
    df = df[existing_cols]
    
    print(f"✓ {len(df)}件の特徴量レコードを生成しました")
    
    return df


def print_statistics(df):
    """
    統計情報を表示
    
    Args:
        df: 特徴量テーブル
    """
    print(f"\n{'='*60}")
    print("📊 特徴量テーブル統計")
    print(f"{'='*60}")
    
    print(f"\n📈 基本情報:")
    print(f"  レコード数: {len(df)}件")
    print(f"  期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    
    print(f"\n📊 平均値:")
    if 'volume' in df.columns:
        print(f"  平均投稿数: {df['volume'].mean():.2f}件/時間")
    if 'negative_rate' in df.columns:
        print(f"  平均ネガティブ率: {df['negative_rate'].mean():.2%}")
    if 'stance_against_rate' in df.columns:
        print(f"  平均AGAINST率: {df['stance_against_rate'].mean():.2%}")
    if 'stance_favor_rate' in df.columns:
        print(f"  平均FAVOR率: {df['stance_favor_rate'].mean():.2%}")
    
    print(f"\n🚀 最大値:")
    if 'volume' in df.columns:
        max_volume_idx = df['volume'].idxmax()
        print(f"  最大投稿数: {df.loc[max_volume_idx, 'volume']:.0f}件 ({df.loc[max_volume_idx, 'timestamp']})")
    
    if 'delta_volume' in df.columns:
        max_delta_idx = df['delta_volume'].idxmax()
        print(f"  最大投稿急増: +{df.loc[max_delta_idx, 'delta_volume']:.0f}件 ({df.loc[max_delta_idx, 'timestamp']})")
    
    if 'delta_volume_rate' in df.columns:
        max_rate_idx = df['delta_volume_rate'].idxmax()
        max_rate = df.loc[max_rate_idx, 'delta_volume_rate']
        if max_rate > 0:
            print(f"  最大投稿急増率: {max_rate:.1%} ({df.loc[max_rate_idx, 'timestamp']})")
    
    if 'negative_rate' in df.columns:
        max_neg_idx = df['negative_rate'].idxmax()
        print(f"  最大ネガティブ率: {df.loc[max_neg_idx, 'negative_rate']:.1%} ({df.loc[max_neg_idx, 'timestamp']})")
    
    print(f"\n{'='*60}")


def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(
        description='感情分析と立場分類の結果を統合し、特徴量テーブルを生成'
    )
    parser.add_argument(
        '--sentiment_csv',
        required=True,
        help='感情分析の時系列CSV（例: sentiment_analysis/松本人志_sentiment_1h.csv）'
    )
    parser.add_argument(
        '--stance_csv',
        required=True,
        help='立場分類の推論結果CSV（例: stance_results/松本人志_stance.csv）'
    )
    parser.add_argument(
        '--output_csv',
        help='出力CSVパス（省略時は自動生成）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("特徴量統合スクリプト (Feature Builder)")
    print("=" * 60)
    
    # スクリプトのディレクトリに移動
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # ファイルの存在確認
    if not Path(args.sentiment_csv).exists():
        print(f"❌ エラー: 感情分析CSVが見つかりません: {args.sentiment_csv}")
        sys.exit(1)
    
    if not Path(args.stance_csv).exists():
        print(f"❌ エラー: 立場分類CSVが見つかりません: {args.stance_csv}")
        sys.exit(1)
    
    try:
        # データ読み込み
        sentiment_df = load_sentiment_timeseries(args.sentiment_csv)
        stance_df = load_and_aggregate_stance(args.stance_csv)
        
        # 特徴量生成
        features_df = create_features(sentiment_df, stance_df)
        
        # 統計表示
        print_statistics(features_df)
        
        # 出力パス決定
        if args.output_csv:
            output_path = args.output_csv
        else:
            # クエリ名を抽出
            query = extract_query_name(args.sentiment_csv)
            output_path = f"outputs/{query}/{query}_feature_table.csv"
        
        # 出力ディレクトリ作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV保存
        print(f"\n💾 特徴量テーブルを保存中: {output_path}")
        features_df.to_csv(output_path, index=False)
        print("✓ 保存完了")
        
        print("\n" + "=" * 60)
        print("✅ すべての処理が正常に完了しました！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# 実行例:
# python feature_builder.py \
#   --sentiment_csv ../../modules/sentiment_analysis/dictionary_based/outputs/松本人志/松本人志_sentiment_1h.csv \
#   --stance_csv ../../modules/stance_detection/outputs/松本人志/松本人志_stance.csv \
#   --output_csv outputs/松本人志/松本人志_feature_table.csv
