#!/usr/bin/env python3
"""
BERT感情分析 + 時系列集計スクリプト
bert_sentiment.py の出力を1時間ごとに集計する
"""

import sys
import pandas as pd
from pathlib import Path


def aggregate_to_hourly(df):
    """
    ツイートデータを1時間ごとに集計
    
    Args:
        df: BERT感情分析済みDataFrame
        
    Returns:
        DataFrame: 1時間ごとの集計データ
    """
    # timestamp列を datetime に変換
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1時間単位で切り捨て
    df['hour'] = df['timestamp'].dt.floor('H')
    
    # 時間ごとにグループ化して集計
    hourly = df.groupby('hour').agg({
        'content': 'count',  # 投稿数
        'bert_positive': 'mean',  # ポジティブ確率の平均
        'bert_neutral': 'mean',   # ニュートラル確率の平均
        'bert_negative': 'mean',  # ネガティブ確率の平均
    }).reset_index()
    
    # 列名を変更（feature_builder.pyと互換性を持たせる）
    hourly.columns = ['timestamp', 'count', 'positive_prob', 'neutral_prob', 'negative_prob']
    
    # ラベルの集計（各時間帯で最も多いラベル）
    label_counts = df.groupby(['hour', 'bert_label']).size().reset_index(name='label_count')
    
    # 各ラベルの件数を計算（英語ラベルに対応）
    label_mapping = {
        'POSITIVE': 'positive',
        'NEUTRAL': 'neutral', 
        'NEGATIVE': 'negative'
    }
    
    for eng_label, col_prefix in label_mapping.items():
        label_df = label_counts[label_counts['bert_label'] == eng_label]
        label_dict = dict(zip(label_df['hour'], label_df['label_count']))
        hourly[f'{col_prefix}_count'] = hourly['timestamp'].map(label_dict).fillna(0).astype(int)
    
    # 各感情の割合を計算
    hourly['positive_rate'] = hourly['positive_count'] / hourly['count']
    hourly['neutral_rate'] = hourly['neutral_count'] / hourly['count']
    hourly['negative_rate'] = hourly['negative_count'] / hourly['count']
    
    # 感情スコア（ポジティブ: +1, ニュートラル: 0, ネガティブ: -1）
    # avg_score として出力（feature_builder.pyとの互換性）
    hourly['avg_score'] = (
        hourly['positive_prob'] - hourly['negative_prob']
    )
    
    return hourly


def main():
    """
    メイン処理
    """
    if len(sys.argv) != 3:
        print("使用法: python bert_sentiment_timeseries.py <bert_output_csv> <hourly_output_csv>")
        print("例: python bert_sentiment_timeseries.py data/processed/三苫_bert.csv data/processed/三苫_sentiment_1h.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=" * 60)
    print("BERT感情分析 時系列集計スクリプト")
    print("=" * 60)
    
    # 入力ファイル確認
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # CSV読み込み
        print(f"\n📂 データ読み込み中: {input_path}")
        df = pd.read_csv(input_path)
        print(f"✓ {len(df)}件のデータを読み込みました")
        
        # timestamp列がなければdate列を使用
        if 'timestamp' not in df.columns and 'date' in df.columns:
            df['timestamp'] = df['date']
            print(f"✓ 'date'列を'timestamp'に変換しました")
        
        # timestamp列がまだない場合エラー
        if 'timestamp' not in df.columns:
            raise ValueError("'timestamp'または'date'列が見つかりません")
        
        # 必須列の確認
        required_cols = ['timestamp', 'content', 'bert_label', 'bert_positive', 
                        'bert_neutral', 'bert_negative']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"必須列が見つかりません: {missing_cols}")
        
        # 時系列集計
        print(f"\n📊 1時間ごとに集計中...")
        hourly_df = aggregate_to_hourly(df)
        print(f"✓ {len(hourly_df)}時間分のデータに集計しました")
        
        # 統計情報表示
        print(f"\n📈 集計結果:")
        print(f"  期間: {hourly_df['timestamp'].min()} 〜 {hourly_df['timestamp'].max()}")
        print(f"  平均投稿数: {hourly_df['count'].mean():.1f}件/時間")
        print(f"  平均ポジティブ率: {hourly_df['positive_rate'].mean()*100:.1f}%")
        print(f"  平均ニュートラル率: {hourly_df['neutral_rate'].mean()*100:.1f}%")
        print(f"  平均ネガティブ率: {hourly_df['negative_rate'].mean()*100:.1f}%")
        print(f"  平均感情スコア: {hourly_df['avg_score'].mean():.3f}")
        
        # 結果保存
        print(f"\n💾 結果を保存中: {output_path}")
        hourly_df.to_csv(output_path, index=False)
        print(f"✓ 保存完了")
        
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
