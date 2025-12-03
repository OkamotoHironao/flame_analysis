#!/usr/bin/env python3
"""
辞書ベース感情分析の時系列集計

BERTの時系列集計と同じフォーマットで出力
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from dictionary_sentiment import DictionarySentiment


def aggregate_dict_sentiment(input_path, output_path, interval='1h'):
    """
    辞書ベース感情分析を実行し、時系列で集計
    
    Args:
        input_path: 標準化されたCSV（text, created_at列を含む）
        output_path: 出力CSVパス
        interval: 集計間隔（'1h', '3h', '1d'など）
    """
    print(f"\n{'='*60}")
    print("📖 辞書ベース感情分析 + 時系列集計")
    print(f"{'='*60}")
    
    # データ読み込み
    df = pd.read_csv(input_path)
    print(f"✓ 入力: {input_path} ({len(df)}件)")
    
    # タイムスタンプ解析
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("created_at または timestamp 列が必要です")
    
    # 辞書分析
    analyzer = DictionarySentiment()
    
    # テキスト列を検出（text, content, body などに対応）
    text_column = None
    for col in ['text', 'content', 'body', 'tweet']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("テキスト列が見つかりません（text, content, body のいずれか必要）")
    
    print(f"📝 テキスト列: {text_column}")
    
    results = []
    for i, row in df.iterrows():
        text = row.get(text_column, '')
        result = analyzer.analyze_text(text)
        results.append(result)
    
    result_df = pd.DataFrame(results)
    df['dict_sentiment'] = result_df['sentiment']
    df['dict_polarity'] = result_df['polarity']
    
    print(f"\n📊 感情分布:")
    dist = df['dict_sentiment'].value_counts(normalize=True) * 100
    for s, pct in dist.items():
        print(f"  {s}: {pct:.1f}%")
    
    # 時系列集計
    print(f"\n⏰ {interval}単位で集計中...")
    
    df.set_index('timestamp', inplace=True)
    
    grouped = df.resample(interval)
    
    aggregated = pd.DataFrame({
        'timestamp': grouped.size().index,
        'volume': grouped.size().values,
        'dict_positive_count': grouped.apply(lambda x: (x['dict_sentiment'] == 'positive').sum()).values,
        'dict_negative_count': grouped.apply(lambda x: (x['dict_sentiment'] == 'negative').sum()).values,
        'dict_neutral_count': grouped.apply(lambda x: (x['dict_sentiment'] == 'neutral').sum()).values,
        'dict_avg_polarity': grouped['dict_polarity'].mean().values,
    })
    
    # 割合を計算
    aggregated['dict_positive_rate'] = aggregated['dict_positive_count'] / aggregated['volume']
    aggregated['dict_negative_rate'] = aggregated['dict_negative_count'] / aggregated['volume']
    aggregated['dict_neutral_rate'] = aggregated['dict_neutral_count'] / aggregated['volume']
    
    # NaN処理
    aggregated = aggregated.fillna(0)
    
    # volumeが0の行を除外
    aggregated = aggregated[aggregated['volume'] > 0]
    
    # 保存
    aggregated.to_csv(output_path, index=False)
    print(f"\n💾 出力: {output_path} ({len(aggregated)}件)")
    
    # 統計
    print(f"\n📊 集計結果:")
    print(f"  期間: {aggregated['timestamp'].min()} 〜 {aggregated['timestamp'].max()}")
    print(f"  平均投稿量: {aggregated['volume'].mean():.1f}/時間")
    print(f"  平均ネガティブ率: {aggregated['dict_negative_rate'].mean()*100:.1f}%")
    
    return aggregated


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='辞書ベース感情分析の時系列集計')
    parser.add_argument('input', help='入力CSVファイル')
    parser.add_argument('-o', '--output', help='出力CSVファイル')
    parser.add_argument('-i', '--interval', default='1h', help='集計間隔')
    
    args = parser.parse_args()
    
    output = args.output or args.input.replace('.csv', '_dict_sentiment_1h.csv')
    aggregate_dict_sentiment(args.input, output, args.interval)
