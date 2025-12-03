#!/usr/bin/env python3
"""
辞書ベース感情分析モジュール

PN辞書を使用した従来型の感情分析
BERTとの比較用
"""

import pandas as pd
import numpy as np
from pathlib import Path
import MeCab
import re
from tqdm import tqdm


class DictionarySentiment:
    """辞書ベースの感情分析クラス"""
    
    def __init__(self, dictionary_path=None):
        """
        Args:
            dictionary_path: PN辞書のパス（CSV形式: word,polarity）
        """
        if dictionary_path is None:
            # デフォルトパス
            base_dir = Path(__file__).parent.parent.parent
            dictionary_path = base_dir / "data" / "dictionary" / "pn_ja.csv"
        
        self.dictionary_path = Path(dictionary_path)
        self.pn_dict = {}
        self._load_dictionary()
        
        # MeCab初期化
        self.mecab = MeCab.Tagger("-Owakati")
    
    def _load_dictionary(self):
        """PN辞書を読み込む"""
        if not self.dictionary_path.exists():
            print(f"⚠️ 辞書ファイルが見つかりません: {self.dictionary_path}")
            return
        
        df = pd.read_csv(self.dictionary_path)
        for _, row in df.iterrows():
            word = str(row['word']).strip()
            polarity = row['polarity']
            self.pn_dict[word] = polarity
        
        print(f"✓ PN辞書読み込み完了: {len(self.pn_dict)}語")
    
    def tokenize(self, text):
        """テキストを形態素解析"""
        if not isinstance(text, str):
            return []
        
        # 前処理
        text = re.sub(r'https?://\S+', '', text)  # URL除去
        text = re.sub(r'@\w+', '', text)  # メンション除去
        text = re.sub(r'#\w+', '', text)  # ハッシュタグ除去
        
        try:
            result = self.mecab.parse(text)
            tokens = result.strip().split()
            return tokens
        except:
            return []
    
    def analyze_text(self, text):
        """
        1つのテキストを感情分析
        
        Returns:
            dict: {
                'positive_score': float,  # ポジティブ語数
                'negative_score': float,  # ネガティブ語数
                'polarity': float,  # 極性スコア (-1 to 1)
                'sentiment': str,  # 'positive', 'negative', 'neutral'
                'word_count': int,
            }
        """
        tokens = self.tokenize(text)
        
        positive_count = 0
        negative_count = 0
        
        for token in tokens:
            if token in self.pn_dict:
                polarity = self.pn_dict[token]
                if polarity > 0:
                    positive_count += polarity
                elif polarity < 0:
                    negative_count += abs(polarity)
        
        total = positive_count + negative_count
        
        if total == 0:
            polarity = 0.0
            sentiment = 'neutral'
        else:
            polarity = (positive_count - negative_count) / total
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        return {
            'positive_score': positive_count,
            'negative_score': negative_count,
            'polarity': polarity,
            'sentiment': sentiment,
            'word_count': len(tokens),
        }
    
    def analyze_dataframe(self, df, text_column='text', batch_size=1000):
        """
        DataFrameに対して感情分析を実行
        
        Returns:
            pd.DataFrame: 感情分析結果が追加されたDataFrame
        """
        results = []
        
        print(f"📖 辞書ベース感情分析: {len(df)}件")
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="辞書分析"):
            text = row.get(text_column, '')
            result = self.analyze_text(text)
            results.append(result)
        
        result_df = pd.DataFrame(results)
        
        # 元のDataFrameと結合
        output_df = df.copy()
        output_df['dict_positive_score'] = result_df['positive_score']
        output_df['dict_negative_score'] = result_df['negative_score']
        output_df['dict_polarity'] = result_df['polarity']
        output_df['dict_sentiment'] = result_df['sentiment']
        
        return output_df


def analyze_csv(input_path, output_path=None, text_column='text'):
    """
    CSVファイルに対して辞書ベース感情分析を実行
    """
    print(f"\n{'='*60}")
    print("📖 辞書ベース感情分析")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    print(f"✓ 入力ファイル: {input_path} ({len(df)}件)")
    
    analyzer = DictionarySentiment()
    result_df = analyzer.analyze_dataframe(df, text_column=text_column)
    
    # 統計表示
    sentiments = result_df['dict_sentiment'].value_counts(normalize=True) * 100
    print(f"\n📊 感情分布:")
    for s, pct in sentiments.items():
        print(f"  {s}: {pct:.1f}%")
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\n💾 出力: {output_path}")
    
    return result_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='辞書ベース感情分析')
    parser.add_argument('input', help='入力CSVファイル')
    parser.add_argument('-o', '--output', help='出力CSVファイル')
    parser.add_argument('-t', '--text-column', default='text', help='テキスト列名')
    
    args = parser.parse_args()
    
    output = args.output or args.input.replace('.csv', '_dict_sentiment.csv')
    analyze_csv(args.input, output, args.text_column)
