#!/usr/bin/env python3
"""
改良版感情スコア計算モジュール

BERTの感情分析結果を補正し、より正確な「炎上度」を算出する

問題点:
- BERTは引用・報道形式の批判を「NEUTRAL」と判定しやすい
- 「性加害」「辞めろ」などの批判キーワードがNEUTRALになる

解決策:
1. キーワードブースト: 批判キーワードを含む場合にネガティブスコアを補正
2. スタンス統合: AGAINST（批判的立場）をネガティブ指標として活用
3. アンサンブル: 複数指標を重み付けして統合

Usage:
    from enhanced_sentiment import calculate_enhanced_negativity
    df = calculate_enhanced_negativity(df)
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional


# 批判的キーワード辞書（カテゴリ別）
NEGATIVE_KEYWORDS = {
    # 強い批判（weight: 0.8）
    'strong': [
        'クソ', 'くそ', '最低', '最悪', '消えろ', '死ね', 'キモい', 'きもい',
        '気持ち悪', 'ゴミ', 'カス', 'クズ', '老害', '害悪', '犯罪', 'レイプ',
        '性加害', '性暴力', '許せない', '許さない', '終わってる', 'ありえない',
    ],
    # 中程度の批判（weight: 0.5）
    'medium': [
        '辞めろ', 'やめろ', '引退', '追放', '降板', '嫌い', '無理', 'ダメ',
        '問題', '批判', '炎上', '疑惑', '告発', '被害', '謝罪', '責任',
        'ひどい', 'ヒドイ', '酷い', '残念', 'がっかり', '失望',
    ],
    # 弱い批判（weight: 0.3）
    'weak': [
        '微妙', 'どうなの', 'いかがなものか', '納得いかない', '疑問',
        '違和感', '不快', '不満', '心配', '危険', '怖い',
    ]
}

# ポジティブキーワード（ネガティブ打ち消し用）
POSITIVE_KEYWORDS = [
    '面白い', '最高', '神', '天才', '好き', '応援', 'すごい', '素晴らしい',
    'おめでとう', '感動', '笑った', '爆笑', 'ウケる',
]


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """テキスト内のキーワードマッチ数をカウント"""
    if pd.isna(text):
        return 0
    text = str(text).lower()
    count = 0
    for kw in keywords:
        if kw.lower() in text:
            count += 1
    return count


def calculate_keyword_negativity(text: str) -> float:
    """
    キーワードベースのネガティブスコアを計算
    
    Returns:
        float: 0.0〜1.0のネガティブスコア
    """
    if pd.isna(text):
        return 0.0
    
    text = str(text)
    
    # 各カテゴリのマッチ数
    strong_count = count_keyword_matches(text, NEGATIVE_KEYWORDS['strong'])
    medium_count = count_keyword_matches(text, NEGATIVE_KEYWORDS['medium'])
    weak_count = count_keyword_matches(text, NEGATIVE_KEYWORDS['weak'])
    positive_count = count_keyword_matches(text, POSITIVE_KEYWORDS)
    
    # 重み付けスコア
    negative_score = (strong_count * 0.8 + medium_count * 0.5 + weak_count * 0.3)
    positive_score = positive_count * 0.5
    
    # ネガティブからポジティブを引く（最小0）
    net_score = max(0, negative_score - positive_score)
    
    # 0〜1に正規化（3以上で1.0）
    normalized = min(1.0, net_score / 3.0)
    
    return normalized


def calculate_enhanced_negativity(
    df: pd.DataFrame,
    bert_weight: float = 0.4,
    keyword_weight: float = 0.3,
    stance_weight: float = 0.3,
    content_col: str = 'content',
    bert_neg_col: str = 'bert_negative',
    bert_label_col: str = 'bert_label',
    stance_col: str = 'stance_label',
) -> pd.DataFrame:
    """
    改良版ネガティブスコアを計算してDataFrameに追加
    
    Args:
        df: 入力DataFrame
        bert_weight: BERTスコアの重み
        keyword_weight: キーワードスコアの重み
        stance_weight: スタンススコアの重み
        
    Returns:
        DataFrame: enhanced_negativity列が追加されたDataFrame
    """
    df = df.copy()
    
    # 1. キーワードベースのネガティブスコア
    df['keyword_negativity'] = df[content_col].apply(calculate_keyword_negativity)
    
    # 2. BERTのネガティブスコア（NEUTRALでもbert_negativeを使用）
    if bert_neg_col in df.columns:
        df['bert_negativity'] = df[bert_neg_col]
    else:
        df['bert_negativity'] = 0.0
    
    # 3. スタンスベースのネガティブスコア
    if stance_col in df.columns:
        # AGAINST=1.0, NEUTRAL=0.3, FAVOR=0.0
        stance_map = {'AGAINST': 1.0, 'NEUTRAL': 0.3, 'FAVOR': 0.0}
        df['stance_negativity'] = df[stance_col].map(stance_map).fillna(0.3)
    else:
        df['stance_negativity'] = 0.0
        stance_weight = 0
        # 重みを再配分
        total = bert_weight + keyword_weight
        bert_weight = bert_weight / total
        keyword_weight = keyword_weight / total
    
    # 4. アンサンブルスコア
    df['enhanced_negativity'] = (
        df['bert_negativity'] * bert_weight +
        df['keyword_negativity'] * keyword_weight +
        df['stance_negativity'] * stance_weight
    )
    
    # 5. 改良版ラベル（閾値0.4でネガティブ判定）
    df['enhanced_label'] = df['enhanced_negativity'].apply(
        lambda x: 'NEGATIVE' if x >= 0.4 else ('POSITIVE' if x <= 0.15 else 'NEUTRAL')
    )
    
    return df


def aggregate_enhanced_hourly(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    改良版スコアを使って1時間ごとに集計
    
    Args:
        df: calculate_enhanced_negativityで処理済みのDataFrame
        
    Returns:
        DataFrame: 1時間ごとの集計データ
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df[timestamp_col].dt.floor('h')
    
    # 集計
    hourly = df.groupby('hour').agg({
        'content': 'count',
        'bert_negativity': 'mean',
        'keyword_negativity': 'mean',
        'enhanced_negativity': 'mean',
        'enhanced_label': lambda x: (x == 'NEGATIVE').mean(),  # ネガティブ率
    }).reset_index()
    
    hourly.columns = [
        'timestamp', 'count', 
        'bert_negative_rate', 'keyword_negative_rate',
        'enhanced_negative_score', 'enhanced_negative_rate'
    ]
    
    return hourly


def compare_methods(df: pd.DataFrame, content_col: str = 'content') -> dict:
    """
    BERTのみ vs 改良版の比較統計
    """
    stats = {
        'total': len(df),
        'bert_negative': (df['bert_label'] == 'NEGATIVE').sum(),
        'bert_neutral': (df['bert_label'] == 'NEUTRAL').sum(),
        'bert_positive': (df['bert_label'] == 'POSITIVE').sum(),
        'enhanced_negative': (df['enhanced_label'] == 'NEGATIVE').sum(),
        'enhanced_neutral': (df['enhanced_label'] == 'NEUTRAL').sum(),
        'enhanced_positive': (df['enhanced_label'] == 'POSITIVE').sum(),
    }
    
    # NEUTRALからNEGATIVEに変わった数
    bert_neutral = df['bert_label'] == 'NEUTRAL'
    enhanced_negative = df['enhanced_label'] == 'NEGATIVE'
    stats['neutral_to_negative'] = (bert_neutral & enhanced_negative).sum()
    
    return stats


# テスト用
if __name__ == '__main__':
    # サンプルテスト
    test_texts = [
        "松本人志は性加害者だ。許せない。",
        "松本人志の件が報じられた",
        "松本人志面白かった！最高！",
        "松本人志どうなの？微妙だなぁ",
        "やっぱりクソ野郎だった",
    ]
    
    for text in test_texts:
        score = calculate_keyword_negativity(text)
        print(f"Score: {score:.2f} | {text}")
