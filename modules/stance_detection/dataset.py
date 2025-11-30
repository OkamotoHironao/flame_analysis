#!/usr/bin/env python3
"""
Stance Detection Dataset
立場分類用のデータセットクラス
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


# ラベルマッピング
LABEL_MAP = {
    "AGAINST": 0,  # 反対
    "FAVOR": 1,    # 賛成
    "NEUTRAL": 2   # 中立
}


class StanceDataset(Dataset):
    """
    立場分類用のPyTorchデータセット
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: テキストのリスト
            labels: ラベルのリスト (None の場合は推論モード)
            tokenizer: トークナイザー
            max_length: 最大トークン長
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # トークナイズ
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # ラベルがある場合は追加
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def load_dataset_from_csv(csv_path, tokenizer, max_length=128):
    """
    CSVファイルからデータセットを読み込む
    
    Args:
        csv_path: CSVファイルのパス
        tokenizer: トークナイザー
        max_length: 最大トークン長
        
    Returns:
        StanceDataset: データセット
    """
    df = pd.read_csv(csv_path, comment='#')
    
    # content列とlabel列の確認
    if 'content' not in df.columns:
        raise ValueError("CSVファイルに 'content' 列が必要です")
    if 'label' not in df.columns:
        raise ValueError("CSVファイルに 'label' 列が必要です")
    
    # テキストとラベルを抽出
    texts = df['content'].fillna("").tolist()
    labels = [LABEL_MAP[label] for label in df['label']]
    
    return StanceDataset(texts, labels, tokenizer, max_length)
