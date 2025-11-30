"""
Stance Detection Dataset クラス
ツイートの立場分類用のPyTorch Dataset
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class StanceDataset(Dataset):
    """
    立場分類用データセット
    
    Args:
        texts: ツイートテキストのリスト
        labels: ラベルのリスト（"AGAINST", "FAVOR", "NEUTRAL"）
        tokenizer: HuggingFace Tokenizer
        max_length: 最大トークン長
    """
    
    # ラベルマッピング
    LABEL_MAP = {
        "AGAINST": 0,
        "FAVOR": 1,
        "NEUTRAL": 2
    }
    
    REVERSE_LABEL_MAP = {
        0: "AGAINST",
        1: "FAVOR",
        2: "NEUTRAL"
    }
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # トークン化
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        # ラベルがある場合（学習時）
        if self.labels is not None:
            label = self.labels[idx]
            item['labels'] = torch.tensor(self.LABEL_MAP[label], dtype=torch.long)
        
        return item


def load_dataset_from_csv(csv_path, tokenizer, max_length=128):
    """
    CSVファイルからデータセットを読み込む
    
    Args:
        csv_path: CSVファイルパス
        tokenizer: Tokenizer
        max_length: 最大トークン長
        
    Returns:
        StanceDataset
    """
    df = pd.read_csv(csv_path)
    
    if 'content' not in df.columns:
        raise ValueError("CSV must have 'content' column")
    
    texts = df['content'].tolist()
    
    # ラベルがある場合
    if 'label' in df.columns:
        labels = df['label'].tolist()
        return StanceDataset(texts, labels, tokenizer, max_length)
    else:
        return StanceDataset(texts, None, tokenizer, max_length)
