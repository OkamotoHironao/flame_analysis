#!/usr/bin/env python3
"""
Stance Detection 学習スクリプト
日本語BERTを使った立場分類モデルのファインチューニング
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from stance_dataset import StanceDataset, load_dataset_from_csv


# シード固定
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Stance Detection モデル
class StanceClassifier(nn.Module):
    """
    BERT + Classification Head
    """
    def __init__(self, model_name, num_labels=3, dropout=0.3):
        super(StanceClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]トークンの出力
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """
    1エポックの学習
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度学習
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 統計
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """
    評価
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("Stance Detection 学習スクリプト")
    print("=" * 60)
    
    # ハイパーパラメータ
    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    DROPOUT = 0.3
    TRAIN_RATIO = 0.8
    SEED = 42
    MODEL_DIR = "stance_model"
    
    # データファイル（仮のパス - 実際のデータに合わせて変更）
    TRAIN_CSV = "data/stance_train.csv"
    
    # シード固定
    set_seed(SEED)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  デバイス: {device}")
    
    # 混合精度学習の設定
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("⚡ 混合精度学習を有効化")
    
    # トークナイザーとモデルの読み込み
    print(f"\n🤖 モデル読み込み中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = StanceClassifier(MODEL_NAME, num_labels=3, dropout=DROPOUT)
    model.to(device)
    print("✓ モデル読み込み完了")
    
    # データ読み込み
    print(f"\n📂 データ読み込み中: {TRAIN_CSV}")
    
    # データが存在しない場合のサンプルデータ生成
    if not Path(TRAIN_CSV).exists():
        print("⚠️  学習データが見つかりません。サンプルデータを作成します...")
        Path("data").mkdir(exist_ok=True)
        
        sample_data = {
            'content': [
                '松本人志は最低だ。許せない。',
                '性加害は絶対に許されない行為だ。',
                '松本人志を支持します。頑張ってほしい。',
                '松本人志の才能は素晴らしい。',
                '松本人志が復帰したというニュースを見た。',
                '事実関係を冷静に見るべきだ。',
            ] * 20,  # 120件のサンプル
            'label': ['AGAINST', 'AGAINST', 'FAVOR', 'FAVOR', 'NEUTRAL', 'NEUTRAL'] * 20
        }
        pd.DataFrame(sample_data).to_csv(TRAIN_CSV, index=False)
        print(f"✓ サンプルデータを作成: {TRAIN_CSV}")
    
    # データセット作成
    full_dataset = load_dataset_from_csv(TRAIN_CSV, tokenizer, MAX_LENGTH)
    print(f"✓ データ件数: {len(full_dataset)}件")
    
    # Train/Val分割
    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"  - 訓練データ: {train_size}件")
    print(f"  - 検証データ: {val_size}件")
    
    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 学習ループ
    print(f"\n🚀 学習開始（{EPOCHS}エポック）")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\n【エポック {epoch + 1}/{EPOCHS}】")
        
        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        print(f"訓練 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # 評価
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"検証 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✨ 新しいベストモデル（Acc: {val_acc:.2f}%）")
            
            # モデル保存
            Path(MODEL_DIR).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
            tokenizer.save_pretrained(MODEL_DIR)
            
            # 設定保存
            config = {
                'model_name': MODEL_NAME,
                'max_length': MAX_LENGTH,
                'dropout': DROPOUT,
                'num_labels': 3
            }
            import json
            with open(f"{MODEL_DIR}/config.json", 'w') as f:
                json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ 学習完了！ ベスト精度: {best_val_acc:.2f}%")
    print(f"📁 モデル保存先: {MODEL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
