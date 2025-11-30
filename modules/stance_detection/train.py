#!/usr/bin/env python3
"""
Stance Detection 学習スクリプト
日本語BERTを使った立場分類モデルのファインチューニング
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import json

from modules.stance_detection.dataset import StanceDataset, load_dataset_from_csv, LABEL_MAP


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
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """1エポックの学習"""
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
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, device):
    """評価"""
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
    
    return total_loss / len(dataloader), 100 * correct / total


def create_sample_data(output_path):
    """サンプルデータ生成"""
    samples = [
        ("このサービスは素晴らしい！おすすめです", "FAVOR"),
        ("最高の体験でした！", "FAVOR"),
        ("とても良い商品だと思います", "FAVOR"),
        ("完璧な対応でした", "FAVOR"),
        ("問題が多すぎる。使えない", "AGAINST"),
        ("最悪のサービスだ", "AGAINST"),
        ("二度と利用したくない", "AGAINST"),
        ("ひどい対応でした", "AGAINST"),
        ("普通だと思います", "NEUTRAL"),
        ("特に問題はない", "NEUTRAL"),
        ("可もなく不可もなく", "NEUTRAL"),
        ("まあまあです", "NEUTRAL"),
    ] * 10
    
    df = pd.DataFrame(samples, columns=["content", "label"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ サンプルデータを作成: {output_path}")


def main():
    """メイン処理"""
    # スクリプトのディレクトリに移動
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if len(sys.argv) < 2:
        print("使用法: python train.py <train_csv>")
        print("例: python train.py ../../data/processed/stance_train.csv")
        sys.exit(1)
    
    TRAIN_CSV = sys.argv[1]
    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    MAX_LENGTH = 128
    BATCH_SIZE = 16 if len(sys.argv) < 3 else int(sys.argv[2])
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    DROPOUT = 0.3
    MODEL_DIR = "model"
    
    print("=" * 60)
    print("Stance Detection 学習スクリプト")
    print("=" * 60)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  デバイス: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("⚡ 混合精度学習を有効化")
    
    # モデルとトークナイザー読み込み
    print(f"\n🤖 モデル読み込み中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = StanceClassifier(MODEL_NAME, num_labels=3, dropout=DROPOUT)
    model.to(device)
    print("✓ モデル読み込み完了")
    
    # データ読み込み
    print(f"\n📂 データ読み込み中: {TRAIN_CSV}")
    if not Path(TRAIN_CSV).exists():
        print("⚠️  学習データが見つかりません。サンプルデータを作成します...")
        create_sample_data(TRAIN_CSV)
    
    dataset = load_dataset_from_csv(TRAIN_CSV, tokenizer, MAX_LENGTH)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"✓ データ件数: {len(dataset)}件")
    print(f"  - 訓練データ: {train_size}件")
    print(f"  - 検証データ: {val_size}件")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
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
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        print(f"訓練 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"検証 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
            tokenizer.save_pretrained(MODEL_DIR)
            
            config = {
                "model_name": MODEL_NAME,
                "num_labels": 3,
                "max_length": MAX_LENGTH,
                "dropout": DROPOUT
            }
            with open(f"{MODEL_DIR}/config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✨ 新しいベストモデル（Acc: {best_val_acc:.2f}%）")
    
    print("\n" + "=" * 60)
    print(f"✅ 学習完了！ ベスト精度: {best_val_acc:.2f}%")
    print(f"📁 モデル保存先: {MODEL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
