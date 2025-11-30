#!/usr/bin/env python3
"""
Stance Detection 推論スクリプト
学習済みモデルを使ってツイートの立場を分類
"""

import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import json

from stance_dataset import StanceDataset


# モデル定義（train.pyと同じ）
import torch.nn as nn
from transformers import AutoModel

class StanceClassifier(nn.Module):
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


def load_model(model_dir, device):
    """
    学習済みモデルを読み込む
    """
    # 設定読み込み
    config_path = Path(model_dir) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # モデル構築
    model = StanceClassifier(
        config['model_name'],
        num_labels=config['num_labels'],
        dropout=config['dropout']
    )
    
    # 重み読み込み
    model.load_state_dict(torch.load(
        Path(model_dir) / "best_model.pth",
        map_location=device
    ))
    model.to(device)
    model.eval()
    
    return model, config


def predict(model, dataloader, device):
    """
    推論実行
    """
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 推論中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return all_predictions, all_probabilities


def main():
    if len(sys.argv) != 3:
        print("使用法: python stance_predict.py <input_csv> <output_csv>")
        print("例: python stance_predict.py data/test.csv results/predictions.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=" * 60)
    print("Stance Detection 推論スクリプト")
    print("=" * 60)
    
    MODEL_DIR = "modules/stance_detection/model"
    BATCH_SIZE = 32
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  デバイス: {device}")
    
    # 入力ファイル確認
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)
    
    # モデルディレクトリ確認
    if not Path(MODEL_DIR).exists():
        print(f"❌ エラー: モデルが見つかりません: {MODEL_DIR}")
        print("先に stance_train.py で学習を実行してください。")
        sys.exit(1)
    
    try:
        # モデル読み込み
        print(f"\n🤖 モデル読み込み中: {MODEL_DIR}")
        model, config = load_model(MODEL_DIR, device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("✓ モデル読み込み完了")
        
        # データ読み込み
        print(f"\n📂 データ読み込み中: {input_path}")
        df = pd.read_csv(input_path, comment='#')
        print(f"✓ {len(df)}件のデータを読み込みました")
        
        # テキスト列の確認
        if 'content' not in df.columns:
            if 'text' in df.columns:
                df['content'] = df['text']
            else:
                raise ValueError("❌ 'content' または 'text' 列が見つかりません")
        
        # データセット作成
        texts = df['content'].fillna("").tolist()
        dataset = StanceDataset(texts, None, tokenizer, config['max_length'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
        
        # 推論実行
        print(f"\n🔍 推論実行中...")
        predictions, probabilities = predict(model, dataloader, device)
        
        # ラベルマッピング
        LABEL_MAP = {0: "AGAINST", 1: "FAVOR", 2: "NEUTRAL"}
        
        # 結果をDataFrameに追加
        df['stance_label'] = [LABEL_MAP[pred] for pred in predictions]
        df['stance_against'] = [prob[0] for prob in probabilities]
        df['stance_favor'] = [prob[1] for prob in probabilities]
        df['stance_neutral'] = [prob[2] for prob in probabilities]
        
        # 統計情報
        label_counts = df['stance_label'].value_counts()
        print(f"\n✓ 推論完了:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {label}: {count}件 ({percentage:.1f}%)")
        
        # 結果保存
        output_dir = Path(output_path).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 結果を保存中: {output_path}")
        df.to_csv(output_path, index=False)
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


# 実行例
# python stance_predict.py original_data/tweets_松本人志_20251112_093317.csv results/stance_predictions.csv
