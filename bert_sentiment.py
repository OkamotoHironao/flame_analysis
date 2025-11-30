#!/usr/bin/env python3
"""
日本語BERTによる感情分析スクリプト
HuggingFace の daigo/bert-base-japanese-sentiment を使用してツイートを分析する
"""

import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_model():
    """
    BERTモデルとトークナイザーを読み込む
    GPU が利用可能な場合は自動的に使用する
    
    Returns:
        tuple: (tokenizer, model, device)
    """
    # 日本語BERTモデル（東北大学）+ 感情分析用にファインチューニング
    # 注: 実際の感情分析には専用モデルまたはファインチューニングが必要
    # ここでは koheiduck/bert-japanese-finetuned-sentiment を使用
    model_name = "koheiduck/bert-japanese-finetuned-sentiment"
    
    print(f"🤖 モデル読み込み中: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # GPU利用可能性チェック
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"✓ モデル読み込み完了")
        print(f"✓ デバイス: {device}")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        sys.exit(1)


def preprocess_text(text):
    """
    テキストの前処理
    
    Args:
        text: 入力テキスト
        
    Returns:
        str: 前処理済みテキスト
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # 連続する空白を1つに
    text = " ".join(text.split())
    
    return text


def analyze_with_bert(texts, tokenizer, model, device, batch_size=32, 
                      threshold_config=None):
    """
    BERTモデルでバッチ推論を実行
    
    Args:
        texts: テキストのリスト
        tokenizer: トークナイザー
        model: BERTモデル
        device: 使用デバイス
        batch_size: バッチサイズ
        threshold_config: 閾値設定の辞書（オプション）
            例: {
                'negative_threshold': 0.5,  # ネガティブと判定する最低確率
                'positive_threshold': 0.5,  # ポジティブと判定する最低確率
                'use_threshold': True       # 閾値を使用するか
            }
        
    Returns:
        list: 各テキストの分析結果 [{"label": str, "positive": float, "neutral": float, "negative": float}, ...]
    """
    results = []
    
    # デフォルト閾値設定
    if threshold_config is None:
        threshold_config = {
            'negative_threshold': 0.4,  # この確率以上でネガティブ
            'positive_threshold': 0.4,  # この確率以上でポジティブ
            'use_threshold': False      # デフォルトは最大確率方式
        }
    
    # ラベルマッピング（モデル依存）
    id2label = model.config.id2label
    
    # バッチ処理
    for i in tqdm(range(0, len(texts), batch_size), desc="🔍 感情分析中"):
        batch_texts = texts[i:i + batch_size]
        
        # トークン化
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # デバイスに転送
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 結果を解析
        for prob in probs:
            prob_dict = {id2label[i]: prob[i].item() for i in range(len(prob))}
            
            # ラベル決定
            if threshold_config['use_threshold']:
                # 閾値ベースの判定
                negative_prob = prob_dict.get('NEGATIVE', 0.0)
                positive_prob = prob_dict.get('POSITIVE', 0.0)
                
                if negative_prob >= threshold_config['negative_threshold']:
                    predicted_label = 'NEGATIVE'
                elif positive_prob >= threshold_config['positive_threshold']:
                    predicted_label = 'POSITIVE'
                else:
                    predicted_label = 'NEUTRAL'
            else:
                # 最大確率方式（デフォルト）
                predicted_label = max(prob_dict, key=prob_dict.get)
            
            result = {
                "label": predicted_label,
                "positive": prob_dict.get("positive", 0.0),
                "neutral": prob_dict.get("neutral", 0.0),
                "negative": prob_dict.get("negative", 0.0)
            }
            results.append(result)
    
    return results


def process_dataframe(df, tokenizer, model, device, threshold_config=None):
    """
    DataFrameに対して感情分析を実行
    
    Args:
        df: 入力DataFrame
        tokenizer: トークナイザー
        model: BERTモデル
        device: 使用デバイス
        threshold_config: 閾値設定（オプション）
        
    Returns:
        pd.DataFrame: 感情分析結果が追加されたDataFrame
    """
    # テキスト列の特定
    if "content" in df.columns:
        text_column = "content"
    elif "text" in df.columns:
        text_column = "text"
    else:
        raise ValueError("❌ 'content' または 'text' 列が見つかりません")
    
    print(f"📝 テキスト列: '{text_column}'")
    print(f"📊 データ件数: {len(df)}件")
    
    # 閾値設定の表示
    if threshold_config and threshold_config.get('use_threshold'):
        print(f"⚙️  閾値設定:")
        print(f"  - ネガティブ閾値: {threshold_config['negative_threshold']}")
        print(f"  - ポジティブ閾値: {threshold_config['positive_threshold']}")
    else:
        print(f"⚙️  判定方式: 最大確率方式")
    
    # 前処理
    texts = df[text_column].apply(preprocess_text).tolist()
    
    # BERT推論
    results = analyze_with_bert(texts, tokenizer, model, device, 
                               threshold_config=threshold_config)
    
    # 結果をDataFrameに追加
    df["bert_label"] = [r["label"] for r in results]
    df["bert_positive"] = [r["positive"] for r in results]
    df["bert_neutral"] = [r["neutral"] for r in results]
    df["bert_negative"] = [r["negative"] for r in results]
    
    # 統計情報
    label_counts = df["bert_label"].value_counts()
    print(f"\n✓ 感情分析完了:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {label}: {count}件 ({percentage:.1f}%)")
    
    return df


def main():
    """
    メイン処理
    """
    if len(sys.argv) != 3:
        print("使用法: python bert_sentiment.py <input_csv> <output_csv>")
        print("例: python bert_sentiment.py original_data/tweets_xxx.csv output/bert_xxx.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=" * 60)
    print("日本語BERT感情分析スクリプト")
    print("=" * 60)
    
    # === 閾値設定（ここで変更可能）===================================
    # use_threshold: False → 最大確率方式（デフォルト）
    # use_threshold: True  → 閾値ベース方式
    threshold_config = {
        'negative_threshold': 0.4,  # ネガティブと判定する最低確率（0.0-1.0）
        'positive_threshold': 0.4,  # ポジティブと判定する最低確率（0.0-1.0）
        'use_threshold': False      # True=閾値使用, False=最大確率
    }
    # ============================================================
    
    # 入力ファイル確認
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. モデル読み込み
        tokenizer, model, device = load_model()
        
        # 2. CSV読み込み
        print(f"\n📂 データ読み込み中: {input_path}")
        df = pd.read_csv(input_path, comment='#')
        print(f"✓ {len(df)}件のデータを読み込みました")
        
        # 3. 感情分析実行
        df_analyzed = process_dataframe(df, tokenizer, model, device, 
                                        threshold_config=threshold_config)
        
        # 4. 結果保存
        print(f"\n💾 結果を保存中: {output_path}")
        df_analyzed.to_csv(output_path, index=False)
        print(f"✓ 保存完了")
        
        print("\n" + "=" * 60)
        print("✅ すべての処理が正常に完了しました！")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ エラー: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# 実行例
# python bert_sentiment.py original_data/tweets_松本人志_20251112_093317.csv sentiment_analysis/松本人志_bert.csv
# python bert_sentiment.py original_data/tweets_xxx.csv output/bert_xxx.csv
