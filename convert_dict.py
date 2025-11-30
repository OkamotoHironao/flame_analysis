"""
pn_ja.dic を CSV形式に変換するスクリプト
"""

import pandas as pd

# 辞書ファイルを読み込む（タブ区切り、cp932エンコーディング）
try:
    # フォーマット: 単語:読み:品詞:極性値
    data = []
    
    with open('dictionary/pn_ja.dic', 'r', encoding='cp932') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(':')
            if len(parts) >= 4:
                word = parts[0]
                polarity_score = float(parts[3])
                
                # 極性値を -1, 0, 1 に変換
                # 正の値（> 0.5）→ 1、負の値（< -0.5）→ -1、それ以外 → 0
                if polarity_score > 0.5:
                    polarity = 1
                elif polarity_score < -0.5:
                    polarity = -1
                else:
                    polarity = 0
                
                # 中立（0）は除外して、ポジティブ・ネガティブのみ保存
                if polarity != 0:
                    data.append({'word': word, 'polarity': polarity})
    
    # DataFrameに変換
    df = pd.DataFrame(data)
    
    # CSV保存
    df.to_csv('dictionary/pn_ja.csv', index=False)
    
    print(f"✅ 変換完了!")
    print(f"総語彙数: {len(df)}語")
    print(f"ポジティブ: {(df['polarity'] == 1).sum()}語")
    print(f"ネガティブ: {(df['polarity'] == -1).sum()}語")
    print(f"保存先: dictionary/pn_ja.csv")
    
except Exception as e:
    print(f"❌ エラー: {e}")
    
    # UTF-8で試す
    print("\nUTF-8で再試行中...")
    try:
        data = []
        with open('dictionary/pn_ja.dic', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(':')
                if len(parts) >= 4:
                    word = parts[0]
                    polarity_score = float(parts[3])
                    
                    if polarity_score > 0.5:
                        polarity = 1
                    elif polarity_score < -0.5:
                        polarity = -1
                    else:
                        polarity = 0
                    
                    if polarity != 0:
                        data.append({'word': word, 'polarity': polarity})
        
        df = pd.DataFrame(data)
        df.to_csv('dictionary/pn_ja.csv', index=False)
        
        print(f"✅ 変換完了!")
        print(f"総語彙数: {len(df)}語")
        print(f"ポジティブ: {(df['polarity'] == 1).sum()}語")
        print(f"ネガティブ: {(df['polarity'] == -1).sum()}語")
        
    except Exception as e2:
        print(f"❌ UTF-8でも失敗: {e2}")
