"""
共起語分析をするスクリプト
"""

import pandas as pd
from collections import Counter
import fugashi
import re
import os
from datetime import datetime

# === 設定 ===
CSV_PATH = "original_data/tweets_松本人志_20251112_093317.csv"  # ←あなたのCSVファイル名に合わせる
TOP_N = 30  # 上位何語を表示するか
OUTPUT_DIR = "related_words"  # 出力先フォルダ

# === CSV読み込み ===
df = pd.read_csv(CSV_PATH, comment='#')  # '#'で始まるコメント行をスキップ
print(f"読み込み完了: {len(df)}件")

# テキスト列の推定
if "content" in df.columns:
    texts = df["content"].dropna().tolist()
elif "text" in df.columns:
    texts = df["text"].dropna().tolist()
else:
    texts = df.iloc[:, 0].dropna().tolist()  # 最初の列をテキスト扱い

# === 前処理 ===
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # URL除去
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # メンション除去
    text = re.sub(r"RT ", "", text)  # RT除去
    text = re.sub(r"[\n\t]", " ", text)
    return text

texts = [clean_text(t) for t in texts]

# === 形態素解析 ===
tagger = fugashi.Tagger()
nouns = []
for text in texts:
    nouns += [w.surface for w in tagger(text) if "名詞" in w.feature]

# === 集計 ===
counter = Counter(nouns)
common = counter.most_common(TOP_N)

# === 結果表示 ===
print(f"\n【上位 {TOP_N} 名詞】")
for w, c in common:
    print(f"{w}\t{c}")

# === CSV出力 ===
# 出力フォルダが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# オリジナルデータのファイル名から検索クエリを抽出
# 形式: tweets_[検索クエリ]_[タイムスタンプ].csv
csv_filename = os.path.basename(CSV_PATH)  # ファイル名のみ取得
if csv_filename.startswith("tweets_"):
    # "tweets_"を除去し、".csv"を除去
    name_part = csv_filename[7:-4]  # "tweets_"(7文字)と".csv"(4文字)を除去
    # 最後の"_"と"_"の間（検索クエリ部分）を取得
    # 例: "松本人志_20251112_093317" → "松本人志"
    parts = name_part.split("_")
    # 数字のみの部分（タイムスタンプ）を除去
    search_query = "_".join([p for p in parts if not p.isdigit()])
else:
    # フォーマットが違う場合は、ファイル名をそのまま使用
    search_query = csv_filename[:-4]

# タイムスタンプ付きファイル名を生成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{search_query}_{timestamp}.csv"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# CSVファイルとして保存
pd.DataFrame(common, columns=["word", "count"]).to_csv(output_path, index=False)
print(f"\n→ '{output_path}' に出力しました。")
