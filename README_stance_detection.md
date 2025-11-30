# 立場分類（Stance Detection）モデル

ツイートの立場を **AGAINST（反対）/ FAVOR（賛成）/ NEUTRAL（中立）** に分類するBERTベースのモデルです。

## 🎯 特徴

- **モデル**: `cl-tohoku/bert-base-japanese-v3`（日本語BERT）
- **3クラス分類**: AGAINST / FAVOR / NEUTRAL
- **GPU対応**: CUDA自動検出で高速推論
- **混合精度学習**: メモリ効率化とスピードアップ
- **カスタマイズ可能**: ドロップアウト率、学習率、バッチサイズなど調整可能

## 📦 インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

**注意**: PyTorchのインストールはCUDAバージョンに応じて調整してください。
- CUDA 11.8の場合: `requirements.txt`のまま
- CPUのみの場合: https://pytorch.org/ を参照

## 📂 ファイル構成

```
.
├── stance_dataset.py      # データセットクラス
├── stance_train.py        # 学習スクリプト
├── stance_predict.py      # 推論スクリプト
├── requirements.txt       # 依存パッケージ
└── README.md              # このファイル
```

## 🚀 使い方

### 1. 学習データの準備

CSVファイルを用意します（例: `data/train.csv`）:

```csv
content,label
このサービスは素晴らしい！,FAVOR
問題が多すぎて使えない,AGAINST
普通だと思います,NEUTRAL
```

**必須カラム**:
- `content`: ツイート本文
- `label`: `AGAINST`, `FAVOR`, `NEUTRAL` のいずれか

**コメント行対応**: `#`で始まる行は無視されます

### 2. モデルの学習

```bash
python stance_train.py data/train.csv
```

**オプション引数**:
```bash
python stance_train.py data/train.csv \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1
```

**学習結果**:
- モデル: `stance_model/best_model.pth`
- トークナイザー: `stance_model/`
- 設定: `stance_model/config.json`

**テスト実行**: 学習データがない場合、サンプルデータで動作確認できます:
```bash
python stance_train.py sample_data.csv
```
→ 自動的にサンプルデータを生成して学習開始

### 3. 推論（予測）

```bash
python stance_predict.py input.csv output.csv
```

**入力CSV例** (`input.csv`):
```csv
content
このニュースに賛成です
反対意見が多いですね
どちらとも言えない
```

**出力CSV例** (`output.csv`):
```csv
content,stance_label,stance_against,stance_favor,stance_neutral
このニュースに賛成です,FAVOR,0.05,0.92,0.03
反対意見が多いですね,AGAINST,0.89,0.06,0.05
どちらとも言えない,NEUTRAL,0.10,0.15,0.75
```

**出力カラム**:
- `stance_label`: 予測ラベル（AGAINST/FAVOR/NEUTRAL）
- `stance_against`: AGAINST確率（0〜1）
- `stance_favor`: FAVOR確率（0〜1）
- `stance_neutral`: NEUTRAL確率（0〜1）

## ⚙️ ハイパーパラメータ

### デフォルト設定

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `model_name` | `cl-tohoku/bert-base-japanese-v3` | ベースモデル |
| `max_length` | 128 | 最大トークン長 |
| `dropout` | 0.3 | ドロップアウト率 |
| `batch_size` | 16 | バッチサイズ |
| `epochs` | 3 | エポック数 |
| `learning_rate` | 2e-5 | 学習率 |
| `warmup_ratio` | 0.1 | ウォームアップ比率 |

### カスタマイズ例

```python
# stance_train.py の main() 内で変更
BATCH_SIZE = 32      # バッチサイズを大きく（GPU使用時）
EPOCHS = 5           # エポック数を増やす
LEARNING_RATE = 3e-5 # 学習率を調整
```

## 🔧 トラブルシューティング

### CUDA Out of Memory エラー
```bash
# バッチサイズを減らす
python stance_train.py data/train.csv --batch_size 8
```

### モデルが見つからない
```bash
# 学習が完了しているか確認
ls stance_model/
# → best_model.pth, config.json, tokenizer_config.json など
```

### 精度が低い
- **学習データを増やす**: 各クラス最低100件以上推奨
- **エポック数を増やす**: `--epochs 5` など
- **学習率を調整**: `--learning_rate 1e-5` など

## 📊 モデル詳細

### アーキテクチャ

```
Input Text
    ↓
BERT Tokenizer (cl-tohoku/bert-base-japanese-v3)
    ↓
BERT Encoder (768次元)
    ↓
Dropout (0.3)
    ↓
Linear Classifier (768 → 3)
    ↓
Softmax
    ↓
[AGAINST, FAVOR, NEUTRAL]
```

### 学習詳細

- **最適化**: AdamW
- **スケジューラ**: Linear Warmup
- **損失関数**: CrossEntropyLoss
- **混合精度**: `torch.cuda.amp.autocast()`
- **評価指標**: Accuracy
- **Early Stopping**: ベストモデル保存

## 📝 ライセンス

このコードは教育・研究目的で自由に使用できます。

## 🤝 貢献

バグ報告や機能追加の提案を歓迎します。

## 📞 サポート

問題が発生した場合は、以下を確認してください:
1. Python 3.10以上がインストールされているか
2. 必要なパッケージがインストールされているか（`requirements.txt`）
3. CUDAドライバーが正しくインストールされているか（GPU使用時）
