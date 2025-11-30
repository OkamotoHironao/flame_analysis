# プロジェクト構造リファレンス

## 📂 完全なディレクトリ構造（2025-11-27更新）

```
flame_analysis/
├── .venv/                              # Python仮想環境
│
├── data/                               # 全データフォルダ
│   ├── original/                       # 元ツイートデータ
│   │   └── tweets_松本人志_20251112_093317.csv
│   ├── processed/                      # 処理済みデータ
│   │   └── stance_train.csv           # 立場分類学習データ
│   └── dictionary/                     # 辞書ファイル
│       ├── pn_ja.csv
│       └── pn_ja.dic
│
├── modules/                            # 機能別モジュール
│   │
│   ├── word_extraction/                # 名詞抽出機能
│   │   ├── extract.py                 # 実行スクリプト
│   │   └── outputs/                   # 出力フォルダ
│   │       └── 松本人志_*.csv
│   │
│   ├── sentiment_analysis/             # 感情分析機能
│   │   ├── dictionary_based/          # 辞書ベース
│   │   │   ├── analyze.py
│   │   │   └── outputs/
│   │   │       ├── 松本人志_analyzed.csv
│   │   │       ├── 松本人志_sentiment_1h.csv
│   │   │       └── ...
│   │   │
│   │   └── bert_based/                 # BERTベース
│   │       ├── analyze.py
│   │       └── outputs/
│   │           ├── 松本人志_bert.csv
│   │           └── ...
│   │
│   ├── stance_detection/              # ⭐立場分類機能
│   │   ├── dataset.py                 # データセットクラス
│   │   ├── train.py                   # 学習スクリプト
│   │   ├── predict.py                 # 推論スクリプト
│   │   ├── model/                     # 学習済みモデル
│   │   │   ├── best_model.pth
│   │   │   ├── config.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   └── vocab.txt
│   │   ├── outputs/                   # 推論結果
│   │   │   └── 松本人志_stance.csv
│   │   └── README.md
│   │
│   └── time_series/                    # 時系列分析機能
│       ├── analyze.py
│       └── outputs/
│           ├── time_series_1h.csv
│           ├── time_series_30m.csv
│           └── time_series_10m.csv
│
├── utils/                              # 共通ユーティリティ
│   └── convert_dict.py
│
├── notebooks/                          # Jupyter Notebook（将来用）
│
├── 旧ファイル（後方互換性のため残存）
│   ├── expand_from_csv.py
│   ├── sentiment_analysis.py
│   ├── bert_sentiment.py
│   ├── stance_dataset.py
│   ├── stance_train.py
│   ├── stance_predict.py
│   ├── time_series_analysis.py
│   └── convert_dict.py
│
├── requirements.txt                    # 依存パッケージ
├── README.md                           # メインREADME
└── README_stance_detection.md          # 立場分類の詳細ドキュメント
```

## 🎯 各モジュールの実行方法

### 1. 名詞抽出
```bash
cd modules/word_extraction
python extract.py
```

### 2. 感情分析（辞書ベース）
```bash
cd modules/sentiment_analysis/dictionary_based
python analyze.py
```

### 3. 感情分析（BERTベース）
```bash
cd modules/sentiment_analysis/bert_based
python analyze.py
```

### 4. 立場分類
```bash
cd modules/stance_detection

# 学習
python train.py ../../data/processed/stance_train.csv

# 推論
python predict.py ../../data/original/tweets_松本人志.csv outputs/stance_result.csv
```

### 5. 時系列分析
```bash
cd modules/time_series
python analyze.py
```

## 📊 実行結果（2025-11-27）

### 立場分類の結果
- **総ツイート数**: 97件
- **NEUTRAL**: 78件 (80.4%)
- **AGAINST**: 18件 (18.6%)
- **FAVOR**: 1件 (1.0%)
- **出力場所**: `modules/stance_detection/outputs/松本人志_stance.csv`

### 学習済みモデル
- **モデル**: cl-tohoku/bert-base-japanese-v3
- **検証精度**: 100.00%
- **保存場所**: `modules/stance_detection/model/`

## 🔄 移行メモ

### 旧構造 → 新構造のマッピング

| 旧ファイル/フォルダ | 新ファイル/フォルダ |
|-------------------|-------------------|
| `original_data/` | `data/original/` |
| `dictionary/` | `data/dictionary/` |
| `expand_from_csv.py` | `modules/word_extraction/extract.py` |
| `related_words/` | `modules/word_extraction/outputs/` |
| `sentiment_analysis.py` | `modules/sentiment_analysis/dictionary_based/analyze.py` |
| `bert_sentiment.py` | `modules/sentiment_analysis/bert_based/analyze.py` |
| `sentiment_analysis/` | `modules/sentiment_analysis/dictionary_based/outputs/` |
| `stance_dataset.py` | `modules/stance_detection/dataset.py` |
| `stance_train.py` | `modules/stance_detection/train.py` |
| `stance_predict.py` | `modules/stance_detection/predict.py` |
| `stance_model/` | `modules/stance_detection/model/` |
| `time_series_analysis.py` | `modules/time_series/analyze.py` |
| `time_series_data/` | `modules/time_series/outputs/` |
| `convert_dict.py` | `utils/convert_dict.py` |

### 注意事項
- 旧ファイルは後方互換性のためプロジェクトルートに残していますが、**新しい開発はすべて `modules/` 配下で実施してください**
- 各モジュールは独立しており、対応するディレクトリ内で実行する必要があります
- パスは相対パスで記述されています（`../../data/original/` など）

## 🚀 今後の開発指針

### 新機能追加時のルール

1. **新しいモジュールの作成**
   ```bash
   mkdir -p modules/new_feature/{outputs,model}
   ```

2. **必須ファイル**
   - スクリプト（処理ロジック）
   - README.md（使い方説明）
   - outputs/ フォルダ（結果保存先）

3. **データの配置**
   - 入力: `data/original/` または `data/processed/`
   - 出力: `modules/<機能名>/outputs/`
   - モデル: `modules/<機能名>/model/`

4. **インポート規則**
   ```python
   # プロジェクトルートをパスに追加
   from pathlib import Path
   import sys
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root))
   
   # 他モジュールのインポート
   from modules.stance_detection.dataset import StanceDataset
   ```

## 📝 チェックリスト

### 新機能追加時
- [ ] `modules/` 配下に機能別フォルダを作成
- [ ] `outputs/` フォルダを作成
- [ ] README.md を作成
- [ ] メインREADME.md を更新
- [ ] このファイル（PROJECT_STRUCTURE.md）を更新

### コード実行時
- [ ] 対応するモジュールディレクトリに移動 (`cd modules/<機能名>`)
- [ ] 相対パスで入力ファイルを指定
- [ ] outputs/ フォルダに結果が保存されることを確認
