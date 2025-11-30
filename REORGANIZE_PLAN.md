# プロジェクト再構成計画

## 新しいディレクトリ構成

```
flame_analysis/
├── data/                           # 全データの共通フォルダ
│   ├── original/                   # 元データ（tweets_*.csv）
│   ├── processed/                  # 処理済みデータ
│   └── dictionary/                 # 辞書ファイル
│
├── modules/                        # 各機能モジュール
│   ├── word_extraction/           # 名詞抽出機能
│   │   ├── extract.py
│   │   └── outputs/
│   │
│   ├── sentiment_analysis/        # 感情分析機能
│   │   ├── dictionary_based/     # 辞書ベース
│   │   │   ├── analyze.py
│   │   │   └── outputs/
│   │   └── bert_based/            # BERTベース
│   │       ├── analyze.py
│   │       └── outputs/
│   │
│   ├── stance_detection/          # 立場分類機能
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── model/                 # 学習済みモデル
│   │   └── outputs/               # 推論結果
│   │
│   └── time_series/               # 時系列分析機能
│       ├── analyze.py
│       └── outputs/
│
├── utils/                          # 共通ユーティリティ
│   └── convert_dict.py
│
├── notebooks/                      # Jupyter Notebook（分析用）
│
├── requirements.txt
└── README.md
```

## 移行マッピング

### スクリプト
- expand_from_csv.py → modules/word_extraction/extract.py
- sentiment_analysis.py → modules/sentiment_analysis/dictionary_based/analyze.py
- bert_sentiment.py → modules/sentiment_analysis/bert_based/analyze.py
- stance_dataset.py → modules/stance_detection/dataset.py
- stance_train.py → modules/stance_detection/train.py
- stance_predict.py → modules/stance_detection/predict.py
- time_series_analysis.py → modules/time_series/analyze.py
- convert_dict.py → utils/convert_dict.py

### データディレクトリ
- original_data/ → data/original/
- dictionary/ → data/dictionary/
- related_words/ → modules/word_extraction/outputs/
- sentiment_analysis/ → modules/sentiment_analysis/dictionary_based/outputs/
- time_series_data/ → modules/time_series/outputs/
- stance_model/ → modules/stance_detection/model/

### 新規データ用
- BERT感情分析の出力 → modules/sentiment_analysis/bert_based/outputs/
- 立場分類の出力 → modules/stance_detection/outputs/
