# Flame Analysis（炎上検知AIシステム）

X（旧Twitter）の投稿データから「炎上」を自動検知するAIモデル構築プロジェクト。

## 🎯 プロジェクトの目的

**炎上の自動検知**：短期間の投稿急増 × ネガティブ発言増加 × 批判的立場の拡大が同時に起きる現象を、機械学習で検知する。

## 🔬 分析パイプライン

1. **データ収集** - X（Twitter）からツイート収集
2. **前処理** - 時系列ウィンドウ化（1時間単位）
3. **感情分析** - 辞書ベース・BERTベースのネガポジ分析
4. **立場分類** - AGAINST/FAVOR/NEUTRALの3値分類（BERT）
5. **特徴量統合** - 時系列・感情・立場の特徴量を結合
6. **モデル学習** - XGBoost/LightGBMで炎上判定
7. **予測・評価** - 新規データで炎上予測＋SHAP分析

## 📂 ディレクトリ構成

```
flame_analysis/
├── data/                              # データ（元データ・辞書）
│   ├── original/                      # 元ツイートデータ (tweets_*.csv)
│   │   ├── 松本人志/
│   │   ├── 台湾有事/
│   │   └── ...
│   ├── processed/                     # 処理済みデータ
│   │   └── flame_labels.csv          # 炎上ラベル（正解データ）
│   └── dictionary/                    # 感情辞書 (pn_ja.csv)
│
├── modules/                           # 機能別モジュール
│   ├── word_extraction/               # 名詞抽出
│   │   ├── extract.py
│   │   └── outputs/<トピック>/
│   │
│   ├── sentiment_analysis/            # 感情分析
│   │   ├── dictionary_based/          # 辞書ベース
│   │   │   ├── analyze.py
│   │   │   └── outputs/<トピック>/
│   │   └── bert_based/                # BERTベース
│   │       ├── analyze.py
│   │       └── outputs/<トピック>/
│   │
│   ├── stance_detection/              # 立場分類
│   │   ├── dataset.py
│   │   ├── train.py                   # BERT学習
│   │   ├── predict.py                 # 推論
│   │   ├── model/<トピック>/          # 学習済みモデル
│   │   ├── outputs/<トピック>/        # 推論結果
│   │   └── README.md
│   │
│   ├── feature_engineering/           # ⭐特徴量エンジニアリング
│   │   ├── feature_builder.py        # 特徴量統合
│   │   ├── outputs/<トピック>/        # 特徴量テーブル
│   │   └── README.md
│   │
│   ├── flame_detection/               # 🔥炎上検知モデル（作成予定）
│   │   ├── train.py                   # XGBoost学習
│   │   ├── predict.py                 # 炎上予測
│   │   ├── evaluate.py                # モデル評価
│   │   ├── shap_analysis.py           # SHAP特徴量重要度分析
│   │   ├── model/                     # 学習済みモデル
│   │   ├── outputs/                   # 予測結果
│   │   └── README.md
│   │
│   ├── time_series/                   # 時系列分析
│   │   ├── analyze.py
│   │   └── outputs/<トピック>/
│   │
│   └── visualization/                 # 可視化（作成予定）
│       ├── dashboard.py               # Streamlitダッシュボード
│       └── plots.py                   # グラフ生成
│
├── utils/                             # 共通ユーティリティ
│   └── convert_dict.py
│
├── notebooks/                         # Jupyter Notebook（実験用）
│
├── requirements.txt
└── README.md                          # このファイル
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. データ準備

元ツイートデータを `data/original/<トピック名>/` に配置:
```
data/original/松本人志/tweets_松本人志_20251112_093317.csv
data/original/台湾有事/tweets_台湾有事_20251120_140000.csv
```

### 3. 完全な分析パイプライン

#### Step 1: 感情分析（辞書ベース + 時系列集計）
```bash
cd modules/sentiment_analysis/dictionary_based
python analyze.py
# 出力: outputs/<トピック>/<トピック>_sentiment_1h.csv
```

#### Step 2: 立場分類（BERT推論）
```bash
cd modules/stance_detection
python predict.py \
  ../../data/original/松本人志/tweets_松本人志.csv \
  outputs/松本人志/松本人志_stance.csv
```

#### Step 3: 特徴量統合
```bash
cd modules/feature_engineering
python feature_builder.py \
  --sentiment_csv ../sentiment_analysis/dictionary_based/outputs/松本人志_sentiment_1h.csv \
  --stance_csv ../stance_detection/outputs/松本人志/松本人志_stance.csv
# 出力: outputs/松本人志/松本人志_feature_table.csv
```

#### Step 4: 炎上検知モデル学習（作成予定）
```bash
cd modules/flame_detection
python train.py \
  --feature_csv ../feature_engineering/outputs/松本人志/松本人志_feature_table.csv \
  --label_csv ../../data/processed/flame_labels.csv
```

#### Step 5: 炎上予測（作成予定）
```bash
python predict.py \
  --feature_csv ../feature_engineering/outputs/新規トピック/新規トピック_feature_table.csv
```

## � データフォーマット

### 入力CSVフォーマット（ツイートデータ）

```csv
date,id,url,content,user,user_displayname,reply_count,retweet_count,like_count
2025-11-02T12:52:35.000Z,1984966871750410536,https://...,ツイート本文,username,表示名,0,0,0
```

**必須カラム**: `content` (ツイート本文), `date` または `created_at` (投稿日時)

### 炎上ラベルフォーマット（正解データ）

```csv
topic,timestamp,is_flame,flame_level
松本人志,2025-11-02 12:00:00,1,high
松本人志,2025-11-02 13:00:00,1,high
松本人志,2025-11-02 14:00:00,0,none
台湾有事,2025-11-20 15:00:00,1,medium
```

**カラム説明**:
- `topic`: トピック名
- `timestamp`: 時刻（1時間単位）
- `is_flame`: 炎上フラグ（1=炎上, 0=非炎上）
- `flame_level`: 炎上レベル（high/medium/low/none）

### 特徴量テーブルフォーマット

```csv
timestamp,volume,negative_rate,stance_against_rate,delta_volume,delta_volume_rate,...
2025-11-02 12:00:00,25,0.8,0.6,15,1.5,...
```

**主要特徴量**:
- `volume`: 投稿数
- `negative_rate`: ネガティブ率
- `stance_against_rate`: 批判的立場の割合
- `delta_volume`: 投稿数増加量
- `delta_volume_rate`: 投稿数変化率

## 🔧 各モジュールの詳細

### 感情分析 (sentiment_analysis)

#### 辞書ベース
- **辞書**: pn_ja.csv (17,838語)
- **出力**: negative_rate, negative_intensity, intensity_level, sentiment_1h.csv
- **特徴**: 高速、説明可能

#### BERTベース
- **モデル**: koheiduck/bert-japanese-finetuned-sentiment
- **出力**: bert_label, bert_positive, bert_neutral, bert_negative
- **特徴**: 文脈理解、高精度

### 立場分類 (stance_detection)
- **モデル**: cl-tohoku/bert-base-japanese-v3（ファインチューニング）
- **クラス**: AGAINST / FAVOR / NEUTRAL
- **出力**: stance_label, stance_against, stance_favor, stance_neutral
- **特徴**: カスタムデータでファインチューニング可能、GPU対応

### 特徴量エンジニアリング (feature_engineering)
- **入力**: 感情分析時系列CSV + 立場分類推論CSV
- **処理**: 1時間単位で集計・結合
- **出力特徴量**:
  - 基本: volume, negative_rate, stance_against_rate, sentiment_avg_score
  - 差分: delta_volume, delta_negative_rate, delta_against_rate
  - 変化率: delta_volume_rate
  - 確率平均: stance_against_mean, stance_favor_mean, stance_neutral_mean

### 炎上検知 (flame_detection) 【作成予定】
- **モデル**: XGBoost / LightGBM
- **タスク**: 炎上/非炎上の2値分類
- **評価指標**: Precision, Recall, F1-Score, ROC-AUC
- **特徴量重要度**: SHAP分析

### 時系列分析 (time_series)
- **集計単位**: 1時間 / 30分 / 10分
- **出力**: ツイート頻度の時間変化
- **用途**: 炎上タイミングの特定、投稿急増検知

## 💡 使用例

### 完全な分析パイプライン

```bash
# 1. 名詞抽出
cd modules/word_extraction && python extract.py && cd ../..

# 2. 感情分析（BERT）
cd modules/sentiment_analysis/bert_based && python analyze.py && cd ../../..

# 3. 立場分類（学習済みモデルで推論）
cd modules/stance_detection
python predict.py ../../data/original/tweets_松本人志.csv outputs/stance_result.csv
cd ../..

# 4. 時系列分析
cd modules/time_series && python analyze.py && cd ../..
```

### 新しい立場分類モデルの学習

```bash
# 学習データを準備
# data/processed/my_stance_data.csv に AGAINST/FAVOR/NEUTRAL ラベル付きデータ

# 学習実行
cd modules/stance_detection
python train.py ../../data/processed/my_stance_data.csv

# 推論
python predict.py ../../data/original/new_tweets.csv outputs/predictions.csv
```

## 🛠️ トラブルシューティング

### CUDA Out of Memory
```bash
# バッチサイズを減らす
python train.py data.csv 8  # バッチサイズ=8
```

### MeCab辞書エラー
```bash
pip install unidic-lite
```

### モジュールが見つからない
```bash
# 各スクリプトは対応するモジュールディレクトリ内で実行してください
cd modules/stance_detection
python predict.py ...
```

## 📊 出力ファイル一覧

| 機能 | 出力場所 | ファイル例 |
|------|---------|-----------|
| 名詞抽出 | `modules/word_extraction/outputs/` | `松本人志_20251127.csv` |
| 感情分析（辞書） | `modules/sentiment_analysis/dictionary_based/outputs/` | `松本人志_analyzed.csv` |
| 感情分析（BERT） | `modules/sentiment_analysis/bert_based/outputs/` | `松本人志_bert.csv` |
| 立場分類 | `modules/stance_detection/outputs/` | `stance_predictions.csv` |
| 時系列分析 | `modules/time_series/outputs/` | `time_series_1h.csv` |

## 📝 ライセンス

教育・研究目的で自由に使用できます。

## 🔄 バージョン履歴

- **v3.0.0** (2025-11-28) - 炎上検知AIシステムとして再設計
  - 特徴量エンジニアリングモジュール追加
  - トピック別管理対応
  - 炎上検知モデル開発に向けた基盤整備
  
- **v2.0.0** (2025-11-27) - モジュール構造へリファクタリング
  - 立場分類機能追加（BERT fine-tuning）
  - 機能別ディレクトリ構成
  
- **v1.0.0** - 初版リリース（名詞抽出、感情分析、時系列分析）

## 🎯 今後の開発ロードマップ

### Phase 1: 炎上検知モデル構築（進行中）
- [x] 特徴量統合スクリプト
- [ ] 炎上ラベル作成ツール
- [ ] XGBoost学習スクリプト
- [ ] モデル評価・SHAP分析

### Phase 2: 予測パイプライン
- [ ] 新規ツイートの自動分析
- [ ] リアルタイム炎上予測
- [ ] アラート機能

### Phase 3: 可視化・ダッシュボード
- [ ] Streamlitダッシュボード
- [ ] 時系列グラフ可視化
- [ ] 特徴量重要度の可視化

### Phase 4: 精度向上
- [ ] アンサンブルモデル
- [ ] ハイパーパラメータ最適化
- [ ] 追加特徴量の検討
