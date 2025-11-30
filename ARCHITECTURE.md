# 炎上検知AIシステム - アーキテクチャドキュメント

## 🎯 システム概要

**目的**: X（旧Twitter）の投稿データから「炎上」を自動検知するAIモデルの構築

**炎上の定義**: 短期間の投稿急増 × ネガティブ発言増加 × 批判的立場の拡大が同時に起きる現象

## 📐 システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     データ収集層                              │
│  X (Twitter) API → CSV保存 (data/original/<トピック>/)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     前処理・分析層                            │
├─────────────────────────────────────────────────────────────┤
│  1. 時系列分析 (time_series)                                 │
│     - 1時間単位で集計                                         │
│     - 投稿量・急増率の計算                                     │
│                                                              │
│  2. 感情分析 (sentiment_analysis)                            │
│     - 辞書ベース: pn_ja.csv（高速）                          │
│     - BERTベース: 文脈理解（高精度）                          │
│     → negative_rate, sentiment_score                        │
│                                                              │
│  3. 立場分類 (stance_detection)                              │
│     - BERT (cl-tohoku/bert-base-japanese-v3)                │
│     - Fine-tuning可能                                        │
│     → stance_against_rate, stance_favor_rate               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  特徴量エンジニアリング層                      │
│  feature_builder.py                                         │
│  - 時系列データ結合（timestamp基準）                          │
│  - 差分特徴量生成（delta_volume, delta_negative_rate）       │
│  - 変化率計算（delta_volume_rate）                           │
│  → feature_table.csv                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    機械学習モデル層                           │
│  flame_detection/                                           │
│  - XGBoost / LightGBM                                       │
│  - 入力: feature_table.csv                                  │
│  - 出力: is_flame (0 or 1)                                  │
│  - 評価: Precision, Recall, F1, ROC-AUC                     │
│  - 解釈: SHAP特徴量重要度分析                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      予測・可視化層                          │
│  - リアルタイム炎上予測                                      │
│  - Streamlitダッシュボード                                   │
│  - アラート機能                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 データフロー

### 1. データ収集 → 前処理

```
tweets_松本人志.csv (97件)
  ├→ time_series/analyze.py
  │   └→ 時系列集計 (273時間分)
  │
  ├→ sentiment_analysis/analyze.py
  │   └→ 感情スコア計算 + 1H集計
  │       └→ 松本人志_sentiment_1h.csv
  │
  └→ stance_detection/predict.py
      └→ 立場分類（BERT推論）
          └→ 松本人志_stance.csv
```

### 2. 特徴量統合

```
松本人志_sentiment_1h.csv (273行)
    + 
松本人志_stance.csv (97行 → 69時間枠に集計)
    ↓
feature_builder.py (timestamp基準でマージ)
    ↓
松本人志_feature_table.csv (273行 × 14特徴量)
```

### 3. モデル学習・予測

```
複数トピックの feature_table.csv
    +
flame_labels.csv (炎上ラベル)
    ↓
flame_detection/train.py (XGBoost学習)
    ↓
flame_model.pkl
    ↓
flame_detection/predict.py (新規データで予測)
    ↓
flame_prediction.csv (is_flame, probability)
```

## 📊 特徴量設計

### 基本特徴量（7次元）
| 特徴量 | 説明 | データソース |
|-------|------|------------|
| `volume` | 投稿数（1時間あたり） | 時系列分析 |
| `negative_rate` | ネガティブ率（0〜1） | 感情分析 |
| `sentiment_avg_score` | 感情スコア平均 | 感情分析 |
| `stance_against_rate` | AGAINST率（0〜1） | 立場分類 |
| `stance_favor_rate` | FAVOR率（0〜1） | 立場分類 |
| `stance_neutral_rate` | NEUTRAL率（0〜1） | 立場分類 |
| `stance_against_mean` | AGAINST確率平均 | 立場分類 |

### 差分・変化率特徴量（4次元）
| 特徴量 | 説明 | 計算式 |
|-------|------|--------|
| `delta_volume` | 投稿数増加量 | volume(t) - volume(t-1) |
| `delta_negative_rate` | ネガティブ率変化 | negative_rate(t) - negative_rate(t-1) |
| `delta_against_rate` | AGAINST率変化 | stance_against_rate(t) - stance_against_rate(t-1) |
| `delta_volume_rate` | 投稿数変化率 | delta_volume / volume(t-1) |

### 追加検討中の特徴量
- 移動平均（3時間、6時間、12時間）
- 分散・標準偏差（感情スコアの揺らぎ）
- エンゲージメント率（RT率、いいね率）
- 影響力スコア（フォロワー数加重）

## 🤖 機械学習モデル

### モデル選定

**候補1: XGBoost** （推奨）
- 特徴: 勾配ブースティング、高精度、解釈可能
- 用途: 表形式データに強い、特徴量重要度取得
- ハイパーパラメータ: max_depth, learning_rate, n_estimators

**候補2: LightGBM**
- 特徴: XGBoostより高速、大規模データ向け
- 用途: 同上

**候補3: Random Forest**
- 特徴: 頑健、過学習しにくい
- 用途: ベースライン比較

### 評価指標

| 指標 | 説明 | 重要度 |
|-----|------|-------|
| **Precision** | 炎上と予測したもののうち実際に炎上していた割合 | ⭐⭐⭐ |
| **Recall** | 実際の炎上のうち検知できた割合 | ⭐⭐⭐ |
| **F1-Score** | PrecisionとRecallの調和平均 | ⭐⭐⭐ |
| **ROC-AUC** | 識別性能の総合指標 | ⭐⭐ |
| **Confusion Matrix** | 誤検知・見逃しの可視化 | ⭐⭐ |

### SHAP分析

特徴量の寄与度を定量的に分析:
- どの特徴量が炎上判定に最も影響するか
- 個別予測の説明（なぜこのタイミングが炎上と判定されたか）
- 特徴量間の相互作用

## 🔧 技術スタック

### データ処理
- **pandas**: データフレーム操作
- **numpy**: 数値計算

### 自然言語処理
- **fugashi**: 形態素解析（MeCab）
- **transformers**: BERT（HuggingFace）
  - 感情分析: koheiduck/bert-japanese-finetuned-sentiment
  - 立場分類: cl-tohoku/bert-base-japanese-v3

### 機械学習
- **torch**: BERTのファインチューニング
- **xgboost**: 炎上判定モデル
- **lightgbm**: 代替モデル
- **scikit-learn**: 評価指標、前処理

### 解釈・可視化
- **shap**: 特徴量重要度分析
- **matplotlib**: グラフ描画
- **seaborn**: 統計可視化
- **streamlit**: Webダッシュボード（オプション）

### その他
- **tqdm**: 進捗バー
- **argparse**: コマンドライン引数

## 🎯 設計原則

### 1. モジュール性
- 各機能は独立したモジュール
- 入出力インターフェース統一（CSV）
- 再利用可能な設計

### 2. トピック別管理
```
outputs/
  ├── 松本人志/
  │   ├── 松本人志_sentiment_1h.csv
  │   ├── 松本人志_stance.csv
  │   └── 松本人志_feature_table.csv
  └── 台湾有事/
      └── ...
```

### 3. 再現性
- Random seed固定（42）
- バージョン管理（requirements.txt）
- ログ記録

### 4. スケーラビリティ
- バッチ処理対応
- GPU活用（BERT推論）
- 複数トピック並列処理

## 📝 実装ステータス

### ✅ 完了
- [x] データ収集（CSV）
- [x] 時系列分析
- [x] 感情分析（辞書・BERT）
- [x] 立場分類（学習・推論）
- [x] 特徴量統合
- [x] トピック別管理

### 🚧 進行中
- [ ] 炎上ラベル作成
- [ ] XGBoost学習スクリプト
- [ ] モデル評価スクリプト
- [ ] SHAP分析スクリプト

### 📅 予定
- [ ] 予測パイプライン
- [ ] ダッシュボード
- [ ] アラート機能

## 🔬 実験設計

### データ分割
- **学習データ**: 複数トピックの過去データ（炎上・非炎上両方）
- **検証データ**: 時系列分割（時間的に後のデータ）
- **テストデータ**: 新規トピック

### クロスバリデーション
- Time Series Split（時系列考慮）
- ストラティファイド分割（炎上/非炎上の比率維持）

### ハイパーパラメータ調整
- Grid Search / Random Search
- Optuna（ベイズ最適化）

## 📚 参考文献・関連研究

- BERT: Devlin et al. (2018)
- XGBoost: Chen & Guestrin (2016)
- SHAP: Lundberg & Lee (2017)
- 日本語感情分析辞書: 高村ら (2006)
- Stance Detection: Mohammad et al. (2016)

---

**最終更新**: 2025-11-28  
**バージョン**: 3.0.0-dev
