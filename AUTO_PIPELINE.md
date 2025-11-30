# 全自動パイプライン (`auto_pipeline.py`)

## 📖 概要

生データから学習済みモデルまで、全処理を自動実行するスクリプト。

## 🚀 使い方

### 基本的な使用方法

```bash
# 仮想環境をアクティベート（必須）
source .venv/bin/activate

# 全自動実行（学習以外）
python3 auto_pipeline.py 三苫 --skip-training

# 全自動実行（学習含む）
python3 auto_pipeline.py 三苫
```

### オプション

```bash
# 特定のステップだけ実行
python3 auto_pipeline.py 三苫 --steps sentiment,stance,feature

# 既存ファイルを上書き
python3 auto_pipeline.py 三苫 --force

# ヘルプ表示
python3 auto_pipeline.py --help
```

## 🔄 処理ステップ

1. **combine** - 複数CSVファイルを1つに結合
2. **sentiment** - 感情分析（辞書ベース）
3. **stance** - 立場検出（BERT）
4. **feature** - 特徴量統合
5. **visualize** - グラフ化
6. **label** - 炎上期間ラベリング（手動設定必要）
7. **train** - モデル学習

## 📁 出力ファイル

```
data/original/
  └── <トピック>_combined.csv          # 結合データ

data/processed/
  └── <トピック>_sentiment_1h.csv      # 感情分析結果

modules/stance_detection/outputs/<トピック>/
  └── <トピック>_stance.csv            # 立場検出結果

modules/feature_engineering/outputs/<トピック>/
  └── <トピック>_feature_table.csv     # 特徴量テーブル

modules/flame_detection/outputs/<トピック>/
  ├── <トピック>_labeled.csv           # ラベル付きデータ
  └── model/
      ├── model.pkl                    # 学習済みモデル
      └── evaluation.txt               # 評価結果
```

## ⚠️ 注意事項

### ラベリングは手動

可視化後、`label_config_<トピック>.yaml` を手動で作成する必要があります。

```yaml
controversy_periods:
  - start: "2022-12-02 00:00:00"
    end: "2022-12-03 23:59:59"
    label: "炎上期間"
```

### 仮想環境必須

必ず仮想環境内で実行してください:

```bash
source .venv/bin/activate
python3 auto_pipeline.py <トピック>
```

## 📝 完全な実行例

```bash
# 1. 仮想環境アクティベート
cd /home/h-okamoto/デスクトップ/sotuken/flame_analysis
source .venv/bin/activate

# 2. 可視化まで自動実行
python3 auto_pipeline.py 三苫 --steps combine,sentiment,stance,feature,visualize

# 3. 可視化結果を確認
# → modules/flame_detection/outputs/三苫_feature_trends.png

# 4. ラベル設定ファイルを作成
code modules/flame_detection/label_config_三苫.yaml

# 5. ラベリング＆学習を実行
python3 auto_pipeline.py 三苫 --steps label,train

# 6. 結果確認
cat modules/flame_detection/outputs/三苫/model/evaluation.txt
```

## 🛠️ トラブルシューティング

### ModuleNotFoundError: No module named 'torch'

→ 仮想環境をアクティベートしてください

```bash
source .venv/bin/activate
```

### ラベル設定ファイルが見つかりません

→ 可視化結果を見て手動で作成してください

```bash
code modules/flame_detection/label_config_<トピック>.yaml
```

### 全データが炎上=1になってしまう

→ 炎上期間を狭めるか、そのトピックはスキップしてください
