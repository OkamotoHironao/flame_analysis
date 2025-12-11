# 設定ファイル管理

## 📁 ファイル構成

```
config/
├── presentation_config.json    # 研究発表サイト・ドキュメント用設定
└── README.md                   # このファイル
```

## 📊 presentation_config.json

研究成果の主要な数値や設定を一元管理する設定ファイル。

### 管理される値

#### 1. プロジェクト情報
- プロジェクト名
- 説明文

#### 2. 性能指標 (metrics)
- `default_best_f1`: デフォルトF1スコア (91.93%)
- `default_best_model`: デフォルト最高性能モデル ("CatBoost")
- `num_features`: 特徴量数 (16)
- `num_models_compared`: 比較したモデル数 (6)
- `latest_best_f1`: 最新の最高F1スコア (93.54%)
- `cross_topic_f1`: クロストピック評価F1 (50.21%)

#### 3. データセット情報 (dataset)
- トピック数
- トピック一覧（名前、カテゴリ）

#### 4. デフォルト特徴量重要度 (feature_importance_default)
- 特徴量名、重要度、カテゴリのリスト

#### 5. 色設定 (colors)
- カテゴリ別のカラーマップ

### 自動反映されるファイル

以下のファイルは `presentation_config.json` を自動的に読み込みます：

- ✅ **presentation_site.py**: 研究発表サイト
  - メトリックカード（F1スコア、特徴量数、モデル数）
  - データセットテーブル
  - デフォルト特徴量重要度
  - カラーマップ

### 手動更新が必要なファイル

- ⚠️ **README.md**: プロジェクトREADME
  - `scripts/show_config_values.py` を実行してコピー&ペースト

## 🔧 使い方

### 設定値の確認

```bash
python3 scripts/show_config_values.py
```

このスクリプトは以下を表示します：
- 現在の設定値一覧
- README.md用のマークダウンテキスト（コピー可能）

### 設定値の更新手順

1. **設定ファイルを編集**
   ```bash
   nano config/presentation_config.json
   # または
   code config/presentation_config.json
   ```

2. **変更内容を確認**
   ```bash
   python3 scripts/show_config_values.py
   ```

3. **README.mdを更新（必要に応じて）**
   - 上記スクリプトの出力をコピー
   - README.mdの該当箇所に貼り付け

4. **presentation_site.pyは自動反映**
   - 次回起動時に自動的に新しい値を読み込む
   - サーバー再起動: `streamlit run presentation_site.py`

## 📝 更新例

### 例1: F1スコアが向上した場合

```json
{
  "metrics": {
    "latest_best_f1": 94.50,  // 93.54 から更新
    "cross_topic_f1": 52.30   // 50.21 から更新
  }
}
```

### 例2: 特徴量を追加した場合

```json
{
  "metrics": {
    "num_features": 20  // 16 から更新
  }
}
```

### 例3: 新しいトピックを追加した場合

```json
{
  "dataset": {
    "num_topics": 13,  // 12 から更新
    "topics": [
      // 既存のトピック...
      {
        "name": "新トピック名",
        "category": "カテゴリ名"
      }
    ]
  }
}
```

## ⚠️ 注意事項

### JSON形式の厳守
- 最後の要素に `,` をつけない
- 文字列は `"` で囲む
- 数値に `"` は不要

### バックアップ
重要な更新前はバックアップを取る：
```bash
cp config/presentation_config.json config/presentation_config.json.bak
```

### バリデーション
編集後、JSONが正しいか確認：
```bash
python3 -c "import json; json.load(open('config/presentation_config.json'))"
```

エラーがなければ `No output` → ✅ OK

## 🔗 関連ファイル

| ファイル | 説明 | 自動/手動 |
|---------|------|----------|
| `presentation_site.py` | 研究発表サイト | ✅ 自動 |
| `scripts/show_config_values.py` | 設定値表示 | ✅ 自動 |
| `README.md` | プロジェクトREADME | ⚠️ 手動 |
| `dashboard.py` | 運用ダッシュボード | N/A（動的読込） |

## 💡 今後の拡張

以下も設定ファイル化を検討：
- モデルのハイパーパラメータ
- データパス設定
- API設定（Slack通知など）
