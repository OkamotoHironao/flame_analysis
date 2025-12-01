# データフォーマット仕様書

## 概要

このプロジェクトでは、ツイートデータを統一フォーマットで管理します。
オリジナルデータは様々なソースから収集されますが、`standardize_csv.py` を使って
統一フォーマットに変換してから分析パイプラインで使用します。

## ディレクトリ構造

```
data/
├── original/           # オリジナルデータ（収集ツールの出力そのまま）
│   ├── 松本人志/       # トピックごとにフォルダ分け
│   │   ├── tweets_松本人志_since-2023-12-24_...csv
│   │   └── ...
│   ├── 三苫/
│   ├── 寿司ペロ/
│   └── aespa/
│
├── standardized/       # 統一フォーマット（パイプライン入力用）
│   ├── 松本人志.csv
│   ├── 松本人志_meta.json
│   ├── 三苫.csv
│   ├── 三苫_meta.json
│   └── ...
│
└── processed/          # 処理済みデータ（パイプライン出力）
    ├── 松本人志_bert.csv
    ├── 松本人志_sentiment_1h.csv
    └── ...
```

## 統一フォーマット仕様

### CSVカラム

| カラム名 | 型 | 説明 |
|----------|------|------|
| `timestamp` | datetime (JST) | ツイート投稿時刻 (ISO 8601形式) |
| `tweet_id` | string | ツイートID |
| `url` | string | ツイートURL |
| `content` | string | ツイート本文（改行は空白に変換） |
| `user_id` | string | ユーザーID |
| `user_name` | string | ユーザー表示名 |
| `reply_count` | int | リプライ数 |
| `retweet_count` | int | リツイート数 |
| `like_count` | int | いいね数 |

### サンプル

```csv
timestamp,tweet_id,url,content,user_id,user_name,reply_count,retweet_count,like_count
2023-12-25 10:30:00+09:00,1234567890,https://x.com/user/status/1234567890,ツイート本文...,user123,ユーザー名,5,10,50
```

### メタデータ（JSON）

各統一CSVには対応するメタデータファイル（`{topic}_meta.json`）が生成されます。

```json
{
  "topic": "松本人志",
  "created_at": "2025-12-01T16:28:54.122101",
  "total_tweets": 3505,
  "period": {
    "start": "2023-12-24T23:17:09+09:00",
    "end": "2023-12-31T08:59:58+09:00"
  },
  "source_files": 7,
  "queries": ["松本人志 since:2023-12-24 until:2023-12-25", ...],
  "columns": ["timestamp", "tweet_id", "url", "content", "user_id", "user_name", "reply_count", "retweet_count", "like_count"]
}
```

## 使用方法

### 1. オリジナルデータの配置

収集したCSVファイルを `data/original/{トピック名}/` に配置します。

```
data/original/新トピック/
├── tweets_xxx_1.csv
├── tweets_xxx_2.csv
└── ...
```

### 2. 統一フォーマットに変換

```bash
# 特定トピックのみ
python standardize_csv.py 新トピック

# 全トピック
python standardize_csv.py --all

# 既存ファイルを上書き
python standardize_csv.py --all --force

# 利用可能トピック一覧
python standardize_csv.py --list
```

### 3. パイプライン実行

統一CSVが存在すれば、パイプラインは自動的にそれを使用します。

```bash
# パイプライン実行（統一CSVを自動使用）
python auto_pipeline.py 新トピック
```

## オリジナルデータ形式（参考）

収集ツールの出力形式（変換元）：

```csv
# 検索クエリ: 松本人志 since:2023-12-24 until:2023-12-25
# 取得日時: 2025-11-25 09:53:08
# 取得件数: 460
date,id,url,content,user,user_displayname,reply_count,retweet_count,like_count
2023-12-24T14:17:09.000Z,1234567890,...
```

### 変換時の処理

1. **コメント行削除**: `#` で始まる行を除去
2. **カラム名変更**: `date` → `timestamp`, `id` → `tweet_id` など
3. **タイムゾーン変換**: UTC → JST
4. **改行処理**: content内の改行を空白に変換
5. **重複削除**: tweet_id ベースで重複を除去
6. **ソート**: timestamp で昇順ソート

## 注意事項

### データ収集の制約

一部のトピックでは、元データの収集時間帯が限られている場合があります：

- **寿司ペロ**: 各日23時のデータのみ（API制限による）
- **三苫**: 各日の一部時間帯のみ

このような場合、時系列分析は「日単位」で行うことを推奨します。

### 時系列集計について

- **理想**: 1時間ごとの集計（24時間 × 日数 のウィンドウ）
- **現実**: データが疎な場合は日単位での集計を検討

## 変更履歴

- 2025-12-01: 統一フォーマット仕様を策定
