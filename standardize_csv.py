#!/usr/bin/env python3
"""
CSVフォーマット統一化スクリプト

機能:
1. 各トピックのオリジナルCSVファイルを統一フォーマットに変換
2. コメント行を削除
3. 改行やカンマを含むcontent列を適切に処理
4. タイムスタンプをJSTに統一
5. 重複ツイートを削除
6. メタデータを別途JSON保存

出力形式:
- data/standardized/{topic}.csv: 統一フォーマットのCSV
- data/standardized/{topic}_meta.json: メタデータ（取得期間、件数等）

Usage:
    python standardize_csv.py                    # 全トピック処理
    python standardize_csv.py 松本人志           # 特定トピックのみ
    python standardize_csv.py --list             # 利用可能トピック一覧
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

# 設定
BASE_DIR = Path(__file__).parent
ORIGINAL_DIR = BASE_DIR / "data" / "original"
OUTPUT_DIR = BASE_DIR / "data" / "standardized"

# 統一カラム定義
STANDARD_COLUMNS = [
    "timestamp",          # datetime (JST)
    "tweet_id",           # str (ID)
    "url",                # str
    "content",            # str (テキスト)
    "user_id",            # str
    "user_name",          # str
    "reply_count",        # int
    "retweet_count",      # int
    "like_count",         # int
]


def discover_topics():
    """利用可能なトピック（フォルダ名）を検出"""
    topics = []
    if ORIGINAL_DIR.exists():
        for item in ORIGINAL_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                topics.append(item.name)
    return sorted(topics)


def parse_comment_metadata(file_path):
    """CSVファイルの先頭コメント行からメタデータを抽出"""
    metadata = {
        "query": None,
        "collected_at": None,
        "original_count": None,
        "source_file": str(file_path.name)
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            line = line.strip()
            if '検索クエリ:' in line:
                metadata["query"] = line.split('検索クエリ:')[1].strip()
            elif '取得日時:' in line:
                metadata["collected_at"] = line.split('取得日時:')[1].strip()
            elif '取得件数:' in line:
                try:
                    metadata["original_count"] = int(line.split('取得件数:')[1].strip())
                except ValueError:
                    pass
    
    return metadata


def read_csv_with_comments(file_path):
    """
    コメント行をスキップしてCSVを読み込む
    改行を含むcontent列も適切に処理
    """
    try:
        # まずコメント行をスキップして読み込み
        df = pd.read_csv(
            file_path,
            comment='#',
            on_bad_lines='warn',
            encoding='utf-8'
        )
        return df
    except Exception as e:
        print(f"  ⚠️ 標準読み込み失敗、行単位処理を試行: {e}")
        return read_csv_line_by_line(file_path)


def read_csv_line_by_line(file_path):
    """
    CSVを行単位で読み込み（パースエラー対策）
    """
    rows = []
    header = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # コメント行を除去
    lines = []
    for line in content.split('\n'):
        if not line.startswith('#'):
            lines.append(line)
    content = '\n'.join(lines)
    
    # pandas で再パース
    from io import StringIO
    try:
        df = pd.read_csv(
            StringIO(content),
            on_bad_lines='skip',
            encoding='utf-8'
        )
        return df
    except Exception as e:
        print(f"  ❌ CSV読み込み完全失敗: {e}")
        return pd.DataFrame()


def convert_to_jst(timestamp_str):
    """タイムスタンプをJSTに変換"""
    try:
        # ISO 8601形式 (2023-01-29T14:25:37.000Z)
        if 'T' in str(timestamp_str) and str(timestamp_str).endswith('Z'):
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.astimezone(ZoneInfo('Asia/Tokyo'))
        # その他の形式
        dt = pd.to_datetime(timestamp_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        return dt.astimezone(ZoneInfo('Asia/Tokyo'))
    except Exception:
        return None


def standardize_dataframe(df):
    """DataFrameを統一フォーマットに変換"""
    std_df = pd.DataFrame()
    
    # カラムマッピング
    column_mapping = {
        'date': 'timestamp',
        'id': 'tweet_id',
        'url': 'url',
        'content': 'content',
        'user': 'user_id',
        'user_displayname': 'user_name',
        'reply_count': 'reply_count',
        'retweet_count': 'retweet_count',
        'like_count': 'like_count',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            std_df[new_col] = df[old_col]
        else:
            std_df[new_col] = None
    
    # タイムスタンプをJSTに変換
    std_df['timestamp'] = std_df['timestamp'].apply(convert_to_jst)
    
    # 数値列の型変換
    for col in ['reply_count', 'retweet_count', 'like_count']:
        if col in std_df.columns:
            std_df[col] = pd.to_numeric(std_df[col], errors='coerce').fillna(0).astype(int)
    
    # tweet_idを文字列に
    std_df['tweet_id'] = std_df['tweet_id'].astype(str)
    
    # content の改行を正規化（\n → 空白）
    std_df['content'] = std_df['content'].fillna('').astype(str).str.replace('\n', ' ').str.replace('\r', '')
    
    return std_df


def process_topic(topic_name, force=False):
    """
    トピックの全CSVファイルを統一フォーマットに変換
    
    Args:
        topic_name: トピック名（フォルダ名）
        force: 既存ファイルを上書きするか
    
    Returns:
        dict: 処理結果
    """
    topic_dir = ORIGINAL_DIR / topic_name
    output_csv = OUTPUT_DIR / f"{topic_name}.csv"
    output_meta = OUTPUT_DIR / f"{topic_name}_meta.json"
    
    print(f"\n{'='*60}")
    print(f"📁 トピック: {topic_name}")
    print(f"{'='*60}")
    
    if not topic_dir.exists():
        print(f"❌ フォルダが見つかりません: {topic_dir}")
        return {"status": "error", "message": "フォルダなし"}
    
    if output_csv.exists() and not force:
        print(f"⏭️ 既に存在します（--forceで上書き）: {output_csv}")
        return {"status": "skipped", "message": "既存"}
    
    # CSVファイル一覧を取得
    csv_files = list(topic_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ CSVファイルが見つかりません")
        return {"status": "error", "message": "CSVなし"}
    
    print(f"📂 {len(csv_files)}個のCSVファイルを検出")
    
    # 全ファイルを読み込み
    all_dfs = []
    all_metadata = []
    
    for csv_file in sorted(csv_files):
        print(f"  📄 {csv_file.name}")
        
        # メタデータ抽出
        meta = parse_comment_metadata(csv_file)
        all_metadata.append(meta)
        
        # CSV読み込み
        df = read_csv_with_comments(csv_file)
        if df.empty:
            print(f"    ⚠️ データなし")
            continue
        
        print(f"    ✓ {len(df)}件")
        all_dfs.append(df)
    
    if not all_dfs:
        print(f"❌ 有効なデータがありません")
        return {"status": "error", "message": "データなし"}
    
    # 結合
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 結合結果: {len(combined_df)}件")
    
    # 統一フォーマットに変換
    std_df = standardize_dataframe(combined_df)
    
    # 無効な行を削除（timestampがNone）
    valid_count_before = len(std_df)
    std_df = std_df.dropna(subset=['timestamp'])
    if len(std_df) < valid_count_before:
        print(f"  ⚠️ 無効な日付の行を除外: {valid_count_before - len(std_df)}件")
    
    # 重複削除（tweet_idベース）
    dup_count = std_df.duplicated(subset=['tweet_id']).sum()
    std_df = std_df.drop_duplicates(subset=['tweet_id'])
    if dup_count > 0:
        print(f"  ✓ 重複削除: {dup_count}件")
    
    # 時系列でソート
    std_df = std_df.sort_values('timestamp').reset_index(drop=True)
    
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # CSV保存
    print(f"\n💾 保存中: {output_csv}")
    std_df.to_csv(output_csv, index=False)
    
    # メタデータ集計
    period_start = std_df['timestamp'].min()
    period_end = std_df['timestamp'].max()
    
    meta_summary = {
        "topic": topic_name,
        "created_at": datetime.now().isoformat(),
        "total_tweets": len(std_df),
        "period": {
            "start": period_start.isoformat() if period_start else None,
            "end": period_end.isoformat() if period_end else None,
        },
        "source_files": len(csv_files),
        "queries": list(set(m["query"] for m in all_metadata if m["query"])),
        "columns": STANDARD_COLUMNS,
    }
    
    with open(output_meta, 'w', encoding='utf-8') as f:
        json.dump(meta_summary, f, ensure_ascii=False, indent=2)
    print(f"💾 メタデータ: {output_meta}")
    
    # 統計表示
    print(f"\n📈 統計情報:")
    print(f"  期間: {period_start} 〜 {period_end}")
    print(f"  総ツイート数: {len(std_df)}")
    print(f"  日数: {(period_end - period_start).days + 1 if period_start and period_end else 'N/A'}日")
    
    # 時間分布の確認
    hourly_dist = std_df.groupby(std_df['timestamp'].dt.hour).size()
    print(f"  時間帯分布 (JST):")
    for hour in range(0, 24, 6):
        count = hourly_dist.get(hour, 0)
        print(f"    {hour:02d}:00 - {count}件")
    
    return {
        "status": "success",
        "total_tweets": len(std_df),
        "output_csv": str(output_csv),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CSVフォーマット統一化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python standardize_csv.py              # 全トピック処理
  python standardize_csv.py 松本人志     # 特定トピックのみ
  python standardize_csv.py --list       # 利用可能トピック一覧
  python standardize_csv.py --force      # 既存ファイルを上書き
        """
    )
    parser.add_argument('topic', nargs='?', help='処理するトピック名')
    parser.add_argument('--list', action='store_true', help='利用可能なトピック一覧を表示')
    parser.add_argument('--force', '-f', action='store_true', help='既存ファイルを上書き')
    parser.add_argument('--all', '-a', action='store_true', help='全トピックを処理')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📋 CSVフォーマット統一化スクリプト")
    print("=" * 60)
    
    # トピック一覧を取得
    topics = discover_topics()
    
    if args.list:
        print(f"\n利用可能なトピック ({len(topics)}件):")
        for topic in topics:
            csv_count = len(list((ORIGINAL_DIR / topic).glob("*.csv")))
            print(f"  - {topic} ({csv_count} files)")
        return 0
    
    if not topics:
        print("❌ エラー: data/original/ にトピックフォルダがありません")
        return 1
    
    # 処理対象を決定
    if args.topic:
        if args.topic not in topics:
            print(f"❌ エラー: トピック '{args.topic}' が見つかりません")
            print(f"利用可能: {', '.join(topics)}")
            return 1
        target_topics = [args.topic]
    elif args.all:
        target_topics = topics
    else:
        # インタラクティブ選択
        print(f"\n利用可能なトピック:")
        for i, topic in enumerate(topics, 1):
            csv_count = len(list((ORIGINAL_DIR / topic).glob("*.csv")))
            print(f"  {i}. {topic} ({csv_count} files)")
        print(f"  0. 全て処理")
        
        try:
            choice = input("\n処理するトピックを選択 (番号): ").strip()
            if choice == '0':
                target_topics = topics
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(topics):
                    target_topics = [topics[idx]]
                else:
                    print("❌ 無効な選択")
                    return 1
        except (ValueError, EOFError):
            print("❌ 無効な入力")
            return 1
    
    # 処理実行
    results = {}
    for topic in target_topics:
        result = process_topic(topic, force=args.force)
        results[topic] = result
    
    # サマリー表示
    print("\n" + "=" * 60)
    print("📊 処理結果サマリー")
    print("=" * 60)
    
    for topic, result in results.items():
        status_emoji = {"success": "✅", "skipped": "⏭️", "error": "❌"}.get(result["status"], "❓")
        if result["status"] == "success":
            print(f"{status_emoji} {topic}: {result['total_tweets']}件")
        else:
            print(f"{status_emoji} {topic}: {result.get('message', result['status'])}")
    
    print("\n" + "=" * 60)
    print("✅ 完了！統一化されたCSVは data/standardized/ に保存されました")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
