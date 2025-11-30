#!/usr/bin/env python3
"""
炎上期間ラベリングスクリプト
時系列特徴量データに炎上/非炎上のラベルを付与
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml


def load_config(config_path):
    """
    YAML設定ファイルから炎上期間を読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        list: 炎上期間のリスト [{start: datetime, end: datetime}, ...]
        
    Raises:
        FileNotFoundError: 設定ファイルが存在しない
        ValueError: 設定ファイルの形式が不正
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    print(f"\n📖 設定ファイル読み込み中: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config or 'controversy_periods' not in config:
        raise ValueError("設定ファイルに 'controversy_periods' が必要です")
    
    periods = config['controversy_periods']
    
    if not periods:
        print("⚠️  警告: 炎上期間が定義されていません（全て非炎上としてラベル付け）")
        return []
    
    # datetime に変換
    parsed_periods = []
    for i, period in enumerate(periods):
        if not period or 'start' not in period or 'end' not in period:
            print(f"⚠️  警告: 期間 {i+1} がスキップされました（start/end不足）")
            continue
        
        try:
            start = pd.to_datetime(period['start'])
            end = pd.to_datetime(period['end'])
            
            if start > end:
                print(f"⚠️  警告: 期間 {i+1} がスキップされました（start > end）")
                continue
            
            parsed_periods.append({
                'start': start,
                'end': end,
                'label': period.get('label', f'period_{i+1}'),
                'description': period.get('description', '')
            })
            
            print(f"  期間 {i+1}: {start} 〜 {end}")
            if period.get('description'):
                print(f"    説明: {period['description']}")
            
        except Exception as e:
            print(f"⚠️  警告: 期間 {i+1} の解析エラー: {e}")
            continue
    
    print(f"✓ {len(parsed_periods)}個の炎上期間を読み込みました")
    
    return parsed_periods


def load_features(feature_path):
    """
    特徴量CSVを読み込む
    
    Args:
        feature_path: 特徴量CSVのパス
        
    Returns:
        pd.DataFrame: 特徴量データ
        
    Raises:
        FileNotFoundError: ファイルが存在しない
        ValueError: timestamp列が存在しない
    """
    if not Path(feature_path).exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {feature_path}")
    
    print(f"\n📊 特徴量データ読み込み中: {feature_path}")
    
    df = pd.read_csv(feature_path, comment='#')
    
    print(f"✓ {len(df)}件のレコードを読み込みました")
    
    # timestamp列の確認
    if 'timestamp' not in df.columns:
        raise ValueError("特徴量CSVに 'timestamp' 列が必要です")
    
    # timestampをdatetimeに変換
    print("\n🕐 timestamp列を解析中...")
    
    original_count = len(df)
    parse_errors = 0
    
    # timestampを変換し、エラー行を記録
    def safe_parse(ts):
        try:
            # UTCタイムゾーンを明示的に設定
            parsed = pd.to_datetime(ts)
            if parsed.tzinfo is None:
                # タイムゾーン未指定の場合はUTCとして扱う
                parsed = parsed.tz_localize('UTC')
            return parsed
        except Exception:
            nonlocal parse_errors
            parse_errors += 1
            return pd.NaT
    
    df['timestamp'] = df['timestamp'].apply(safe_parse)
    
    # NaTを含む行を削除
    if parse_errors > 0:
        print(f"⚠️  警告: {parse_errors}件のタイムスタンプ解析エラー（該当行をスキップ）")
        df = df.dropna(subset=['timestamp'])
        print(f"  有効レコード数: {len(df)}件 / {original_count}件")
    
    if len(df) == 0:
        raise ValueError("有効なタイムスタンプが1件もありません")
    
    print(f"✓ タイムスタンプ範囲: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    
    return df


def apply_labels(df, periods):
    """
    炎上期間に基づいてラベルを付与
    
    Args:
        df: 特徴量データフレーム
        periods: 炎上期間のリスト
        
    Returns:
        pd.DataFrame: ラベル付きデータフレーム
    """
    print(f"\n🏷️  ラベル付け処理中...")
    
    # ラベル列を初期化（全て非炎上）
    df['is_controversy'] = 0
    
    if not periods:
        print("  全てのレコードに is_controversy=0 を付与（炎上期間未定義）")
        return df
    
    # 各期間について判定
    total_labeled = 0
    
    for i, period in enumerate(periods):
        start = period['start']
        end = period['end']
        label_name = period['label']
        
        # タイムゾーンを統一（UTCに変換）
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        if end.tzinfo is None:
            end = end.tz_localize('UTC')
        
        # 期間内のレコードを抽出
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        count = mask.sum()
        
        if count > 0:
            df.loc[mask, 'is_controversy'] = 1
            total_labeled += count
            print(f"  期間 {i+1} ({label_name}): {count}件を炎上としてラベル付け")
        else:
            print(f"  期間 {i+1} ({label_name}): 該当レコードなし")
    
    # 統計情報
    controversy_count = (df['is_controversy'] == 1).sum()
    non_controversy_count = (df['is_controversy'] == 0).sum()
    
    print(f"\n✓ ラベル付け完了:")
    print(f"  炎上 (is_controversy=1): {controversy_count}件 ({controversy_count/len(df)*100:.1f}%)")
    print(f"  非炎上 (is_controversy=0): {non_controversy_count}件 ({non_controversy_count/len(df)*100:.1f}%)")
    
    return df


def save_output(df, output_path):
    """
    ラベル付きデータをCSVに保存
    
    Args:
        df: ラベル付きデータフレーム
        output_path: 出力ファイルパス
    """
    print(f"\n💾 出力ファイル保存中: {output_path}")
    
    # 出力ディレクトリを作成
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV保存
    df.to_csv(output_path, index=False)
    
    print(f"✓ 保存完了 ({len(df)}件)")


def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(
        description='時系列特徴量データに炎上/非炎上のラベルを付与',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python label_windows.py feature_table.csv label_config.yaml feature_table_labeled.csv
  
  # カレントディレクトリの場合
  python label_windows.py \\
    ../feature_engineering/outputs/松本人志/松本人志_feature_table.csv \\
    label_config.yaml \\
    outputs/松本人志_labeled.csv
        """
    )
    
    parser.add_argument(
        'feature_csv',
        help='入力: 特徴量CSV（timestamp列必須）'
    )
    parser.add_argument(
        'config_yaml',
        help='入力: 炎上期間設定ファイル（YAML形式）'
    )
    parser.add_argument(
        'output_csv',
        help='出力: ラベル付き特徴量CSV'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("炎上期間ラベリングスクリプト")
    print("=" * 60)
    
    try:
        # 1. 設定ファイル読み込み
        periods = load_config(args.config_yaml)
        
        # 2. 特徴量データ読み込み
        df = load_features(args.feature_csv)
        
        # 3. ラベル付け
        df_labeled = apply_labels(df, periods)
        
        # 4. 保存
        save_output(df_labeled, args.output_csv)
        
        print("\n" + "=" * 60)
        print("✅ すべての処理が正常に完了しました！")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
