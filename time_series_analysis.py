"""
時系列ウィンドウ化分析スクリプト
X（Twitter）データの投稿時刻を集計し、炎上分析の前処理を行う
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import sys
import glob

# 日本語フォント設定（matplotlib用）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    CSVファイルを読み込み、時系列データとして準備する
    
    Args:
        csv_path: 入力CSVファイルのパス
        
    Returns:
        pd.DataFrame: 時系列インデックス化されたDataFrame
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 日付変換に失敗した場合
    """
    # ファイル存在確認
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {csv_path}")
    
    print(f"📂 データ読み込み中: {csv_path}")
    
    # CSV読み込み（コメント行をスキップ）
    try:
        df = pd.read_csv(csv_path, comment='#')
        print(f"✓ {len(df)}件のデータを読み込みました")
    except Exception as e:
        raise ValueError(f"CSV読み込みエラー: {e}")
    
    # 列名の確認と調整
    if 'created_at' in df.columns:
        date_column = 'created_at'
    elif 'date' in df.columns:
        date_column = 'date'
    else:
        raise ValueError("created_at または date 列が見つかりません")
    
    # 日付列をdatetime型に変換
    try:
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
        print(f"✓ 日付列 '{date_column}' をUTC時刻に変換しました")
    except Exception as e:
        raise ValueError(f"日付変換エラー: {e}")
    
    # NaT（欠損値）を除去
    before_count = len(df)
    df = df.dropna(subset=[date_column])
    after_count = len(df)
    if before_count > after_count:
        print(f"⚠ 日付が欠損している {before_count - after_count}件を除外しました")
    
    # インデックスを日付列に設定
    df = df.set_index(date_column)
    df = df.sort_index()
    
    print(f"✓ 期間: {df.index.min()} ～ {df.index.max()}")
    
    return df


def resample_time_series(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    指定されたウィンドウ幅で時系列データをリサンプルする
    
    Args:
        df: 時系列インデックス化されたDataFrame
        window: ウィンドウ幅（例: "1H", "30min", "10min"）
        
    Returns:
        pd.DataFrame: リサンプルされたデータ（投稿件数）
    """
    # ウィンドウごとの投稿件数を集計
    resampled = df.resample(window).size()
    
    # DataFrameに変換
    result = pd.DataFrame({
        'timestamp': resampled.index,
        'count': resampled.values
    })
    
    return result


def save_time_series_data(df: pd.DataFrame, windows: dict, output_dir: str = "time_series_data", file_prefix: str = ""):
    """
    複数のウィンドウ幅で集計し、CSVファイルに保存する
    
    Args:
        df: 時系列インデックス化されたDataFrame
        windows: ウィンドウ幅の辞書 {ファイル名接尾辞: ウィンドウ幅}
        output_dir: 出力先ディレクトリ
        file_prefix: 出力ファイル名のプレフィックス（空文字列の場合は付加しない）
    """
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📊 時系列集計を開始...")
    
    results = {}
    
    for suffix, window in windows.items():
        print(f"\n▶ {window} ウィンドウで集計中...")
        
        # リサンプル
        resampled = resample_time_series(df, window)
        
        # 統計情報を表示
        print(f"  - データポイント数: {len(resampled)}")
        print(f"  - 平均投稿件数: {resampled['count'].mean():.2f}")
        print(f"  - 最大投稿件数: {resampled['count'].max()}")
        print(f"  - 最小投稿件数: {resampled['count'].min()}")
        
        # CSV保存（プレフィックスがある場合は付加）
        if file_prefix:
            output_file = output_path / f"{file_prefix}_time_series_{suffix}.csv"
        else:
            output_file = output_path / f"time_series_{suffix}.csv"
        resampled.to_csv(output_file, index=False)
        print(f"  ✓ 保存完了: {output_file}")
        
        results[suffix] = resampled
    
    return results


def plot_time_series(df: pd.DataFrame, output_dir: str = "time_series_data", file_prefix: str = ""):
    """
    1時間ごとの投稿件数をグラフ化する
    
    Args:
        df: 時系列インデックス化されたDataFrame
        output_dir: 出力先ディレクトリ
        file_prefix: 出力ファイル名のプレフィックス（空文字列の場合は付加しない）
    """
    print(f"\n📈 グラフ生成中...")
    
    # 1時間ごとに集計
    hourly = resample_time_series(df, "1H")
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 投稿件数をプロット
    ax.plot(hourly['timestamp'], hourly['count'], 
            marker='o', markersize=4, linewidth=1.5, 
            color='#2E86AB', label='Posts per hour')
    
    # グリッド
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ラベル設定
    ax.set_xlabel('Date/Time (UTC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Posts', fontsize=12, fontweight='bold')
    ax.set_title('Time Series Analysis: Posts per Hour', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X軸の日付フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(hourly) // 20)))
    plt.xticks(rotation=45, ha='right')
    
    # 凡例
    ax.legend(loc='upper right', framealpha=0.9)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存（プレフィックスがある場合は付加）
    output_path = Path(output_dir)
    if file_prefix:
        output_file = output_path / f"{file_prefix}_time_series_plot.png"
    else:
        output_file = output_path / "time_series_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存完了: {output_file}")
    
    # 表示（オプション）
    # plt.show()
    plt.close()


def extract_query_from_filename(csv_path: str) -> str:
    """
    CSVファイル名から検索クエリを抽出する
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        str: 検索クエリ（抽出できない場合はファイル名）
    """
    csv_filename = Path(csv_path).stem  # 拡張子なしのファイル名
    
    if csv_filename.startswith("tweets_"):
        # "tweets_"を除去
        name_part = csv_filename[7:]
        # 数字のみの部分（タイムスタンプ）を除去
        parts = name_part.split("_")
        query_parts = [p for p in parts if not p.isdigit()]
        return "_".join(query_parts) if query_parts else csv_filename
    else:
        return csv_filename


def process_single_file(csv_path: str, windows: dict, output_dir: str):
    """
    単一のCSVファイルを処理する
    
    Args:
        csv_path: CSVファイルのパス
        windows: ウィンドウ幅の辞書
        output_dir: 出力先ディレクトリ
    """
    # ファイル名から検索クエリを抽出
    query = extract_query_from_filename(csv_path)
    
    print(f"\n{'='*60}")
    print(f"処理中: {Path(csv_path).name}")
    print(f"検索クエリ: {query}")
    print(f"{'='*60}")
    
    # データ読み込みと準備
    df = load_and_prepare_data(csv_path)
    
    # 時系列集計とCSV保存
    results = save_time_series_data(df, windows, output_dir, file_prefix=query)
    
    # グラフ生成
    plot_time_series(df, output_dir, file_prefix=query)


def main():
    """
    メイン処理
    """
    print("=" * 60)
    print("時系列ウィンドウ化分析スクリプト")
    print("=" * 60)
    
    # 設定===========================================================
    # 複数ファイル対応: ワイルドカードや個別ファイルパスのリスト
    INPUT_PATTERNS = [
        "original_data/tweets_*.csv",  # ワイルドカード指定
        # "original_data/tweets_松本人志_20251112_093317.csv",  # 個別指定も可能
    ]
    OUTPUT_DIR = "time_series_data"
    # ===============================================================
    
    # ウィンドウ幅の定義
    WINDOWS = {
        "1h": "1H",      # 1時間
        "30m": "30min",  # 30分
        "10m": "10min"   # 10分
    }
    
    try:
        # 入力ファイル一覧を取得
        csv_files = []
        for pattern in INPUT_PATTERNS:
            if '*' in pattern or '?' in pattern:
                # ワイルドカードの場合
                matched_files = glob.glob(pattern)
                csv_files.extend(matched_files)
            else:
                # 個別ファイルの場合
                csv_files.append(pattern)
        
        # 重複を除去してソート
        csv_files = sorted(set(csv_files))
        
        if not csv_files:
            print("\n⚠ 処理対象のCSVファイルが見つかりませんでした")
            print(f"パターン: {INPUT_PATTERNS}")
            sys.exit(0)
        
        print(f"\n📋 処理対象ファイル数: {len(csv_files)}")
        for i, f in enumerate(csv_files, 1):
            print(f"  {i}. {f}")
        
        # 各ファイルを処理
        success_count = 0
        error_count = 0
        
        for csv_file in csv_files:
            try:
                process_single_file(csv_file, WINDOWS, OUTPUT_DIR)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"\n❌ エラー ({Path(csv_file).name}): {e}")
                continue
        
        print("\n" + "=" * 60)
        print(f"✅ 処理完了: 成功 {success_count}件 / エラー {error_count}件")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
