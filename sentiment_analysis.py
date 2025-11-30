"""
辞書ベースの日本語ネガポジ分析スクリプト
Xから取得したツイートデータを分析し、ネガティブ率を時系列で集計する
"""

import pandas as pd
from pathlib import Path
import sys
import glob


def load_sentiment_dictionary(dict_path: str) -> dict:
    """
    極性辞書を読み込む
    
    Args:
        dict_path: 辞書ファイルのパス (pn_ja.csv)
        
    Returns:
        dict: {単語: 極性値} の辞書
        
    Raises:
        FileNotFoundError: 辞書ファイルが存在しない場合
    """
    if not Path(dict_path).exists():
        raise FileNotFoundError(f"辞書ファイルが見つかりません: {dict_path}")
    
    print(f"📖 辞書読み込み中: {dict_path}")
    
    try:
        # 辞書ファイルを読み込む
        df_dict = pd.read_csv(dict_path)
        
        # word列とpolarity列が存在するか確認
        if 'word' not in df_dict.columns or 'polarity' not in df_dict.columns:
            raise ValueError("辞書ファイルには 'word' と 'polarity' 列が必要です")
        
        # 辞書形式に変換
        sentiment_dict = dict(zip(df_dict['word'], df_dict['polarity']))
        
        # 統計情報
        positive_count = sum(1 for v in sentiment_dict.values() if v > 0)
        negative_count = sum(1 for v in sentiment_dict.values() if v < 0)
        
        print(f"✓ 辞書語彙数: {len(sentiment_dict)}語")
        print(f"  - ポジティブ: {positive_count}語")
        print(f"  - ネガティブ: {negative_count}語")
        
        return sentiment_dict
        
    except Exception as e:
        raise ValueError(f"辞書読み込みエラー: {e}")


def calculate_sentiment_score(text: str, sentiment_dict: dict) -> int:
    """
    テキストの感情スコアを計算する（辞書ベース・簡易版）
    
    Args:
        text: 分析対象のテキスト
        sentiment_dict: 極性辞書
        
    Returns:
        int: 感情スコア（負の値＝ネガティブ、正の値＝ポジティブ）
    """
    if pd.isna(text):
        return 0
    
    text = str(text)
    score = 0
    
    # 辞書内の各単語がテキストに含まれているかチェック
    for word, polarity in sentiment_dict.items():
        if word in text:
            score += polarity
    
    return score


def analyze_sentiment(df: pd.DataFrame, sentiment_dict: dict, text_column: str = 'content') -> pd.DataFrame:
    """
    データフレームの各行に対して感情分析を実行する
    
    Args:
        df: 分析対象のDataFrame
        sentiment_dict: 極性辞書
        text_column: テキストが格納されている列名
        
    Returns:
        pd.DataFrame: sentiment_score と is_negative 列が追加されたDataFrame
    """
    print(f"\n📊 感情分析を実行中...")
    
    # テキスト列の確認
    if text_column not in df.columns:
        # 代替列名を探す
        if 'text' in df.columns:
            text_column = 'text'
        elif 'content' in df.columns:
            text_column = 'content'
        else:
            raise ValueError(f"テキスト列が見つかりません（探索: {text_column}, text, content）")
    
    print(f"  テキスト列: '{text_column}'")
    
    # 感情スコアを計算
    df['sentiment_score'] = df[text_column].apply(
        lambda x: calculate_sentiment_score(x, sentiment_dict)
    )
    
    # ネガティブ判定（スコアが0未満）
    df['is_negative'] = df['sentiment_score'] < 0
    
    # 統計情報
    total_count = len(df)
    negative_count = df['is_negative'].sum()
    negative_rate = (negative_count / total_count * 100) if total_count > 0 else 0
    avg_score = df['sentiment_score'].mean()
    
    print(f"✓ 分析完了:")
    print(f"  - 総投稿数: {total_count}件")
    print(f"  - ネガティブ投稿: {negative_count}件 ({negative_rate:.1f}%)")
    print(f"  - 平均スコア: {avg_score:.2f}")
    
    return df


def aggregate_time_series_sentiment(df: pd.DataFrame, window: str = "1H") -> pd.DataFrame:
    """
    時系列でネガポジ分析結果を集計する
    
    Args:
        df: 感情分析済みのDataFrame（インデックスは datetime）
        window: 集計ウィンドウ幅（デフォルト: 1時間）
        
    Returns:
        pd.DataFrame: 時系列集計結果
    """
    print(f"\n📈 時系列集計中（ウィンドウ: {window}）...")
    
    # ウィンドウごとに集計
    aggregated = df.resample(window).agg({
        'sentiment_score': ['count', 'mean'],  # 投稿数と平均スコア
        'is_negative': 'mean'  # ネガティブ率（True=1, False=0の平均）
    })
    
    # 列名を整理
    aggregated.columns = ['count', 'avg_score', 'negative_rate']
    
    # インデックスをリセットしてtimestamp列にする
    aggregated = aggregated.reset_index()
    aggregated.columns = ['timestamp', 'count', 'avg_score', 'negative_rate']
    
    # 統計情報
    print(f"✓ 集計完了:")
    print(f"  - データポイント数: {len(aggregated)}")
    print(f"  - 平均投稿数/ウィンドウ: {aggregated['count'].mean():.2f}")
    print(f"  - 平均ネガティブ率: {aggregated['negative_rate'].mean()*100:.1f}%")
    print(f"  - 最大ネガティブ率: {aggregated['negative_rate'].max()*100:.1f}%")
    
    return aggregated


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
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {csv_path}")
    
    print(f"\n📂 データ読み込み中: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, comment='#')
        print(f"✓ {len(df)}件のデータを読み込みました")
    except Exception as e:
        raise ValueError(f"CSV読み込みエラー: {e}")
    
    # 日付列の確認
    if 'created_at' in df.columns:
        date_column = 'created_at'
    elif 'date' in df.columns:
        date_column = 'date'
    else:
        raise ValueError("created_at または date 列が見つかりません")
    
    # datetime型に変換
    try:
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
        print(f"✓ 日付列 '{date_column}' をUTC時刻に変換しました")
    except Exception as e:
        raise ValueError(f"日付変換エラー: {e}")
    
    # 欠損値を除去
    before_count = len(df)
    df = df.dropna(subset=[date_column])
    after_count = len(df)
    if before_count > after_count:
        print(f"⚠ 日付が欠損している {before_count - after_count}件を除外しました")
    
    # インデックスに設定
    df = df.set_index(date_column)
    df = df.sort_index()
    
    print(f"✓ 期間: {df.index.min()} ～ {df.index.max()}")
    
    return df


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


def save_results(df_sentiment: pd.DataFrame, df_timeseries: pd.DataFrame, 
                output_dir: str = "sentiment_analysis", file_prefix: str = ""):
    """
    分析結果を保存する
    
    Args:
        df_sentiment: 感情分析済みの元データ
        df_timeseries: 時系列集計データ
        output_dir: 出力先ディレクトリ
        file_prefix: 出力ファイル名のプレフィックス（検索クエリなど）
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n💾 結果を保存中...")
    
    # ファイル名の生成（プレフィックスがある場合は付加）
    if file_prefix:
        timeseries_file = output_path / f"{file_prefix}_sentiment_1h.csv"
        sentiment_file = output_path / f"{file_prefix}_analyzed.csv"
    else:
        timeseries_file = output_path / "sentiment_1h.csv"
        sentiment_file = output_path / "analyzed.csv"
    
    # 時系列集計結果を保存
    df_timeseries.to_csv(timeseries_file, index=False)
    print(f"✓ 時系列集計: {timeseries_file}")
    
    # 感情分析済みの全データも保存
    df_sentiment.to_csv(sentiment_file)
    print(f"✓ 感情分析済みデータ: {sentiment_file}")


def process_single_file(csv_path: str, sentiment_dict: dict, window: str, output_dir: str):
    """
    単一のCSVファイルを処理する
    
    Args:
        csv_path: CSVファイルのパス
        sentiment_dict: 極性辞書
        window: 集計ウィンドウ幅
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
    
    # 感情分析を実行
    df_analyzed = analyze_sentiment(df, sentiment_dict)
    
    # 時系列集計
    df_timeseries = aggregate_time_series_sentiment(df_analyzed, window=window)
    
    # 結果を保存
    save_results(df_analyzed, df_timeseries, output_dir, file_prefix=query)


def main():
    """
    メイン処理
    """
    print("=" * 60)
    print("辞書ベースの日本語ネガポジ分析スクリプト")
    print("=" * 60)
    
    # 設定===========================================================
    # コマンドライン引数から入力パターンを取得
    if len(sys.argv) > 1:
        INPUT_PATTERNS = []
        for arg in sys.argv[1:]:
            # ディレクトリが指定された場合
            if Path(arg).is_dir():
                INPUT_PATTERNS.append(f"{arg}/**/*.csv")
            else:
                INPUT_PATTERNS.append(arg)
    else:
        # デフォルト
        INPUT_PATTERNS = ["data/original/**/*.csv"]
    
    DICT_PATH = "data/dictionary/pn_ja.csv"
    OUTPUT_DIR = "data/processed"  # 出力先フォルダ
    WINDOW = "1h"  # 1時間ごとに集計
    # ===============================================================
    
    try:
        # 1. 極性辞書を読み込む（1回のみ）
        sentiment_dict = load_sentiment_dictionary(DICT_PATH)
        
        # 2. 入力ファイル一覧を取得
        csv_files = []
        for pattern in INPUT_PATTERNS:
            if '*' in pattern or '?' in pattern:
                # ワイルドカードの場合
                matched_files = glob.glob(pattern, recursive=True)
                csv_files.extend(matched_files)
            else:
                # 個別ファイルの場合
                if Path(pattern).exists():
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
        
        # 3. 各ファイルを処理
        success_count = 0
        error_count = 0
        
        for csv_file in csv_files:
            try:
                process_single_file(csv_file, sentiment_dict, WINDOW, OUTPUT_DIR)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"\n❌ エラー ({Path(csv_file).name}): {e}")
                continue
        
        print("\n" + "=" * 60)
        print(f"✅ 処理完了: 成功 {success_count}件 / エラー {error_count}件")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
