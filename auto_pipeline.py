#!/usr/bin/env python3
"""
全自動炎上検知パイプライン v2

生データから学習済みモデルまで一気通貫で処理

v2 改善点:
- enhanced_sentiment（キーワードブースト）統合
- unified_model_v2（複合特徴量）統合
- 全トピック統合学習オプション追加

データソース:
- 統一フォーマット (data/standardized/{topic}.csv) を優先使用
- 存在しない場合は従来の結合処理を実行
"""

import argparse
import subprocess
import sys
from pathlib import Path
import glob
import pandas as pd
import numpy as np


def run_command(cmd, description, background=False):
    """コマンドを実行"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    # 仮想環境を使う
    venv_python = "python3"  # 既に仮想環境内で実行される前提
    
    print(f"$ {cmd}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=not background,
        text=True,
        executable='/bin/bash'
    )
    
    if result.returncode != 0 and not background:
        print(f"❌ エラー: {description}が失敗しました")
        if result.stderr:
            print(result.stderr)
        return False
    
    if not background and result.stdout:
        print(result.stdout)
    
    return True


def combine_csv_files(topic_name, data_dir):
    """複数のCSVファイルを1つに結合"""
    print(f"\n{'='*60}")
    print(f"📂 CSVファイル結合: {topic_name}")
    print(f"{'='*60}")
    
    # CSVファイル検索
    pattern = f"{data_dir}/**/*.csv"
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        print(f"⚠️  CSVファイルが見つかりません: {pattern}")
        return None
    
    print(f"✓ {len(files)}個のファイルを発見")
    
    # 結合
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, comment='#')
            dfs.append(df)
            print(f"  ✓ {Path(f).name}: {len(df)}件")
        except Exception as e:
            print(f"  ⚠️  スキップ: {Path(f).name} ({e})")
            continue
    
    if not dfs:
        print("❌ 読み込めるファイルがありません")
        return None
    
    # 結合
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ 結合完了: {len(combined)}件")
    
    # 重複削除（content列がある場合）
    if 'content' in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['content'], keep='first')
        print(f"✓ 重複削除: {before}件 → {len(combined)}件")
    
    # 保存
    output_path = f"data/original/{topic_name}_combined.csv"
    combined.to_csv(output_path, index=False)
    print(f"✓ 保存: {output_path}")
    
    return output_path


def combine_sentiment_results(topic_name):
    """感情分析結果（複数ファイル）を1つに結合"""
    print(f"\n{'='*60}")
    print(f"📊 感情分析結果を結合: {topic_name}")
    print(f"{'='*60}")
    
    # 感情分析結果ファイルを検索
    pattern = f"data/processed/{topic_name}*_sentiment_1h.csv"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"⚠️  感情分析結果が見つかりません: {pattern}")
        return None
    
    print(f"✓ {len(files)}個のファイルを発見")
    
    # 結合
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            continue
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # timestampでソートして重複削除
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp')
    combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
    
    print(f"✓ 結合完了: {len(combined)}時間分")
    
    # 保存
    output_path = f"data/processed/{topic_name}_sentiment_1h.csv"
    combined.to_csv(output_path, index=False)
    print(f"✓ 保存: {output_path}")
    
    return output_path


def apply_enhanced_sentiment(bert_csv, stance_csv, output_csv):
    """
    enhanced_sentimentを適用して感情分析を補正
    
    BERTの感情分析結果にキーワードブースト＋スタンス統合を適用
    """
    print(f"\n{'='*60}")
    print(f"🔧 感情分析補正（キーワードブースト＋スタンス統合）")
    print(f"{'='*60}")
    
    # モジュールをインポート
    sys.path.insert(0, str(Path(__file__).parent / "modules" / "sentiment_analysis"))
    from enhanced_sentiment import calculate_enhanced_negativity, compare_methods
    
    # BERTデータ読み込み
    df = pd.read_csv(bert_csv)
    print(f"✓ BERT結果読み込み: {len(df)}件")
    
    # カラム名の正規化
    col_mapping = {
        'negative': 'bert_negative',
        'positive': 'bert_positive',
        'neutral': 'bert_neutral',
        'label': 'bert_label',
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # スタンスデータがあれば結合
    if Path(stance_csv).exists():
        stance_df = pd.read_csv(stance_csv)
        print(f"✓ スタンス結果読み込み: {len(stance_df)}件")
        
        # スタンスデータを結合（content列で）
        if 'content' in df.columns and 'content' in stance_df.columns:
            stance_subset = stance_df[['content', 'stance_label']].drop_duplicates(subset=['content'])
            df = df.merge(stance_subset, on='content', how='left')
            df['stance_label'] = df['stance_label'].fillna('NEUTRAL')
    else:
        print(f"⚠️ スタンスデータなし: スタンス統合をスキップ")
        df['stance_label'] = 'NEUTRAL'
    
    # enhanced_sentiment適用
    df = calculate_enhanced_negativity(
        df,
        bert_weight=0.4,
        keyword_weight=0.3,
        stance_weight=0.3,
        content_col='content',
        bert_neg_col='bert_negative',
        bert_label_col='bert_label',
        stance_col='stance_label',
    )
    
    # 比較統計
    if 'bert_label' in df.columns:
        stats = compare_methods(df)
        print(f"\n📊 補正効果:")
        print(f"  BERT NEGATIVE:     {stats['bert_negative']}件 ({stats['bert_negative']/stats['total']*100:.1f}%)")
        print(f"  補正後 NEGATIVE:   {stats['enhanced_negative']}件 ({stats['enhanced_negative']/stats['total']*100:.1f}%)")
        print(f"  NEUTRAL→NEGATIVE:  {stats['neutral_to_negative']}件")
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"\n✓ 補正済みデータ保存: {output_csv}")
    
    return df


def aggregate_enhanced_to_timeseries(enhanced_csv, output_csv):
    """
    enhanced_sentimentの結果を1時間ごとに集計
    """
    print(f"\n{'='*60}")
    print(f"📈 補正済み感情分析の時系列集計")
    print(f"{'='*60}")
    
    df = pd.read_csv(enhanced_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.floor('h')
    
    # 集計
    hourly = df.groupby('hour').agg({
        'content': 'count',
        'bert_negative': 'mean',
        'enhanced_negativity': 'mean',
        'enhanced_label': lambda x: (x == 'NEGATIVE').mean(),
    }).reset_index()
    
    hourly.columns = [
        'timestamp', 'count',
        'bert_negative_rate', 'enhanced_negative_score', 'negative_rate'
    ]
    
    # 既存形式との互換性のため追加
    hourly['positive_count'] = 0  # 簡略化
    hourly['neutral_count'] = 0
    hourly['negative_count'] = (hourly['count'] * hourly['negative_rate']).astype(int)
    
    print(f"✓ {len(hourly)}時間分の集計完了")
    print(f"  平均ネガティブ率: {hourly['negative_rate'].mean()*100:.1f}%")
    
    hourly.to_csv(output_csv, index=False)
    print(f"✓ 保存: {output_csv}")
    
    return hourly


def run_unified_training(topics_str=None):
    """
    全トピック統合学習（train_unified_model_v2.py）を実行
    """
    print("="*60)
    print("🔥 全トピック統合学習")
    print("="*60)
    
    cmd = "cd modules/flame_detection && python3 train_unified_model_v2.py"
    
    if topics_str:
        cmd += f" --topics {topics_str}"
    
    if not run_command(cmd, "統合学習 (v2)"):
        sys.exit(1)
    
    print("\n✅ 統合学習完了！")
    print("📂 出力: modules/flame_detection/outputs/unified_model_v2/")


def main():
    parser = argparse.ArgumentParser(
        description='全自動炎上検知パイプライン v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 三苫の全処理
  python3 auto_pipeline.py 三苫
  
  # aespaの全処理
  python3 auto_pipeline.py aespa
  
  # 特定のステップだけ実行
  python3 auto_pipeline.py 三苫 --steps sentiment,stance
  
  # 学習をスキップ
  python3 auto_pipeline.py 三苫 --skip-training
  
  # 全トピック統合学習
  python3 auto_pipeline.py --unified-train
  
  # 感情補正あり（キーワードブースト）
  python3 auto_pipeline.py 三苫 --enhanced-sentiment
        """
    )
    
    parser.add_argument(
        'topic',
        type=str,
        nargs='?',
        default=None,
        help='トピック名（例: 三苫, aespa, 松本人志）'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='実行するステップ（カンマ区切り: combine,sentiment,enhance,stance,feature,visualize,label,train）'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='モデル学習をスキップ'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='既存ファイルを上書き'
    )
    
    parser.add_argument(
        '--enhanced-sentiment',
        action='store_true',
        help='感情分析補正（キーワードブースト＋スタンス統合）を適用'
    )
    
    parser.add_argument(
        '--unified-train',
        action='store_true',
        help='全トピックで統合学習（train_unified_model_v2.py）'
    )
    
    parser.add_argument(
        '--unified-topics',
        type=str,
        default=None,
        help='統合学習に使用するトピック（カンマ区切り）'
    )
    
    args = parser.parse_args()
    
    # 統合学習モード
    if args.unified_train:
        run_unified_training(args.unified_topics)
        return
    
    # トピック必須チェック
    if args.topic is None:
        parser.error("トピック名を指定してください（または --unified-train を使用）")
    
    topic = args.topic
    
    # ステップ設定
    if args.steps == 'all':
        steps = ['combine', 'sentiment', 'stance', 'feature', 'visualize', 'label', 'train']
        # enhanced-sentimentオプション時は'enhance'ステップを追加
        if args.enhanced_sentiment:
            steps.insert(steps.index('stance') + 1, 'enhance')
    else:
        steps = [s.strip() for s in args.steps.split(',')]
    
    if args.skip_training and 'train' in steps:
        steps.remove('train')
    
    print("="*60)
    print(f"🔥 全自動炎上検知パイプライン")
    print("="*60)
    print(f"  トピック: {topic}")
    print(f"  実行ステップ: {', '.join(steps)}")
    print("="*60)
    
    # 出力ディレクトリ作成
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path(f"modules/stance_detection/outputs/{topic}").mkdir(parents=True, exist_ok=True)
    Path(f"modules/feature_engineering/outputs/{topic}").mkdir(parents=True, exist_ok=True)
    Path(f"modules/flame_detection/outputs/{topic}").mkdir(parents=True, exist_ok=True)
    
    # データディレクトリ確認
    data_dir = f"data/original/{topic}"
    standardized_csv = f"data/standardized/{topic}.csv"
    
    # 統一CSVが存在するか確認
    use_standardized = Path(standardized_csv).exists()
    
    if use_standardized:
        print(f"✓ 統一フォーマットCSVを使用: {standardized_csv}")
        combined_csv = standardized_csv
    elif not Path(data_dir).exists():
        print(f"❌ エラー: データが見つかりません")
        print(f"   統一CSV: {standardized_csv}")
        print(f"   オリジナルディレクトリ: {data_dir}")
        print(f"\n   まず standardize_csv.py を実行してください:")
        print(f"   → python3 standardize_csv.py {topic}")
        sys.exit(1)
    else:
        combined_csv = f"data/original/{topic}_combined.csv"
    
    # 出力ファイルパス
    bert_output_csv = f"data/processed/{topic}_bert.csv"
    sentiment_csv = f"data/processed/{topic}_sentiment_1h.csv"
    stance_csv = f"modules/stance_detection/outputs/{topic}/{topic}_stance.csv"
    feature_csv = f"modules/feature_engineering/outputs/{topic}/{topic}_feature_table.csv"
    labeled_csv = f"modules/flame_detection/outputs/{topic}/{topic}_labeled.csv"
    
    # ========================================
    # Step 1: CSVファイル結合（統一CSV使用時はスキップ）
    # ========================================
    if 'combine' in steps:
        if use_standardized:
            print(f"\n✓ スキップ: 統一フォーマットCSVを使用中")
        elif args.force or not Path(combined_csv).exists():
            combined_csv = combine_csv_files(topic, data_dir)
            if not combined_csv:
                print("❌ CSV結合に失敗しました")
                sys.exit(1)
        else:
            print(f"\n✓ スキップ: {combined_csv} は既に存在します")
    
    # ========================================
    # Step 2: 感情分析（BERTベース）
    # ========================================
    if 'sentiment' in steps:
        if args.force or not Path(sentiment_csv).exists():
            # Step 2-1: BERT感情分析
            if not run_command(
                f"python3 bert_sentiment.py {combined_csv} {bert_output_csv}",
                "感情分析（BERT）"
            ):
                sys.exit(1)
            
            # Step 2-2: 時系列集計
            if not run_command(
                f"python3 bert_sentiment_timeseries.py {bert_output_csv} {sentiment_csv}",
                "時系列集計"
            ):
                sys.exit(1)
        else:
            print(f"\n✓ スキップ: {sentiment_csv} は既に存在します")
    
    # ========================================
    # Step 3: 立場検出
    # ========================================
    if 'stance' in steps:
        if args.force or not Path(stance_csv).exists():
            if not run_command(
                f"python3 stance_predict.py {combined_csv} {stance_csv}",
                "立場検出"
            ):
                sys.exit(1)
        else:
            print(f"\n✓ スキップ: {stance_csv} は既に存在します")
    
    # ========================================
    # Step 3.5: 感情分析補正（enhanced_sentiment）
    # ========================================
    enhanced_csv = f"data/processed/{topic}_enhanced.csv"
    enhanced_sentiment_csv = f"data/processed/{topic}_enhanced_sentiment_1h.csv"
    
    if 'enhance' in steps:
        if args.force or not Path(enhanced_csv).exists():
            # 補正を適用
            apply_enhanced_sentiment(bert_output_csv, stance_csv, enhanced_csv)
            
            # 補正済みデータを時系列集計
            aggregate_enhanced_to_timeseries(enhanced_csv, enhanced_sentiment_csv)
            
            # 補正済みの感情CSVを使用するよう切り替え
            sentiment_csv = enhanced_sentiment_csv
            print(f"\n✓ 感情分析を補正済みデータに切り替え: {sentiment_csv}")
        else:
            print(f"\n✓ スキップ: {enhanced_csv} は既に存在します")
            # 補正済みが存在する場合はそれを使用
            sentiment_csv = enhanced_sentiment_csv
    
    # ========================================
    # Step 4: 特徴量統合
    # ========================================
    if 'feature' in steps:
        if args.force or not Path(feature_csv).exists():
            if not run_command(
                f"cd modules/feature_engineering && python3 feature_builder.py "
                f"--sentiment_csv ../../{sentiment_csv} "
                f"--stance_csv ../../{stance_csv}",
                "特徴量統合"
            ):
                sys.exit(1)
        else:
            print(f"\n✓ スキップ: {feature_csv} は既に存在します")
    
    # ========================================
    # Step 5: 可視化
    # ========================================
    if 'visualize' in steps:
        vis_output = f"modules/flame_detection/outputs/{topic}_feature_trends.png"
        if args.force or not Path(vis_output).exists():
            if not run_command(
                f"python3 visualize_features.py {feature_csv} {vis_output}",
                "特徴量可視化"
            ):
                sys.exit(1)
        else:
            print(f"\n✓ スキップ: {vis_output} は既に存在します")
    
    # ========================================
    # Step 6: ラベリング
    # ========================================
    if 'label' in steps:
        label_config = f"modules/flame_detection/label_config_{topic}.yaml"
        
        if not Path(label_config).exists():
            print(f"\n⚠️  警告: ラベル設定ファイルが見つかりません: {label_config}")
            print(f"   可視化結果を確認して、以下のコマンドで設定ファイルを作成してください:")
            print(f"   → code {label_config}")
            print(f"\n   その後、以下のコマンドでラベリングを実行:")
            print(f"   → python3 modules/flame_detection/label_windows.py \\")
            print(f"       {feature_csv} \\")
            print(f"       {label_config} \\")
            print(f"       {labeled_csv}")
        else:
            if args.force or not Path(labeled_csv).exists():
                if not run_command(
                    f"cd modules/flame_detection && python3 label_windows.py "
                    f"../../{feature_csv} "
                    f"label_config_{topic}.yaml "
                    f"outputs/{topic}/{topic}_labeled.csv",
                    "ラベリング"
                ):
                    sys.exit(1)
            else:
                print(f"\n✓ スキップ: {labeled_csv} は既に存在します")
    
    # ========================================
    # Step 7: モデル学習（トピック単体）
    # ========================================
    if 'train' in steps:
        model_output = f"modules/flame_detection/outputs/{topic}/model/model.pkl"
        
        if not Path(labeled_csv).exists():
            print(f"\n⚠️  警告: ラベル付きデータが見つかりません: {labeled_csv}")
            print(f"   先にラベリングを実行してください")
        else:
            if args.force or not Path(model_output).exists():
                if not run_command(
                    f"cd modules/flame_detection && python3 train_classifier.py "
                    f"outputs/{topic}/{topic}_labeled.csv "
                    f"outputs/{topic}/model/",
                    "モデル学習（トピック単体）"
                ):
                    sys.exit(1)
            else:
                print(f"\n✓ スキップ: {model_output} は既に存在します")
    
    # ========================================
    # 完了
    # ========================================
    print("\n" + "="*60)
    print("✅ パイプライン完了！")
    print("="*60)
    
    print(f"\n📂 出力ファイル:")
    if Path(combined_csv).exists():
        print(f"  ✓ 結合データ: {combined_csv}")
    if Path(bert_output_csv).exists():
        print(f"  ✓ BERT感情分析: {bert_output_csv}")
    if 'enhance' in steps and Path(enhanced_csv).exists():
        print(f"  ✓ 感情補正: {enhanced_csv}")
    if Path(sentiment_csv).exists():
        print(f"  ✓ 時系列集計: {sentiment_csv}")
    if Path(stance_csv).exists():
        print(f"  ✓ 立場検出: {stance_csv}")
    if Path(feature_csv).exists():
        print(f"  ✓ 特徴量: {feature_csv}")
    if Path(labeled_csv).exists():
        print(f"  ✓ ラベル付き: {labeled_csv}")
    if Path(f"modules/flame_detection/outputs/{topic}/model/model.pkl").exists():
        print(f"  ✓ モデル: modules/flame_detection/outputs/{topic}/model/")
    
    print(f"\n💡 次のステップ:")
    print(f"  # 全トピック統合学習（推奨）:")
    print(f"  python3 auto_pipeline.py --unified-train")
    print(f"\n  # 特定トピックで統合学習:")
    print(f"  python3 auto_pipeline.py --unified-train --unified-topics 松本人志,三苫,寿司ペロ")
    print()


if __name__ == '__main__':
    main()
