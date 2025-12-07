#!/usr/bin/env python3
"""
リアルタイム炎上スコア予測モジュール

新しいツイートデータに対して、現在の炎上スコアを判定する
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import sys
import torch
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sentiment_model():
    """感情分析モデルを読み込み"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "koheiduck/bert-japanese-finetuned-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device


def analyze_sentiment_single(text, tokenizer, model, device):
    """単一テキストの感情分析"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # ラベルマッピング
    id2label = model.config.id2label
    
    result = {}
    for idx, label in id2label.items():
        label_lower = label.lower()
        if 'positive' in label_lower or 'ポジティブ' in label_lower:
            result['positive'] = probs[idx].item()
        elif 'negative' in label_lower or 'ネガティブ' in label_lower:
            result['negative'] = probs[idx].item()
        else:
            result['neutral'] = probs[idx].item()
    
    # デフォルト値
    result.setdefault('positive', 0.33)
    result.setdefault('neutral', 0.34)
    result.setdefault('negative', 0.33)
    
    # ラベル判定
    max_key = max(['positive', 'neutral', 'negative'], key=lambda k: result.get(k, 0))
    result['label'] = max_key
    
    return result


class FlamePredictor:
    """炎上スコア予測クラス"""
    
    def __init__(self, model_dir=None):
        """
        Args:
            model_dir: モデルディレクトリのパス（デフォルト: outputs/unified_model_v2）
        """
        if model_dir is None:
            model_dir = PROJECT_ROOT / "outputs" / "unified_model_v2"
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_columns = None
        
        # 感情分析モデル
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.device = None
        
        # スタンス検出モデル
        self.stance_tokenizer = None
        self.stance_model = None
        
        self._load_model()
    
    def _load_model(self):
        """学習済みモデルを読み込み"""
        print("📦 モデル読み込み中...")
        
        # XGBoostモデル
        model_path = self.model_dir / "model.pkl"
        self.model = joblib.load(model_path)
        
        # スケーラー
        scaler_path = self.model_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        
        # メタデータ
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_columns = self.metadata.get('feature_columns', [
            'volume', 'delta_volume', 'negative_rate', 'delta_negative_rate',
            'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate',
            'flame_score', 'against_count', 'sentiment_polarity'
        ])
        
        # 感情分析モデル
        print("  🔍 感情分析モデル読み込み中...")
        self.sentiment_tokenizer, self.sentiment_model, self.device = load_sentiment_model()
        
        # スタンス検出モデル
        stance_model_dir = PROJECT_ROOT / "outputs" / "stance_model"
        if stance_model_dir.exists():
            print("  🎯 スタンス検出モデル読み込み中...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.stance_tokenizer = AutoTokenizer.from_pretrained(str(stance_model_dir))
            self.stance_model = AutoModelForSequenceClassification.from_pretrained(str(stance_model_dir))
            self.stance_model = self.stance_model.to(self.device)
            self.stance_model.eval()
        
        print(f"  ✅ モデル読み込み完了")
        print(f"  - 特徴量数: {len(self.feature_columns)}")
        print(f"  - 閾値: {self.metadata.get('optimal_threshold', 0.5)}")
    
    def _analyze_sentiment_batch(self, texts, batch_size=16):
        """バッチで感情分析"""
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            if batch_idx % 10 == 0:  # 10バッチごとに進捗表示
                print(f"    進捗: {batch_idx}/{total_batches} バッチ ({100*batch_idx/total_batches:.1f}%)")
            batch = texts[i:i+batch_size]
            
            inputs = self.sentiment_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            id2label = self.sentiment_model.config.id2label
            
            for prob in probs:
                result = {}
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'positive' in label_lower or 'ポジティブ' in label_lower:
                        result['positive'] = prob[idx].item()
                    elif 'negative' in label_lower or 'ネガティブ' in label_lower:
                        result['negative'] = prob[idx].item()
                    else:
                        result['neutral'] = prob[idx].item()
                
                result.setdefault('positive', 0.33)
                result.setdefault('neutral', 0.34)
                result.setdefault('negative', 0.33)
                
                max_key = max(['positive', 'neutral', 'negative'], key=lambda k: result.get(k, 0))
                result['label'] = max_key
                results.append(result)
        
        print(f"    ✅ 感情分析完了 ({len(results)}件)")
        return results
    
    def _analyze_stance_batch(self, texts, topic, batch_size=16):
        """バッチでスタンス検出"""
        if self.stance_model is None:
            return [{'label': 'NEUTRAL', 'favor': 0.33, 'against': 0.33, 'neutral': 0.34}] * len(texts)
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            if batch_idx % 10 == 0:  # 10バッチごとに進捗表示
                print(f"    進捗: {batch_idx}/{total_batches} バッチ ({100*batch_idx/total_batches:.1f}%)")
            batch = [f"{topic} [SEP] {t}" for t in texts[i:i+batch_size]]
            
            inputs = self.stance_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.stance_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for prob in probs:
                result = {
                    'favor': prob[0].item(),
                    'against': prob[1].item(),
                    'neutral': prob[2].item() if len(prob) > 2 else 0.0,
                }
                max_key = max(result.keys(), key=lambda k: result[k])
                result['label'] = max_key.upper()
                results.append(result)
        
        print(f"    ✅ スタンス検出完了 ({len(results)}件)")
        return results
    
    def analyze_tweets(self, tweets_df, topic=None):
        """
        ツイートデータを分析して特徴量を計算
        
        Args:
            tweets_df: ツイートのDataFrame（content, timestampカラムが必要）
            topic: トピック名（スタンス検出用）
        
        Returns:
            特徴量を含むDataFrame
        """
        print(f"\n📊 ツイート分析中... ({len(tweets_df)}件)")
        
        df = tweets_df.copy()
        
        # タイムスタンプの処理
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'created_at' in df.columns:
            df['timestamp'] = pd.to_datetime(df['created_at'])
        
        # テキストカラムの確認
        text_col = None
        for col in ['content', 'text', 'tweet']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError("テキストカラムが見つかりません（content, text, tweetのいずれか）")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        
        # 感情分析（バッチ処理）
        print("  🔍 感情分析中...")
        sentiments = self._analyze_sentiment_batch(texts)
        
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['negative_prob'] = [s['negative'] for s in sentiments]
        df['is_negative'] = df['sentiment_label'] == 'negative'
        
        # スタンス検出（バッチ処理）
        if topic:
            print(f"  🎯 スタンス検出中... (トピック: {topic})")
            stances = self._analyze_stance_batch(texts, topic)
            
            df['stance_label'] = [s['label'] for s in stances]
            df['stance_favor'] = [s['favor'] for s in stances]
            df['stance_against'] = [s['against'] for s in stances]
            df['stance_neutral'] = [s['neutral'] for s in stances]
        else:
            df['stance_label'] = 'NEUTRAL'
            df['stance_favor'] = 0.33
            df['stance_against'] = 0.33
            df['stance_neutral'] = 0.34
        
        return df
    
    def aggregate_features(self, analyzed_df, window='1h'):
        """
        分析済みデータを時間窓で集計して特徴量を作成
        
        Args:
            analyzed_df: analyze_tweetsの出力
            window: 集計ウィンドウ（デフォルト: 1時間）
        
        Returns:
            時間窓ごとの特徴量DataFrame
        """
        df = analyzed_df.copy()
        df = df.set_index('timestamp')
        
        # リサンプリング
        agg = df.resample(window).agg({
            'is_negative': ['count', 'mean'],
            'stance_favor': 'mean',
            'stance_against': 'mean',
            'stance_neutral': 'mean',
        })
        
        agg.columns = ['volume', 'negative_rate', 'stance_favor_rate', 'stance_against_rate', 'stance_neutral_rate']
        agg = agg.reset_index()
        
        # delta特徴量
        agg['delta_volume'] = agg['volume'].diff().fillna(0)
        agg['delta_negative_rate'] = agg['negative_rate'].diff().fillna(0)
        
        # 複合特徴量
        agg['flame_score'] = agg['volume'] * agg['negative_rate']
        agg['against_count'] = agg['volume'] * agg['stance_against_rate']
        agg['sentiment_polarity'] = agg['stance_against_rate'] - agg['stance_favor_rate']
        
        return agg
    
    def predict_score(self, features_df):
        """
        特徴量から炎上スコアを予測
        
        Args:
            features_df: 特徴量DataFrame
        
        Returns:
            炎上スコア（0-100）と判定結果のリスト
        """
        results = []
        
        for idx, row in features_df.iterrows():
            # 特徴量を準備
            X = pd.DataFrame([row[self.feature_columns]])
            X = X.fillna(0)
            
            # スケーリング
            X_scaled = self.scaler.transform(X)
            
            # 予測確率を取得
            proba = self.model.predict_proba(X_scaled)[0][1]
            
            # 0-100のスコアに変換
            score = int(proba * 100)
            
            # 判定
            if score >= 80:
                status = "🔴 炎上中"
                level = "critical"
            elif score >= 60:
                status = "🟠 炎上の可能性高"
                level = "high"
            elif score >= 40:
                status = "🟡 炎上の兆候あり"
                level = "medium"
            elif score >= 20:
                status = "🟢 やや注意"
                level = "low"
            else:
                status = "⚪ 平常"
                level = "normal"
            
            results.append({
                'timestamp': row.get('timestamp', idx),
                'score': score,
                'status': status,
                'level': level,
                'volume': row.get('volume', 0),
                'negative_rate': row.get('negative_rate', 0),
            })
        
        return results
    
    def predict_current(self, tweets_df, topic=None):
        """
        ツイートデータから現在の炎上スコアを予測（一括処理）
        
        Args:
            tweets_df: ツイートのDataFrame
            topic: トピック名
        
        Returns:
            最新の炎上スコアと判定結果
        """
        # 分析
        analyzed = self.analyze_tweets(tweets_df, topic)
        
        # 集計
        features = self.aggregate_features(analyzed)
        
        if len(features) == 0:
            return {
                'score': 0,
                'status': '⚪ データなし',
                'level': 'unknown'
            }
        
        # 最新のウィンドウで予測
        results = self.predict_score(features)
        
        # 最新の結果を返す
        latest = results[-1]
        
        return latest
    
    def predict_timeseries(self, tweets_df, topic=None):
        """
        時系列での炎上スコア推移を予測
        
        Args:
            tweets_df: ツイートのDataFrame
            topic: トピック名
        
        Returns:
            時系列の炎上スコアリスト
        """
        # 分析
        analyzed = self.analyze_tweets(tweets_df, topic)
        
        # 集計
        features = self.aggregate_features(analyzed)
        
        # 全時間窓で予測
        results = self.predict_score(features)
        
        return results


def print_score_gauge(score):
    """炎上スコアをゲージ表示"""
    bar_length = 50
    filled = int(score / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n  炎上スコア: [{bar}] {score}/100")


def main():
    """デモ実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='リアルタイム炎上スコア予測')
    parser.add_argument('input', help='入力CSVファイルのパス')
    parser.add_argument('--topic', '-t', help='トピック名（スタンス検出用）')
    parser.add_argument('--timeseries', '-ts', action='store_true', help='時系列で表示')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 リアルタイム炎上スコア予測")
    print("=" * 60)
    
    # データ読み込み
    df = pd.read_csv(args.input)
    print(f"\n📂 データ読み込み: {len(df)}件")
    
    # 予測器を初期化
    predictor = FlamePredictor()
    
    if args.timeseries:
        # 時系列予測
        results = predictor.predict_timeseries(df, args.topic)
        
        print("\n" + "=" * 60)
        print("📈 時系列炎上スコア")
        print("=" * 60)
        
        for r in results:
            ts = r['timestamp']
            if hasattr(ts, 'strftime'):
                ts = ts.strftime('%Y-%m-%d %H:%M')
            print(f"\n  [{ts}]")
            print(f"    {r['status']}")
            print_score_gauge(r['score'])
            print(f"    投稿数: {r['volume']:.0f}, ネガティブ率: {r['negative_rate']*100:.1f}%")
    else:
        # 現在のスコア
        result = predictor.predict_current(df, args.topic)
        
        print("\n" + "=" * 60)
        print("🎯 現在の炎上スコア")
        print("=" * 60)
        
        print(f"\n  {result['status']}")
        print_score_gauge(result['score'])
        
        if result['score'] >= 60:
            print("\n  ⚠️ 警告: 炎上リスクが高い状態です！")
        elif result['score'] >= 40:
            print("\n  📢 注意: 炎上の兆候が見られます。監視を続けてください。")
        else:
            print("\n  ✅ 現在は平常な状態です。")


if __name__ == "__main__":
    main()
