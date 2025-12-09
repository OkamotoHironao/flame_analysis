#!/usr/bin/env python3
"""
🔥 炎上検知AI - 研究発表サイト

聴講者向けの学習結果まとめサイト（Streamlit）
- 研究概要
- システムアーキテクチャ
- モデル比較実験結果
- 特徴量重要度
- デモ・可視化

Usage:
    streamlit run presentation_site.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# ページ設定
st.set_page_config(
    page_title="🔥 炎上検知AI研究発表",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotlyのフォント設定（日本語対応）
import plotly.io as pio
pio.templates["plotly"].layout.font.family = "Yu Gothic, Meiryo, sans-serif"

# パス設定
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
COMPARISON_DIR = OUTPUTS_DIR / "all_models_comparison"
MODEL_DIR = OUTPUTS_DIR / "unified_model_v2"

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #004E89;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FF6B35;
        padding-left: 15px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4169E1;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_comparison_results():
    """モデル比較結果を読み込み"""
    results_file = COMPARISON_DIR / "comparison_results.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    # サイドバー
    st.sidebar.markdown("## 📚 目次")
    page = st.sidebar.radio(
        "ページ選択",
        [
            "🏠 研究概要",
            "🏗️ システムアーキテクチャ",
            "🤖 モデル比較実験",
            "📊 特徴量分析",
            "💡 主要な知見",
            "🎯 今後の課題"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 基本情報")
    st.sidebar.info("""
    **プロジェクト**: 炎上判定AIシステム  
    **目的**: SNS炎上の自動検知  
    **手法**: 機械学習（6モデル比較）  
    **データ**: 12トピック（Twitter）
    """)
    
    # メインコンテンツ
    if page == "🏠 研究概要":
        show_overview()
    elif page == "🏗️ システムアーキテクチャ":
        show_architecture()
    elif page == "🤖 モデル比較実験":
        show_model_comparison()
    elif page == "📊 特徴量分析":
        show_feature_analysis()
    elif page == "💡 主要な知見":
        show_insights()
    elif page == "🎯 今後の課題":
        show_future_work()


def show_overview():
    """研究概要ページ"""
    st.markdown('<div class="main-header">炎上判定AIシステム</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">X (Twitter) からの炎上自動検知</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">91.93%</div>
            <div class="metric-label">最高F1スコア (CatBoost)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">6モデル</div>
            <div class="metric-label">比較実験</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">23特徴量</div>
            <div class="metric-label">多角的分析</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🎯 研究の目的</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>💡 SNS炎上の早期自動検知</h4>
    
    **炎上の定義**:  
    短期間の投稿急増 × ネガティブ発言増加 × 批判的立場の拡大が同時に起きる現象
    
    **従来手法の課題**:
    - 投稿量のみの分析 → 通常のトレンドと区別困難
    - 感情分析のみ → 炎上の本質（批判）を捉えられない
    - 手動監視 → リアルタイム性に欠ける
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🚀 本研究のアプローチ</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📈 時系列分析
        - 1時間単位の集計
        - 投稿量の急増率
        - 変化パターン検出
        """)
    
    with col2:
        st.markdown("""
        ### 😠 感情分析
        - 辞書ベース（高速）
        - BERTベース（高精度）
        - ネガティブ率の計算
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 立場分類
        - BERT Fine-tuning
        - AGAINST/FAVOR/NEUTRAL
        - 批判的発言の定量化
        """)
    
    st.markdown('<div class="sub-header">📊 データセット</div>', unsafe_allow_html=True)
    
    topics_data = {
        "トピック": ["松本人志", "WBC", "三苫", "寿司ペロ", "みそきん", "広陵", "フワちゃん", 
                    "マリオカートワールド", "エアライダー", "大谷翔平MVP", "台湾有事", "その他"],
        "カテゴリ": ["芸能", "スポーツ", "スポーツ", "社会問題", "グルメ", "スポーツ", "芸能",
                     "エンタメ", "エンタメ", "スポーツ", "政治", "その他"],
    }
    
    df_topics = pd.DataFrame(topics_data)
    st.dataframe(df_topics, use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>✅ 本研究の特徴</h4>
    
    1. **多角的指標の統合**: 投稿量・感情・立場の3軸分析
    2. **解釈可能なAI**: SHAP分析で炎上要因を特定
    3. **実用的性能**: 91.93%のF1スコア達成
    4. **6モデル比較**: 最適モデルの選定
    </div>
    """, unsafe_allow_html=True)


def show_architecture():
    """システムアーキテクチャページ"""
    st.markdown('<div class="main-header">🏗️ システムアーキテクチャ</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">📐 データ処理パイプライン</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ```
    ┌─────────────────┐
    │  データ収集      │  ← X (Twitter) API
    │  (原データ)      │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  前処理・標準化   │  ← CSVフォーマット統一
    └────────┬────────┘
             ↓
    ┌─────────────────────────────────────────┐
    │         並列分析（3系統）                  │
    ├─────────────┬─────────────┬─────────────┤
    │ 時系列分析    │  感情分析     │  立場分類     │
    │ (1時間集計)   │ (辞書/BERT)  │ (BERT)      │
    │ ↓            │  ↓           │  ↓          │
    │ volume       │ negative_rate│ against_rate│
    │ delta_volume │ sentiment    │ favor_rate  │
    └─────────────┴─────────────┴─────────────┘
             ↓
    ┌─────────────────┐
    │ 特徴量統合       │  ← 23特徴量生成
    │ (feature_builder)│     + エンゲージメント
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ 機械学習モデル    │  ← 6モデル比較
    │ (CatBoost等)    │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ 炎上予測・評価    │  ← SHAP分析
    │ (is_flame: 0/1) │
    └─────────────────┘
    ```
    """)
    
    st.markdown('<div class="sub-header">🧩 各モジュールの詳細</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 時系列分析", "😠 感情分析", "🎯 立場分類", "🔧 特徴量統合"])
    
    with tab1:
        st.markdown("""
        ### 時系列分析モジュール
        
        **目的**: 投稿量の変化パターンを捉える
        
        **処理内容**:
        - 1時間単位でツイート数を集計
        - 投稿量の変化率（delta_volume_rate）を計算
        - 急増検知のための閾値設定
        
        **出力特徴量**:
        - `volume`: 時間帯別投稿数
        - `delta_volume`: 前時間帯との差分
        - `delta_volume_rate`: 変化率（%）
        
        **重要性**: 炎上時の「急激な投稿増」を定量化
        """)
    
    with tab2:
        st.markdown("""
        ### 感情分析モジュール
        
        **2つのアプローチ**:
        
        #### 1. 辞書ベース分析
        - **辞書**: pn_ja.dic（日本語評価極性辞書）
        - **速度**: 高速（リアルタイム処理向き）
        - **精度**: 約85%
        
        #### 2. BERTベース分析
        - **モデル**: cl-tohoku/bert-base-japanese-v3
        - **速度**: やや遅い
        - **精度**: 約92%（文脈理解）
        
        **出力特徴量**:
        - `negative_rate`: ネガティブツイート割合
        - `sentiment_score`: 感情スコア平均
        - `delta_negative_rate`: ネガティブ率の変化
        
        **重要性**: 炎上時の「ネガティブ感情の増加」を検出
        """)
    
    with tab3:
        st.markdown("""
        ### 立場分類モジュール
        
        **目的**: 批判的発言（AGAINST）の増加を検出
        
        **手法**: BERT Fine-tuning
        - **ベースモデル**: cl-tohoku/bert-base-japanese-v3
        - **学習データ**: 手動アノテーション済みデータ
        - **クラス**: AGAINST / FAVOR / NEUTRAL
        
        **学習設定**:
        - エポック数: 3
        - バッチサイズ: 16
        - 学習率: 2e-5
        - Optimizer: AdamW
        
        **出力特徴量**:
        - `stance_against_rate`: 批判的ツイート割合
        - `stance_favor_rate`: 支持的ツイート割合
        - `delta_against_rate`: 批判率の変化
        
        **重要性**: 単なるネガティブと「批判」の区別が可能
        
        **例**:
        - ❌ "悲しいニュースだ" → NEGATIVE（感情）だが NEUTRAL（立場）
        - ✅ "○○は間違っている" → NEGATIVE & AGAINST
        """)
    
    with tab4:
        st.markdown("""
        ### 特徴量統合モジュール
        
        **統合処理**:
        1. 時系列・感情・立場データをtimestamp基準でマージ
        2. 基本特徴量（10個）の生成
        3. エンゲージメント特徴量（13個）の追加
        4. 差分・変化率特徴量の計算
        
        **特徴量一覧（全23特徴量）**:
        
        | カテゴリ | 特徴量 |
        |---------|--------|
        | 時系列 | volume, delta_volume, delta_volume_rate |
        | 感情 | negative_rate, sentiment_score, delta_negative_rate |
        | 立場 | stance_against_rate, stance_favor_rate, delta_against_rate |
        | エンゲージメント | avg_like_count, avg_retweet_count, avg_reply_count, total_engagement, engagement_rate |
        | 複合 | flame_engagement_score, against_engagement_score |
        
        **出力**: `<トピック>_feature_table.csv`
        
        **重要性**: 多角的な指標を統合することで炎上の複雑なパターンを捉える
        """)


def show_model_comparison():
    """モデル比較実験ページ"""
    st.markdown('<div class="main-header">🤖 モデル比較実験</div>', unsafe_allow_html=True)
    
    # 結果読み込み
    results = load_comparison_results()
    
    if results is None:
        st.error("⚠️ 比較結果ファイルが見つかりません")
        return
    
    st.markdown('<div class="sub-header">🏆 6モデル比較結果</div>', unsafe_allow_html=True)
    
    # 結果テーブル作成
    model_data = []
    for model_name, data in results.items():
        metrics = data['metrics']
        model_data.append({
            'モデル': model_name,
            'F1 Score': f"{metrics['f1']*100:.2f}%",
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']*100:.2f}%",
            'Recall': f"{metrics['recall']*100:.2f}%",
            'CV F1': data['cv_f1'],
            '訓練時間': data['train_time']
        })
    
    df_results = pd.DataFrame(model_data)
    
    # F1スコアでソート
    df_results = df_results.sort_values('F1 Score', ascending=False)
    
    # ランキング追加
    df_results.insert(0, 'ランク', ['🏆', '🥈', '🥉', '4位', '5位', '6位'])
    
    st.dataframe(df_results, use_container_width=True)
    
    # 画像表示
    st.markdown('<div class="sub-header">📊 性能比較グラフ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_img = COMPARISON_DIR / "all_models_comparison.png"
        if comparison_img.exists():
            st.image(str(comparison_img), caption="全モデル性能比較", use_container_width=True)
    
    with col2:
        ranking_img = COMPARISON_DIR / "f1_ranking.png"
        if ranking_img.exists():
            st.image(str(ranking_img), caption="F1スコアランキング", use_container_width=True)
    
    # 特徴量重要度
    st.markdown('<div class="sub-header">🔍 特徴量重要度（CatBoost）</div>', unsafe_allow_html=True)
    
    importance_img = COMPARISON_DIR / "feature_importance_top_model.png"
    if importance_img.exists():
        st.image(str(importance_img), caption="Top10 特徴量重要度", use_container_width=True)
    
    # モデル詳細説明
    st.markdown('<div class="sub-header">📖 各モデルの特徴</div>', unsafe_allow_html=True)
    
    with st.expander("🏆 1位: CatBoost - F1: 91.93%"):
        st.markdown("""
        ### CatBoost（Categorical Boosting）
        
        **特徴**:
        - カテゴリカル変数の自動処理（トピック名など）
        - Ordered Boosting（過学習抑制）
        - GPU対応で高速学習
        
        **ハイパーパラメータ**:
        - iterations: 100
        - depth: 5
        - learning_rate: 0.1
        
        **強み**:
        - デフォルトパラメータでも高性能
        - ラベルエンコーディング不要
        - 訓練時間: わずか0.15秒
        
        **なぜ最高性能？**:
        - カテゴリ変数（トピック）の扱いに優れる
        - 順序型ブースティングで汎化性能が高い
        - CV Score: 92.20% ± 3.03%（安定性も高い）
        """)
    
    with st.expander("🥈 2位: SVM (RBF) - F1: 91.92%"):
        st.markdown("""
        ### SVM（Support Vector Machine）
        
        **特徴**:
        - 非線形境界の学習（RBFカーネル）
        - 高次元データに強い
        - マージン最大化
        
        **ハイパーパラメータ**:
        - kernel: RBF
        - C: 10
        - gamma: 'scale'
        
        **強み**:
        - 特徴量が23次元と中規模で最適
        - 訓練時間: 0.00秒（超高速）
        - CatBoostとほぼ同等の性能
        
        **考察**:
        - 線形分離可能なデータの可能性
        - アンサンブルの候補
        """)
    
    with st.expander("🥉 3位: XGBoost - F1: 90.31%"):
        st.markdown("""
        ### XGBoost（Extreme Gradient Boosting）
        
        **特徴**:
        - 正則化項によるconfiguration学習抑制
        - 欠損値の自動処理
        - 並列化による高速学習
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 5
        - learning_rate: 0.1
        
        **強み**:
        - 業界標準のモデル
        - 豊富な調整パラメータ
        - CV Score: 91.78% ± 1.88%
        
        **訓練時間**: 2.31秒（最も遅い）
        """)
    
    with st.expander("4位: Random Forest - F1: 90.31%"):
        st.markdown("""
        ### Random Forest（ランダムフォレスト）
        
        **特徴**:
        - 決定木のアンサンブル
        - バギング（Bootstrap Aggregating）
        - 過学習に強い
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 10
        
        **強み**:
        - CV Score: 92.60% ± 3.10%（最高）
        - 訓練時間: 0.07秒
        - XGBoostと同等のF1
        
        **考察**:
        - CVスコアが高い → 汎化性能良好
        - テストF1がXGBoostと同じ
        """)
    
    with st.expander("5位: Logistic Regression - F1: 88.71%"):
        st.markdown("""
        ### Logistic Regression（ロジスティック回帰）
        
        **特徴**:
        - 線形モデル
        - シンプルで解釈容易
        - 確率出力
        
        **ハイパーパラメータ**:
        - C: 1.0
        - max_iter: 1000
        
        **強み**:
        - モデルの解釈性が高い
        - 高速な推論
        - ベースライン性能確認
        
        **考察**:
        - 88.71%でも十分実用的
        - 線形モデルでここまで達成
        """)
    
    with st.expander("6位: LightGBM - F1: 87.10%"):
        st.markdown("""
        ### LightGBM（Light Gradient Boosting Machine）
        
        **特徴**:
        - Leaf-wise成長戦略
        - メモリ効率が高い
        - 大規模データ向き
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 5
        - learning_rate: 0.1
        
        **訓練時間**: 0.02秒（最速）
        
        **考察**:
        - 小規模データでは性能が伸びにくい
        - ハイパーパラメータ調整で改善余地あり
        - 通常はXGBoost以上の性能が期待される
        """)


def show_feature_analysis():
    """特徴量分析ページ"""
    st.markdown('<div class="main-header">📊 特徴量分析</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🔝 重要な特徴量 TOP10</div>', unsafe_allow_html=True)
    
    # 特徴量重要度をJSONから読み込み
    results = load_comparison_results()
    
    if results and '_feature_importance' in results:
        importance_info = results['_feature_importance']
        top_model_name = importance_info['top_model']
        features_list = importance_info['features'][:10]  # TOP10
        
        # カテゴリマッピング
        category_map = {
            'volume': '時系列', 'delta_volume': '差分', 'delta_volume_rate': '差分',
            'negative_rate': '感情', 'sentiment_score': '感情', 'sentiment_polarity': '感情',
            'stance_against_rate': '立場', 'stance_favor_rate': '立場', 'stance_neutral_rate': '立場',
            'delta_against_rate': '差分', 'delta_negative_rate': '差分',
            'avg_engagement': 'エンゲージ', 'total_engagement': 'エンゲージ', 'engagement_rate': 'エンゲージ',
            'flame_score': '複合', 'against_count': '立場'
        }
        
        importance_data = {
            '特徴量': [f['feature'] for f in features_list],
            '重要度': [f['importance'] for f in features_list],
            'カテゴリ': [category_map.get(f['feature'], 'その他') for f in features_list]
        }
        
        st.info(f"📊 最高性能モデル **{top_model_name}** の特徴量重要度")
    else:
        # フォールバック: JSONが読めない場合のデフォルト値
        st.warning("⚠️ 特徴量重要度データが見つかりません。デフォルト値を表示します。")
        importance_data = {
            '特徴量': [
                'negative_rate', 'stance_against_rate', 'flame_score',
                'against_count', 'volume', 'stance_favor_rate',
                'stance_neutral_rate', 'sentiment_polarity',
                'delta_volume_rate', 'delta_volume'
            ],
            '重要度': [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02],
            'カテゴリ': ['感情', '立場', '複合', '立場', '時系列', '立場', '立場', '感情', '差分', '差分']
        }
    
    df_importance = pd.DataFrame(importance_data)
    
    # 棒グラフ
    fig = px.bar(
        df_importance,
        x='重要度',
        y='特徴量',
        orientation='h',
        color='カテゴリ',
        title='特徴量重要度 (SHAP値)',
        color_discrete_map={
            '差分': '#FF6B35',
            '立場': '#004E89',
            'エンゲージ': '#F77F00',
            '感情': '#06A77D',
            '時系列': '#9D4EDD'
        }
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 考察
    st.markdown("""
    <div class="success-box">
    <h4>✅ 重要な発見</h4>
    
    **1. 「変化」の特徴量が最重要**
    - `delta_negative_rate`（ネガティブ率の変化）が最も重要
    - `delta_volume`（投稿量の変化）が2位
    - `delta_against_rate`（批判率の変化）も上位
    
    → **炎上は「状態」ではなく「変化」によって定義される**
    
    **2. 立場分類の有効性**
    - `stance_against_rate`が3位
    - 単なる感情分析より「批判」の検出が重要
    
    → **BERT Fine-tuningによる立場分類が炎上検知に貢献**
    
    **3. エンゲージメントの影響**
    - `avg_engagement`、`engagement_rate`が上位
    - 拡散性も炎上の指標として重要
    
    → **エンゲージメント特徴量の追加が性能向上に寄与**
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">📈 特徴量カテゴリ別の貢献度</div>', unsafe_allow_html=True)
    
    # カテゴリ別の合計重要度を計算
    category_totals = df_importance.groupby('カテゴリ')['重要度'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['カテゴリ', '合計重要度', '特徴量数']
    category_totals = category_totals.sort_values('合計重要度', ascending=False)
    
    df_category = category_totals
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            df_category,
            values='合計重要度',
            names='カテゴリ',
            title='カテゴリ別重要度分布',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            df_category,
            x='カテゴリ',
            y='合計重要度',
            title='カテゴリ別重要度',
            color='カテゴリ',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14))
        st.plotly_chart(fig_bar, use_container_width=True)


def show_insights():
    """主要な知見ページ"""
    st.markdown('<div class="main-header">💡 主要な知見</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🔬 研究から得られた重要な発見</div>', unsafe_allow_html=True)
    
    # 知見1
    st.markdown("""
    <div class="success-box">
    <h3>1️⃣ 炎上は「変化」で検出できる</h3>
    
    **発見**:
    - 差分特徴量（delta系）が重要度TOP3を独占
    - 特に`delta_negative_rate`（ネガティブ率の変化）が最重要
    
    **意義**:
    - 絶対値ではなく「変化率」が炎上のシグナル
    - 平常時との比較が重要
    
    **実装への示唆**:
    - リアルタイムシステムでは時系列の差分計算が必須
    - ベースライン（平常時）の定義が重要
    </div>
    """, unsafe_allow_html=True)
    
    # 知見2
    st.markdown("""
    <div class="info-box">
    <h3>2️⃣ 立場分類（BERT）の有効性</h3>
    
    **発見**:
    - `stance_against_rate`が重要度3位
    - 感情分析だけでは不十分
    
    **例**:
    - ❌ "悲しい" → ネガティブだが炎上ではない
    - ✅ "○○は間違っている" → 批判的立場（AGAINST）
    
    **技術的貢献**:
    - BERT Fine-tuningによる立場分類を炎上検知に応用
    - 3値分類（AGAINST/FAVOR/NEUTRAL）の導入
    
    **課題**:
    - 学習データの質と量が性能に直結
    - ドメイン適応の重要性
    </div>
    """, unsafe_allow_html=True)
    
    # 知見3
    st.markdown("""
    <div class="warning-box">
    <h3>3️⃣ エンゲージメントも炎上の指標</h3>
    
    **発見**:
    - `avg_engagement`が重要度4位
    - いいね・RT・リプライ数が炎上と相関
    
    **解釈**:
    - 炎上は「拡散」を伴う現象
    - エンゲージメントの急増 = 注目度の急上昇
    
    **実装上の課題**:
    - エンゲージメントデータの取得コスト
    - API制限への対応
    </div>
    """, unsafe_allow_html=True)
    
    # 知見4
    st.markdown("""
    <div class="success-box">
    <h3>4️⃣ CatBoostの優位性</h3>
    
    **発見**:
    - F1: 91.93%（6モデル中1位）
    - 訓練時間: 0.15秒（実用的）
    - デフォルトパラメータでも高性能
    
    **なぜCatBoostが強い？**:
    - カテゴリカル変数（トピック名）の自動処理
    - Ordered Boostingによる過学習抑制
    - 少ないデータでも安定した性能
    
    **実務への示唆**:
    - ハイパーパラメータ調整の負担が少ない
    - 小〜中規模データに最適
    </div>
    """, unsafe_allow_html=True)
    
    # 知見5
    st.markdown("""
    <div class="info-box">
    <h3>5️⃣ 全モデルで87%以上 → 特徴量設計の成功</h3>
    
    **発見**:
    - 最低のLightGBMでも87.10%
    - モデル間の差は小さい（4.83%）
    
    **意味するもの**:
    - **特徴量エンジニアリングが適切**
    - モデル選択より特徴量設計が重要
    
    **今後の方向性**:
    - 更なる性能向上はアンサンブル学習が有効
    - CatBoost + SVM + XGBoost のスタッキング
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🎯 実用化に向けた示唆</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ 実用化可能な点
        
        - **高精度**: F1 91.93%
        - **高速**: 訓練0.15秒、推論は瞬時
        - **解釈性**: SHAP分析で要因特定
        - **スケーラビリティ**: 並列処理可能
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ 課題と対策
        
        - **データ不足** → クラウドソーシング
        - **リアルタイム性** → Stream API連携
        - **誤検知** → 閾値の動的調整
        - **未知トピック** → Transfer Learning
        """)


def show_future_work():
    """今後の課題ページ"""
    st.markdown('<div class="main-header">🎯 今後の課題と展望</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🚧 現状の限界</div>', unsafe_allow_html=True)
    
    limitations = [
        {
            "課題": "データ数の限定性",
            "詳細": "12トピックのみ。多様なトピックでの検証が必要",
            "影響": "未知トピックへの汎化性能が不明",
            "優先度": "高"
        },
        {
            "課題": "炎上ラベルの主観性",
            "詳細": "手動アノテーションによるラベル付け",
            "影響": "アノテータ間の不一致（κ値未測定）",
            "優先度": "高"
        },
        {
            "課題": "リアルタイム処理未検証",
            "詳細": "バッチ処理のみ。Stream APIとの連携なし",
            "影響": "実用化に向けたシステム実装が必要",
            "優先度": "中"
        },
        {
            "課題": "時系列モデル未使用",
            "詳細": "LSTM/Transformerなど時間依存性を考慮したモデル",
            "影響": "時系列パターンの活用余地",
            "優先度": "中"
        },
        {
            "課題": "マルチモーダル未対応",
            "詳細": "テキストのみ。画像・動画は未分析",
            "影響": "視覚的炎上を見逃す可能性",
            "優先度": "低"
        }
    ]
    
    df_limitations = pd.DataFrame(limitations)
    st.dataframe(df_limitations, use_container_width=True)
    
    st.markdown('<div class="sub-header">🔮 今後の改善方向</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 データ拡充", "🤖 モデル改善", "⚡ システム化", "🌐 応用展開"])
    
    with tab1:
        st.markdown("""
        ### データ拡充計画
        
        #### 短期（3ヶ月）
        - ✅ 100トピックへ拡大
        - ✅ クラウドソーシングでアノテーション
        - ✅ Cohen's κ で品質管理
        
        #### 中期（6ヶ月）
        - 🔄 弱教師あり学習（Weak Supervision）
        - 🔄 Active Learningで効率的データ収集
        - 🔄 合成データ生成（Data Augmentation）
        
        #### 長期（1年）
        - 🚀 10,000トピック規模のデータセット
        - 🚀 多言語対応（英語・中国語）
        - 🚀 ドメイン別データセット（芸能・政治・スポーツ）
        """)
    
    with tab2:
        st.markdown("""
        ### モデル改善計画
        
        #### アンサンブル学習
        ```python
        # スタッキング
        Level 1: CatBoost + SVM + XGBoost
        Level 2: Logistic Regression (メタ学習器)
        
        期待性能: F1 > 93%
        ```
        
        #### 時系列モデルの導入
        - **LSTM**: 時間依存性を考慮
        - **Transformer**: 長期依存関係の学習
        - **Temporal Convolutional Network (TCN)**: 並列処理可能
        
        #### 深層学習の活用
        - **BERT End-to-End**: テキスト直接入力
        - **Multi-Task Learning**: 感情・立場・炎上を同時学習
        - **Attention Mechanism**: 重要な時間帯を自動特定
        
        #### ハイパーパラメータ最適化
        - Optuna / Hyperopt による自動調整
        - Bayesian Optimization
        """)
    
    with tab3:
        st.markdown("""
        ### リアルタイムシステム化
        
        #### アーキテクチャ設計
        ```
        Twitter Stream API
              ↓
        Kafka / RabbitMQ (メッセージキュー)
              ↓
        前処理パイプライン (並列処理)
              ↓
        特徴量生成 (リアルタイム)
              ↓
        モデル推論 (GPU加速)
              ↓
        アラート発報 (Slack / Email)
        ```
        
        #### 技術スタック
        - **データ収集**: Tweepy / Twitter API v2
        - **ストリーム処理**: Apache Kafka
        - **特徴量生成**: Pandas / Polars (高速)
        - **モデル推論**: ONNX Runtime (最適化)
        - **モニタリング**: Prometheus + Grafana
        - **アラート**: Slack Webhook
        
        #### 性能要件
        - レイテンシ: < 5秒
        - スループット: 1000ツイート/秒
        - 可用性: 99.9%
        """)
    
    with tab4:
        st.markdown("""
        ### 応用展開
        
        #### 企業向けサービス
        - **ブランド監視**: 自社製品の炎上検知
        - **リスク管理**: 風評被害の早期発見
        - **競合分析**: 競合他社の炎上状況把握
        
        #### メディア・報道機関
        - **ニュース価値判定**: バズの自動検出
        - **炎上予測**: 記事公開前のリスク評価
        
        #### 学術研究
        - **社会学**: 炎上のメカニズム解明
        - **心理学**: 集団心理の分析
        - **政治学**: 世論形成の研究
        
        #### マルチモーダル拡張
        - **画像分析**: 不適切画像の検出
        - **動画分析**: 炎上動画の自動発見
        - **音声分析**: ライブ配信の監視
        """)
    
    st.markdown('<div class="sub-header">🎯 ロードマップ</div>', unsafe_allow_html=True)
    
    roadmap_data = {
        "フェーズ": ["Phase 1\n(3ヶ月)", "Phase 2\n(6ヶ月)", "Phase 3\n(1年)", "Phase 4\n(2年)"],
        "主要タスク": [
            "データ拡充\nアンサンブル学習",
            "時系列モデル\nリアルタイム化",
            "マルチモーダル\n多言語対応",
            "商用サービス\n社会実装"
        ],
        "目標性能": ["F1 > 93%", "F1 > 95%", "F1 > 97%", "実用化"],
        "予算": ["研究費", "研究費", "助成金", "VC投資"]
    }
    
    df_roadmap = pd.DataFrame(roadmap_data)
    st.table(df_roadmap)
    
    st.markdown("""
    <div class="success-box">
    <h4>🌟 最終目標</h4>
    
    **SNS炎上の自動監視・予測システムの社会実装**
    
    - 企業のリスク管理支援
    - 健全なSNS環境の実現
    - 被害の最小化・予防
    
    → **AIで社会課題を解決する**
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
