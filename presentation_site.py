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
CONFIG_FILE = BASE_DIR / "config" / "presentation_config.json"

# 設定ファイル読み込み
def load_config():
    """プレゼンテーション設定を読み込み"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

CONFIG = load_config()

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
            "🔬 統合モデル比較",
            "📊 特徴量分析",
            "💡 主要な知見"
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
    elif page == "🔬 統合モデル比較":
        show_unified_models_comparison()
    elif page == "📊 特徴量分析":
        show_feature_analysis()
    elif page == "💡 主要な知見":
        show_insights()


def show_overview():
    """研究概要ページ"""
    st.markdown('<div class="main-header">炎上判定AIシステム</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">X (Twitter) からの炎上自動検知</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 最新結果を読み込み
    results = load_comparison_results()
    
    # デフォルト値を設定ファイルから取得
    if CONFIG:
        best_f1 = CONFIG['metrics']['default_best_f1']
        best_model = CONFIG['metrics']['default_best_model']
        num_features = CONFIG['metrics']['num_features']
        num_models = CONFIG['metrics']['num_models_compared']
    else:
        best_f1 = 91.93
        best_model = "CatBoost"
        num_features = 16
        num_models = 6
    
    if results:
        # 全モデルのF1スコアを取得して最高値を見つける
        f1_scores = {}
        for model_name, data in results.items():
            if model_name != '_feature_importance' and 'metrics' in data:
                f1_scores[model_name] = data['metrics']['f1']
        
        if f1_scores:
            best_model = max(f1_scores, key=f1_scores.get)
            best_f1 = f1_scores[best_model] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_f1:.2f}%</div>
            <div class="metric-label">最高F1スコア ({best_model})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{num_models}モデル</div>
            <div class="metric-label">比較実験</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{num_features}特徴量</div>
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
    
    # 最新の性能指標を取得
    latest_f1 = CONFIG['metrics']['latest_best_f1'] if CONFIG else 93.54
    cross_topic_f1 = CONFIG['metrics']['cross_topic_f1'] if CONFIG else 50.21
    latest_models = CONFIG['metrics'].get('latest_best_models', ['LightGBM', 'CatBoost', 'Random Forest']) if CONFIG else ['LightGBM', 'CatBoost', 'Random Forest']
    cross_model = CONFIG['metrics'].get('cross_topic_best_model', 'Logistic Regression') if CONFIG else 'Logistic Regression'
    unified_f1 = CONFIG['metrics'].get('unified_model_best_f1', 96.88) if CONFIG else 96.88
    unified_model = CONFIG['metrics'].get('unified_model_best_model', 'LightGBM') if CONFIG else 'LightGBM'
    
    st.markdown(f"""
    <div class="success-box">
    <h4>✅ 本研究の特徴と成果</h4>
    
    1. **多角的指標の統合**: 時系列・感情・立場の{num_features}特徴量による分析
    2. **解釈可能なAI**: 特徴量重要度分析で炎上要因を特定可能
    3. **高精度達成**: 
       - 標準評価（同一トピック内）: **{latest_f1}%** ({', '.join(latest_models[:2])}等)
       - **統合モデル（閾値最適化）**: **{unified_f1}%** ({unified_model})
       - クロストピック評価（未知トピック）: **{cross_topic_f1}%** ({cross_model})
    4. **{num_models}モデル比較**: 最適モデルの選定とベンチマーク確立
    5. **実用的性能**: Precision 100%達成でビジネス適用可能
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
    │ 炎上予測・評価    │  ← 特徴量重要度分析
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
        if model_name != '_feature_importance' and 'metrics' in data:
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
    
    # F1スコアでソート（文字列なので数値に変換）
    df_results['_f1_num'] = df_results['F1 Score'].str.replace('%', '').astype(float)
    df_results = df_results.sort_values('_f1_num', ascending=False)
    df_results = df_results.drop(columns=['_f1_num'])
    
    # ランキング追加
    df_results.insert(0, 'ランク', ['🏆', '🥈', '🥉', '4位', '5位', '6位'])
    
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
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
    
    # モデル詳細説明（動的に順位とF1スコアを取得）
    st.markdown('<div class="sub-header">📖 各モデルの特徴</div>', unsafe_allow_html=True)
    
    results = load_comparison_results()
    model_details = {}
    
    if results:
        for model_name, data in results.items():
            if model_name != '_feature_importance' and 'metrics' in data:
                model_details[model_name] = {
                    'f1': data['metrics']['f1'] * 100,
                    'cv': data.get('cv_f1', 'N/A'),
                    'train_time': data.get('train_time', 'N/A')
                }
    
    # F1スコアでソート
    sorted_models = sorted(model_details.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # ランキングアイコン
    rank_icons = {0: "🏆", 1: "🥈", 2: "🥉", 3: "4位", 4: "5位", 5: "6位"}
    
    # CatBoost
    rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'CatBoost':
            rank = i
            break
    f1_score = model_details.get('CatBoost', {}).get('f1', 0)
    cv_score = model_details.get('CatBoost', {}).get('cv', 'N/A')
    train_time = model_details.get('CatBoost', {}).get('train_time', 'N/A')
    
    with st.expander(f"{rank_icons.get(rank, str(rank+1)+'位')}: CatBoost - F1: {f1_score:.2f}%"):
        st.markdown(f"""
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
        - 訓練時間: わずか{train_time}
        
        **なぜ最高性能？**:
        - カテゴリ変数（トピック）の扱いに優れる
        - 順序型ブースティングで汎化性能が高い
        - CV Score: {cv_score}（安定性も高い）
        """)
    
    # SVM
    svm_rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'SVM (RBF)':
            svm_rank = i
            break
    svm_f1 = model_details.get('SVM (RBF)', {}).get('f1', 0)
    svm_time = model_details.get('SVM (RBF)', {}).get('train_time', 'N/A')
    
    with st.expander(f"{rank_icons.get(svm_rank, str(svm_rank+1)+'位')}: SVM (RBF) - F1: {svm_f1:.2f}%"):
        st.markdown(f"""
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
        - 訓練時間: {svm_time}（超高速）
        - CatBoostとほぼ同等の性能
        
        **考察**:
        - 線形分離可能なデータの可能性
        - アンサンブルの候補
        """)
    
    # クロストピック評価結果を追加
    st.markdown('<div class="sub-header">🌐 クロストピック評価（汎化性能）</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Leave-One-Topic-Out クロスバリデーション</h4>
    
    **評価方法**:
    - 1つのトピックをテストデータ、残りを訓練データとして評価
    - 未知のトピックに対する汎化性能を測定
    - 5トピック × 6モデル = 30通りの実験
    
    **目的**:
    - 実世界での適用可能性の検証
    - トピック依存性の評価
    </div>
    """, unsafe_allow_html=True)
    
    # クロストピック結果の表示
    cross_topic_data = {
        'モデル': ['Logistic Regression', 'Random Forest', 'LightGBM', 'CatBoost', 'XGBoost', 'SVM (RBF)'],
        '平均F1': ['50.21%', '49.46%', '49.39%', '49.08%', '35.28%', '25.90%'],
        '標準偏差': ['±46.94%', '±46.46%', '±46.43%', '±46.17%', '±46.78%', '±40.46%'],
        '評価': ['🏆 最高', '🥈 2位', '🥉 3位', '4位', '5位', '6位']
    }
    
    df_cross = pd.DataFrame(cross_topic_data)
    st.dataframe(df_cross, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ 重要な発見</h4>
    
    **1. 汎化性能の大幅な低下**:
    - 標準評価（93.54%）→ クロストピック評価（50.21%）
    - トピック間の特徴分布の違いが顕著
    
    **2. モデル順位の逆転現象**:
    - Logistic Regression が1位（標準評価では下位）
    - 線形モデルの方が汎化性能が高い
    - ツリー系モデルは訓練データへの過適合傾向
    
    **3. スポーツトピックの困難性**:
    - WBC・三苫: F1スコア 0.00%（全モデル）
    - ドメイン特性の違いが大きい
    
    **4. 実用化への示唆**:
    - 新規トピックには追加学習（Fine-tuning）が必須
    - Transfer Learning の活用
    - トピック固有の特徴量設計が重要
    </div>
    """, unsafe_allow_html=True)

    
    # XGBoost
    xgb_rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'XGBoost':
            xgb_rank = i
            break
    xgb_f1 = model_details.get('XGBoost', {}).get('f1', 90.31)
    xgb_cv = model_details.get('XGBoost', {}).get('cv', '91.78 ± 1.88%')
    xgb_time = model_details.get('XGBoost', {}).get('train_time', '3.65秒')
    
    with st.expander(f"{rank_icons.get(xgb_rank, str(xgb_rank+1)+'位')}: XGBoost - F1: {xgb_f1:.2f}%"):
        st.markdown(f"""
        ### XGBoost（Extreme Gradient Boosting）
        
        **特徴**:
        - 正則化項による過学習抑制
        - 欠損値の自動処理
        - 並列化による高速学習
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 5
        - learning_rate: 0.1
        
        **強み**:
        - 業界標準のモデル
        - 豊富な調整パラメータ
        - CV Score: {xgb_cv}
        
        **訓練時間**: {xgb_time}
        """)
    
    # Random Forest
    rf_rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'Random Forest':
            rf_rank = i
            break
    rf_f1 = model_details.get('Random Forest', {}).get('f1', 90.31)
    rf_cv = model_details.get('Random Forest', {}).get('cv', '92.60 ± 3.10%')
    rf_time = model_details.get('Random Forest', {}).get('train_time', '0.07秒')
    
    with st.expander(f"{rank_icons.get(rf_rank, str(rf_rank+1)+'位')}: Random Forest - F1: {rf_f1:.2f}%"):
        st.markdown(f"""
        ### Random Forest（ランダムフォレスト）
        
        **特徴**:
        - 決定木のアンサンブル
        - バギング（Bootstrap Aggregating）
        - 過学習に強い
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 10
        
        **強み**:
        - CV Score: {rf_cv}（最高）
        - 訓練時間: {rf_time}
        - XGBoostと同等のF1
        
        **考察**:
        - CVスコアが高い → 汎化性能良好
        - テストF1がXGBoostと同じ
        """)
    
    # Logistic Regression
    lr_rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'Logistic Regression':
            lr_rank = i
            break
    lr_f1 = model_details.get('Logistic Regression', {}).get('f1', 88.71)
    
    with st.expander(f"{rank_icons.get(lr_rank, str(lr_rank+1)+'位')}: Logistic Regression - F1: {lr_f1:.2f}%"):
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
    
    # LightGBM
    lgb_rank = 0
    for i, (name, _) in enumerate(sorted_models):
        if name == 'LightGBM':
            lgb_rank = i
            break
    lgb_f1 = model_details.get('LightGBM', {}).get('f1', 87.10)
    lgb_cv = model_details.get('LightGBM', {}).get('cv', '92.20 ± 2.71%')
    lgb_time = model_details.get('LightGBM', {}).get('train_time', '0.01秒')
    
    with st.expander(f"{rank_icons.get(lgb_rank, str(lgb_rank+1)+'位')}: LightGBM - F1: {lgb_f1:.2f}%"):
        st.markdown(f"""
        ### LightGBM（Light Gradient Boosting Machine）
        
        **特徴**:
        - Leaf-wise成長戦略
        - メモリ効率が高い
        - 大規模データ向き
        
        **ハイパーパラメータ**:
        - n_estimators: 100
        - max_depth: 5
        - learning_rate: 0.1
        
        **訓練時間**: {lgb_time}（最速）
        
        **考察**:
        - 小規模データでは性能が伸びにくい
        - ハイパーパラメータ調整で改善余地あり
        - 通常はXGBoost以上の性能が期待される
        - CV Score: {lgb_cv}
        """)


def show_feature_analysis():
    """特徴量分析ページ"""
    st.markdown('<div class="main-header">📊 特徴量分析</div>', unsafe_allow_html=True)
    
    # カテゴリマッピング
    category_map = {
        'volume': '時系列', 'delta_volume': '差分', 'delta_volume_rate': '差分',
        'negative_rate': '感情', 'sentiment_score': '感情', 'sentiment_polarity': '感情',
        'sentiment_avg_score': '感情',
        'stance_against_rate': '立場', 'stance_favor_rate': '立場', 'stance_neutral_rate': '立場',
        'stance_against_mean': '立場', 'stance_favor_mean': '立場', 'stance_neutral_mean': '立場',
        'delta_against_rate': '差分', 'delta_negative_rate': '差分',
        'avg_engagement': 'エンゲージ', 'total_engagement': 'エンゲージ', 'engagement_rate': 'エンゲージ',
        'flame_score': '複合', 'against_count': '立場'
    }
    
    # カラーマップを設定ファイルから取得
    if CONFIG and 'colors' in CONFIG:
        color_map = CONFIG['colors']['category_map']
    else:
        color_map = {
            '差分': '#FF6B35',
            '立場': '#004E89',
            'エンゲージ': '#F77F00',
            '感情': '#06A77D',
            '時系列': '#9D4EDD',
            '複合': '#E63946'
        }
    
    # === 総合特徴量重要度（全モデル平均） ===
    st.markdown('<div class="sub-header">🏆 総合特徴量重要度（全モデル平均）</div>', unsafe_allow_html=True)
    
    # 統合モデルのメタデータから特徴量重要度を読み込み
    unified_dir = Path("outputs/unified_models_comparison")
    models = {
        'CatBoost': 'CatBoost',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'Random Forest': 'Random_Forest',
        'Logistic Regression': 'Logistic_Regression',
        'SVM (RBF)': 'SVM_RBF'
    }
    
    from collections import defaultdict
    all_importances = defaultdict(list)
    model_importances = {}
    
    for display_name, dir_name in models.items():
        model_dir = unified_dir / dir_name
        metadata_file = model_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            fi = metadata.get('feature_importance')
            if fi and isinstance(fi, dict):
                # 生の重要度をそのまま使用
                model_importances[display_name] = fi
                
                for feature, importance in fi.items():
                    all_importances[feature].append(importance)
    
    # 平均を計算
    averaged_importance = {}
    for feature, values in all_importances.items():
        averaged_importance[feature] = np.mean(values)
    
    # TOP10でデータフレーム作成
    sorted_features = sorted(averaged_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    importance_data = {
        '特徴量': [f[0] for f in sorted_features],
        '重要度': [f[1] for f in sorted_features],
        'カテゴリ': [category_map.get(f[0], 'その他') for f in sorted_features]
    }
    
    df_importance = pd.DataFrame(importance_data)
    
    st.info(f"📊 **{len(model_importances)}モデル**の特徴量重要度を平均化")
    
    # 棒グラフ
    fig = px.bar(
        df_importance,
        x='重要度',
        y='特徴量',
        orientation='h',
        color='カテゴリ',
        title='総合特徴量重要度（6モデル平均）',
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # === 各モデル別の特徴量重要度（プルダウン） ===
    st.markdown('<div class="sub-header">🔍 モデル別特徴量重要度</div>', unsafe_allow_html=True)
    
    if model_importances:
        selected_model = st.selectbox(
            "モデルを選択",
            options=list(model_importances.keys()),
            index=0
        )
        
        # 選択されたモデルの特徴量重要度
        model_fi = model_importances[selected_model]
        sorted_model_fi = sorted(model_fi.items(), key=lambda x: x[1], reverse=True)[:10]
        
        model_data = {
            '特徴量': [f[0] for f in sorted_model_fi],
            '重要度': [f[1] for f in sorted_model_fi],
            'カテゴリ': [category_map.get(f[0], 'その他') for f in sorted_model_fi]
        }
        
        df_model = pd.DataFrame(model_data)
        
        fig_model = px.bar(
            df_model,
            x='重要度',
            y='特徴量',
            orientation='h',
            color='カテゴリ',
            title=f'{selected_model} - 特徴量重要度',
            color_discrete_map=color_map
        )
        
        fig_model.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14)
        )
        st.plotly_chart(fig_model, use_container_width=True)
    
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
    <h3>1️⃣ 「感情分析の平均スコア」が最重要特徴量</h3>
    
    **発見**:
    - `sentiment_avg_score`（感情分析の平均スコア）が重要度24.08でトップ
    - `stance_favor_mean`（賛成立場の平均値）が12.69で2位
    - `delta_negative_rate`（ネガティブ率の変化）が10.06で3位
    - `negative_rate`（ネガティブ率）が8.94で4位
    
    **意義**:
    - 炎上は「感情の総合的な傾向」と「立場の分布」の組み合わせで定義される
    - 単一指標ではなく、感情と立場の両面からの分析が必要
    
    **実装への示唆**:
    - BERTベース感情分析の精度が全体性能を左右
    - 立場分類（賛成/反対/中立）の重要性
    - 変化量の検出が炎上の早期発見に有効
    </div>
    """, unsafe_allow_html=True)
    
    # 知見2
    st.markdown("""
    <div class="info-box">
    <h3>2️⃣ 立場分類（BERT）の有効性</h3>
    
    **発見**:
    - `stance_favor_rate`、`stance_against_rate`が重要度上位
    - 感情分析だけでは不十分（立場の把握が必須）
    
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
    <h3>3️⃣ 複合特徴量の有効性</h3>
    
    **発見**:
    - `flame_score`が重要度5.97で5位
    - 複数の指標を組み合わせた特徴量が効果的
    
    **flame_scoreの定義**:
    - `negative_rate * 100 + stance_against_rate * delta_volume_rate`
    - ネガティブ感情、批判的立場、投稿急増を統合
    
    **解釈**:
    - 単一指標より複合指標が炎上の本質を捉える
    - ドメイン知識を反映した特徴量設計が重要
    - TOP10のうち複合・時系列・差分特徴量が多数を占める
    
    **実装上の利点**:
    - 解釈可能性を維持しつつ予測精度向上
    - 新しい複合特徴量の追加が容易
    </div>
    """, unsafe_allow_html=True)
    
    # 知見4
    st.markdown("""
    <div class="success-box">
    <h3>4️⃣ 複数モデルが高性能を達成</h3>
    
    **発見**:
    - 最高F1: **96.88%**（LightGBM）
    - 全6モデルが93%以上を達成
    - 訓練時間: 0.002〜0.138秒（実用的）
    
    **なぜLightGBMが最高性能？**:
    - Leaf-wise成長戦略で複雑なパターン学習
    - 差分特徴量（delta_negative_rate等）を効果的に活用
    - Recall 100%を達成（炎上の見逃しゼロ）
    
    **実務への示唆**:
    - ハイパーパラメータ調整の負担が少ない
    - 高速かつ高精度で実用的
    </div>
    """, unsafe_allow_html=True)
    
    # 知見5
    st.markdown("""
    <div class="info-box">
    <h3>5️⃣ 全モデルで93%以上 → 特徴量設計の成功</h3>
    
    **発見**:
    - 最低のSVM (RBF)でも93.10%
    - モデル間の差は小さい（3.77%）
    - 変動係数: 1.42%（極めて安定）
    
    **意味するもの**:
    - **特徴量エンジニアリングが適切**
    - 16特徴量が炎上の本質を捉えている
    - モデル選択より特徴量設計が重要
    
    **今後の方向性**:
    - 更なる性能向上はアンサンブル学習が有効
    - LightGBM + Random Forest + Logistic Regression のスタッキング
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🎯 実用化に向けた示唆</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ 実用化可能な点
        
        - **高精度**: F1 96.88%（統合モデル・LightGBM）
        - **安定性**: 全6モデルで93%以上達成
        - **高速**: 訓練0.002〜0.138秒、推論は瞬時
        - **解釈性**: 特徴量重要度で要因特定可能
        - **スケーラビリティ**: 並列処理対応
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ 課題と対策
        
        - **データ不足** → クラウドソーシング
        - **リアルタイム性** → Stream API連携
        - **誤検知** → 閾値の動的調整
        - **未知トピック** → Transfer Learning
        """)


def show_unified_models_comparison():
    """統合モデル比較ページ"""
    st.markdown('<div class="main-header">🔬 統合モデル比較</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 統合モデルとは？</h4>
    
    **目的**: 複数トピックのデータを統合して訓練した汎用的な炎上検知モデル
    
    **特徴**:
    - 5トピック306サンプルで訓練
    - 16特徴量を使用
    - 各アルゴリズムで独立に訓練
    - 閾値を最適化してF1スコアを最大化
    
    **利点**:
    - トピック横断的な炎上パターンを学習
    - 新規トピックへの適用が可能（ある程度）
    - 運用時のモデル管理が容易
    </div>
    """, unsafe_allow_html=True)
    
    # データ読み込み
    summary_file = Path("outputs/unified_models_comparison/summary.json")
    
    if not summary_file.exists():
        st.warning("⚠️ 統合モデルの結果ファイルが見つかりません。先に `python train_all_unified_models.py` を実行してください。")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    st.markdown('<div class="sub-header">🏆 6アルゴリズム性能ランキング</div>', unsafe_allow_html=True)
    
    # ランキングテーブル作成
    results = summary['results']
    ranking_data = []
    
    for model_name, data in results.items():
        ranking_data.append({
            'モデル': model_name,
            'F1 Score': f"{data['metrics']['f1']:.4f}",
            'Accuracy': f"{data['metrics']['accuracy']:.4f}",
            'Precision': f"{data['metrics']['precision']:.4f}",
            'Recall': f"{data['metrics']['recall']:.4f}",
            'ROC-AUC': f"{data['metrics']['roc_auc']:.4f}" if data['metrics']['roc_auc'] else "N/A",
            'CV F1': f"{data['cv_scores']['f1_mean']:.4f} ± {data['cv_scores']['f1_std']:.4f}",
            '訓練時間': f"{data['train_time']:.2f}秒",
            '閾値': f"{data['threshold']:.4f}"
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    
    # F1スコアでソート
    df_ranking['_f1_num'] = df_ranking['F1 Score'].astype(float)
    df_ranking = df_ranking.sort_values('_f1_num', ascending=False)
    df_ranking = df_ranking.drop(columns=['_f1_num'])
    
    # ランキング追加
    df_ranking.insert(0, 'ランク', ['🏆 1位', '🥈 2位', '🥉 3位', '4位', '5位', '6位'])
    
    st.dataframe(df_ranking, use_container_width=True, hide_index=True)
    
    # 動的に最低・最速・最遅モデルを計算
    min_model = min(results.items(), key=lambda x: x[1]['metrics']['f1'])
    train_times = sorted(results.items(), key=lambda x: x[1]['train_time'])
    fastest_models = [name for name, _ in train_times[:3]]
    slowest_model = train_times[-1]
    
    # 閾値範囲を計算
    thresholds = [data['threshold'] for data in results.values()]
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)
    
    # 重要な発見
    best_model_data = results[summary['best_model']]
    st.markdown(f"""
    <div class="success-box">
    <h4>✅ 重要な発見</h4>
    
    **1. {summary['best_model']}が最高性能**:
    - F1スコア: **{summary['best_f1']:.4f}** ({summary['best_f1']*100:.2f}%)
    - 訓練時間: わずか{best_model_data['train_time']:.2f}秒
    - CV F1: {best_model_data['cv_scores']['f1_mean']:.4f}（安定性も高い）
    
    **2. 全モデルが{min_model[1]['metrics']['f1']*100:.0f}%以上を達成**:
    - 最低モデル（{min_model[0]}）でも F1 = {min_model[1]['metrics']['f1']:.2%}
    - 特徴量設計の成功を示す
    
    **3. 訓練速度の違い**:
    - 最速: {', '.join(fastest_models[:2])} (0.00秒), {fastest_models[2]} ({results[fastest_models[2]]['train_time']:.2f}秒)
    - 最遅: {slowest_model[0]} ({slowest_model[1]['train_time']:.2f}秒)
    
    **4. 閾値最適化の効果**:
    - デフォルト0.5から大きく調整
    - モデルごとに最適値が異なる（{min_threshold:.2f}〜{max_threshold:.2f}）
    </div>
    """, unsafe_allow_html=True)
    
    # メトリクス比較グラフ
    st.markdown('<div class="sub-header">📊 メトリクス比較</div>', unsafe_allow_html=True)
    
    # 棒グラフ用データ準備
    metrics_data = []
    for model_name, data in results.items():
        metrics_data.append({
            'モデル': model_name,
            'F1': data['metrics']['f1'],
            'Accuracy': data['metrics']['accuracy'],
            'Precision': data['metrics']['precision'],
            'Recall': data['metrics']['recall']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.sort_values('F1', ascending=True)  # 横棒グラフ用に昇順
    
    # メトリクス選択
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_metric = st.radio(
            "表示メトリクス",
            ["F1", "Accuracy", "Precision", "Recall"]
        )
    
    with col2:
        fig = px.bar(
            df_metrics,
            x=selected_metric,
            y='モデル',
            orientation='h',
            title=f'{selected_metric} Score 比較',
            color=selected_metric,
            color_continuous_scale='Viridis',
            range_x=[0.9, 1.0]
        )
        
        fig.update_layout(
            height=400,
            font=dict(family="Yu Gothic, Meiryo, sans-serif", size=14),
            showlegend=False
        )
        
        # 値を表示
        for i, row in df_metrics.iterrows():
            fig.add_annotation(
                x=row[selected_metric],
                y=row['モデル'],
                text=f"{row[selected_metric]:.4f}",
                showarrow=False,
                xanchor='left',
                xshift=5
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # モデル詳細
    st.markdown('<div class="sub-header">🔍 各モデルの詳細</div>', unsafe_allow_html=True)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        icon = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}位"
        
        with st.expander(f"{icon} {model_name} - F1: {data['metrics']['f1']:.4f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**テスト性能**")
                st.write(f"- Accuracy: {data['metrics']['accuracy']:.4f}")
                st.write(f"- Precision: {data['metrics']['precision']:.4f}")
                st.write(f"- Recall: {data['metrics']['recall']:.4f}")
                st.write(f"- F1 Score: {data['metrics']['f1']:.4f}")
                if data['metrics']['roc_auc']:
                    st.write(f"- ROC-AUC: {data['metrics']['roc_auc']:.4f}")
            
            with col2:
                st.markdown("**クロスバリデーション**")
                st.write(f"- CV Accuracy: {data['cv_scores']['accuracy_mean']:.4f} ± {data['cv_scores']['accuracy_std']:.4f}")
                st.write(f"- CV F1: {data['cv_scores']['f1_mean']:.4f} ± {data['cv_scores']['f1_std']:.4f}")
                st.write(f"- CV ROC-AUC: {data['cv_scores']['roc_auc_mean']:.4f} ± {data['cv_scores']['roc_auc_std']:.4f}")
                st.write(f"- 訓練時間: {data['train_time']:.3f}秒")
                st.write(f"- 最適閾値: {data['threshold']:.4f}")
    
    # 実用化への示唆
    st.markdown('<div class="sub-header">💼 実用化への示唆</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 モデル選択のポイント</h4>
    
    **精度重視の場合**:
    - **LightGBM** または **Random Forest** を選択
    - F1 > 96.5%の高精度
    - CV安定性も高い
    
    **速度重視の場合**:
    - **SVM** または **Logistic Regression** を選択
    - 訓練時間 < 0.01秒
    - リアルタイム再学習に有利
    
    **バランス重視の場合**:
    - **XGBoost** または **CatBoost** を選択
    - 精度93-94%、訓練0.1-0.15秒
    - 業界標準で信頼性が高い
    
    **解釈性重視の場合**:
    - **Logistic Regression** を選択
    - 線形モデルで解釈容易
    - 特徴量の係数が明確
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
