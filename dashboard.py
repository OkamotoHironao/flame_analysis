#!/usr/bin/env python3
"""
🔥 炎上検知パイプライン GUIダッシュボード

Streamlitベースの操作画面
- データ管理
- パイプライン実行
- 結果可視化
- モデル学習

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import subprocess
import sys
import time
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yaml

# ページ設定
st.set_page_config(
    page_title="🔥 炎上検知AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# パス設定
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_DIR = DATA_DIR / "original"
STANDARDIZED_DIR = DATA_DIR / "standardized"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "modules" / "flame_detection" / "outputs"


def get_available_topics():
    """利用可能なトピック一覧を取得（data/original/配下を参照）"""
    topics = set()
    
    # original配下のフォルダを取得（メインのソース）
    if ORIGINAL_DIR.exists():
        for item in ORIGINAL_DIR.iterdir():
            if item.is_dir() and item.name not in ['.gitkeep', '__pycache__']:
                topics.add(item.name)
    
    # standardized（既に標準化済みのもの）
    if STANDARDIZED_DIR.exists():
        for f in STANDARDIZED_DIR.glob("*.csv"):
            if not f.name.endswith("_meta.json"):
                topics.add(f.stem)
    
    # outputs (ラベル付きデータがあるもの)
    if OUTPUTS_DIR.exists():
        for d in OUTPUTS_DIR.iterdir():
            if d.is_dir() and (d / f"{d.name}_labeled.csv").exists():
                topics.add(d.name)
    
    return sorted(list(topics))


def get_original_data_info(topic):
    """original配下のデータ情報を取得"""
    topic_dir = ORIGINAL_DIR / topic
    info = {
        'has_folder': topic_dir.exists() and topic_dir.is_dir(),
        'csv_files': [],
        'total_files': 0
    }
    
    if info['has_folder']:
        csv_files = list(topic_dir.glob("*.csv"))
        info['csv_files'] = [f.name for f in csv_files]
        info['total_files'] = len(csv_files)
    
    return info


def get_topic_status(topic):
    """トピックの処理状況を確認"""
    # original配下のデータ確認
    original_dir = ORIGINAL_DIR / topic
    has_original = original_dir.exists() and original_dir.is_dir() and len(list(original_dir.glob("*.csv"))) > 0
    
    status = {
        'original': has_original,
        'standardized': (STANDARDIZED_DIR / f"{topic}.csv").exists(),
        'bert': (PROCESSED_DIR / f"{topic}_bert.csv").exists(),
        'sentiment': (PROCESSED_DIR / f"{topic}_sentiment_1h.csv").exists(),
        'stance': (BASE_DIR / "modules" / "stance_detection" / "outputs" / topic / f"{topic}_stance.csv").exists(),
        'feature': (BASE_DIR / "modules" / "feature_engineering" / "outputs" / topic / f"{topic}_feature_table.csv").exists(),
        'labeled': (OUTPUTS_DIR / topic / f"{topic}_labeled.csv").exists(),
        'model': (OUTPUTS_DIR / topic / "model" / "model.pkl").exists(),
        'label_config': (BASE_DIR / "modules" / "flame_detection" / f"label_config_{topic}.yaml").exists(),
    }
    return status


def run_pipeline_step_with_log(topic, steps, force=False, status_container=None):
    """パイプラインステップを実行（ターミナルにログ出力）"""
    cmd = f"python auto_pipeline.py {topic} --steps {steps}"
    if force:
        cmd += " --force"
    
    # ターミナルにコマンドを表示
    print(f"\n{'='*60}")
    print(f"🔥 実行コマンド: {cmd}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    if status_container:
        status_container.info(f"⏳ 実行中: `{cmd}`")
    
    # ターミナルにリアルタイムでログを出力
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
        bufsize=1
    )
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line)
        # ターミナルに出力
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    full_output = "".join(output_lines)
    
    print(f"\n{'='*60}")
    print(f"✅ 完了 (exit code: {process.returncode})")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    return process.returncode == 0, full_output, ""


def run_unified_training_with_log(topics_str=None, status_container=None):
    """統合学習を実行（ターミナルにログ出力）"""
    cmd = "python auto_pipeline.py --unified-train"
    if topics_str:
        cmd += f" --unified-topics {topics_str}"
    
    # ターミナルにコマンドを表示
    print(f"\n{'='*60}")
    print(f"🔥 実行コマンド: {cmd}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    if status_container:
        status_container.info(f"⏳ 実行中: `{cmd}`")
    
    # ターミナルにリアルタイムでログを出力
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
        bufsize=1
    )
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line)
        # ターミナルに出力
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    full_output = "".join(output_lines)
    
    print(f"\n{'='*60}")
    print(f"✅ 完了 (exit code: {process.returncode})")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    return process.returncode == 0, full_output


def run_pipeline_step(topic, steps, force=False):
    """パイプラインステップを実行（後方互換性のため残す）"""
    cmd = f"python auto_pipeline.py {topic} --steps {steps}"
    if force:
        cmd += " --force"
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=str(BASE_DIR)
    )
    return result.returncode == 0, result.stdout, result.stderr


def load_feature_data(topic):
    """特徴量データを読み込み"""
    path = BASE_DIR / "modules" / "feature_engineering" / "outputs" / topic / f"{topic}_feature_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_labeled_data(topic):
    """ラベル付きデータを読み込み"""
    path = OUTPUTS_DIR / topic / f"{topic}_labeled.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_label_config(topic):
    """ラベル設定を読み込み"""
    path = BASE_DIR / "modules" / "flame_detection" / f"label_config_{topic}.yaml"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None


def save_label_config(topic, config):
    """ラベル設定を保存"""
    path = BASE_DIR / "modules" / "flame_detection" / f"label_config_{topic}.yaml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


# ========================================
# サイドバー
# ========================================
st.sidebar.title("🔥 炎上検知AI")
st.sidebar.markdown("---")

# ページ選択
page = st.sidebar.radio(
    "ページ選択",
    ["📊 ダッシュボード", "🔮 リアルタイム予測", "🔄 パイプライン実行", "🏷️ ラベリング", "🤖 モデル学習", "🔬 手法比較", "📈 結果分析"]
)

# トピック選択
topics = get_available_topics()
if topics:
    selected_topic = st.sidebar.selectbox("トピック選択", topics)
else:
    selected_topic = None
    st.sidebar.warning("トピックがありません")

st.sidebar.markdown("---")
st.sidebar.markdown("### クイックアクション")

# サイドバーの統合学習ボタン用のセッション状態
if 'unified_training_result' not in st.session_state:
    st.session_state.unified_training_result = None

if st.sidebar.button("🔄 統合学習を実行"):
    # 直接実行（ログはターミナルに出力）
    success, output = run_unified_training_with_log()
    st.session_state.unified_training_result = {"success": success, "output": output}
    st.rerun()


# ========================================
# メインコンテンツ
# ========================================

if page == "📊 ダッシュボード":
    st.title("📊 ダッシュボード")
    
    # 統合学習の結果表示（サイドバーから実行された場合）
    if st.session_state.unified_training_result is not None:
        result = st.session_state.unified_training_result
        if result["success"]:
            st.success("✅ 統合学習が完了しました！")
        else:
            st.error("❌ 統合学習でエラーが発生しました")
        st.session_state.unified_training_result = None
    
    # 概要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("トピック数", len(topics))
    
    # ラベル付き済みカウント
    labeled_count = sum(1 for t in topics if get_topic_status(t)['labeled'])
    with col2:
        st.metric("ラベル済み", labeled_count)
    
    # 統合モデルの状態
    unified_model_exists = (OUTPUTS_DIR / "unified_model_v2" / "model.pkl").exists()
    with col3:
        st.metric("統合モデル", "✅ あり" if unified_model_exists else "❌ なし")
    
    # 総サンプル数
    total_samples = 0
    for t in topics:
        df = load_labeled_data(t)
        if df is not None:
            total_samples += len(df)
    with col4:
        st.metric("総サンプル数", total_samples)
    
    st.markdown("---")
    
    # トピック一覧
    st.subheader("📂 トピック一覧")
    
    topic_data = []
    for t in topics:
        status = get_topic_status(t)
        df = load_labeled_data(t)
        original_info = get_original_data_info(t)
        
        topic_data.append({
            "トピック": t,
            "元データ": f"✅ ({original_info['total_files']})" if status['original'] else "❌",
            "標準化": "✅" if status['standardized'] else "❌",
            "感情分析": "✅" if status['sentiment'] else "❌",
            "立場検出": "✅" if status['stance'] else "❌",
            "特徴量": "✅" if status['feature'] else "❌",
            "ラベル設定": "✅" if status['label_config'] else "❌",
            "ラベル付き": "✅" if status['labeled'] else "❌",
            "サンプル数": len(df) if df is not None else 0,
        })
    
    st.dataframe(pd.DataFrame(topic_data), use_container_width=True, hide_index=True)
    
    # 統合モデルの性能
    if unified_model_exists:
        st.markdown("---")
        st.subheader("🤖 統合モデル性能")
        
        metadata_path = OUTPUTS_DIR / "unified_model_v2" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            metrics = metadata.get('metrics', {})
            
            with col1:
                st.metric("CV Accuracy", f"{metrics.get('cv_accuracy_mean', 0)*100:.1f}%")
            with col2:
                st.metric("CV F1 Score", f"{metrics.get('cv_f1_mean', 0)*100:.1f}%")
            with col3:
                st.metric("CV ROC-AUC", f"{metrics.get('cv_roc_auc_mean', 0)*100:.1f}%")
            with col4:
                st.metric("使用トピック数", len(metadata.get('topics', [])))
            
            # 特徴量重要度
            st.markdown("#### 特徴量重要度")
            importance = metadata.get('feature_importance', {})
            if importance:
                fig = px.bar(
                    x=list(importance.values()),
                    y=list(importance.keys()),
                    orientation='h',
                    labels={'x': '重要度', 'y': '特徴量'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)


elif page == "🔮 リアルタイム予測":
    st.title("🔮 リアルタイム炎上スコア予測")
    
    st.markdown("""
    ツイートデータをアップロードして、リアルタイムで炎上スコアを予測します。
    
    ### スコアの意味
    | スコア | 状態 | アイコン |
    |--------|------|----------|
    | 80-100 | 炎上中 | 🔴 |
    | 60-79 | 炎上の可能性高 | 🟠 |
    | 40-59 | 炎上の兆候あり | 🟡 |
    | 20-39 | やや注意 | 🟢 |
    | 0-19 | 平常 | ⚪ |
    """)
    
    st.markdown("---")
    
    # モデルの確認
    unified_model_path = OUTPUTS_DIR.parent.parent / "outputs" / "unified_model_v2"
    if not (unified_model_path / "model.pkl").exists():
        st.error("❌ 統合モデルが見つかりません。先に「モデル学習」ページで学習を実行してください。")
    else:
        # 入力方法の選択
        st.subheader("📂 データ入力")
        
        input_method = st.radio(
            "入力方法を選択",
            ["📤 CSVファイルをアップロード", "📁 既存の標準化データを使用"],
            horizontal=True
        )
        
        tweets_df = None
        topic_name = None
        
        if input_method == "📤 CSVファイルをアップロード":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "CSVファイルをアップロード",
                    type=['csv'],
                    help="content, timestampカラムを含むCSVファイル"
                )
            
            with col2:
                topic_name = st.text_input(
                    "トピック名（オプション）",
                    help="スタンス検出に使用します",
                    placeholder="例: 松本人志"
                )
            
            if uploaded_file is not None:
                try:
                    tweets_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ {len(tweets_df)}件のデータを読み込みました")
                    
                    # カラム確認
                    with st.expander("📋 データプレビュー"):
                        st.dataframe(tweets_df.head(10))
                except Exception as e:
                    st.error(f"❌ ファイルの読み込みに失敗しました: {e}")
        
        else:
            # 既存データの選択
            col1, col2 = st.columns(2)
            
            with col1:
                # 標準化データから選択
                standardized_files = list(STANDARDIZED_DIR.glob("*.csv")) if STANDARDIZED_DIR.exists() else []
                file_options = {f.stem: f for f in standardized_files}
                
                if file_options:
                    selected_file = st.selectbox(
                        "データを選択",
                        options=list(file_options.keys())
                    )
                    topic_name = selected_file
                    
                    if st.button("📂 データを読み込む"):
                        try:
                            tweets_df = pd.read_csv(file_options[selected_file])
                            st.session_state['realtime_tweets_df'] = tweets_df
                            st.session_state['realtime_topic'] = topic_name
                            st.success(f"✅ {len(tweets_df)}件のデータを読み込みました")
                        except Exception as e:
                            st.error(f"❌ ファイルの読み込みに失敗しました: {e}")
                else:
                    st.warning("標準化データがありません")
            
            with col2:
                # 期間フィルタ
                st.markdown("##### 📅 期間フィルタ（オプション）")
                use_date_filter = st.checkbox("期間を指定する")
                
                if use_date_filter:
                    filter_start = st.date_input("開始日", key="filter_start")
                    filter_end = st.date_input("終了日", key="filter_end")
            
            # セッションからデータを復元
            if 'realtime_tweets_df' in st.session_state:
                tweets_df = st.session_state['realtime_tweets_df']
                topic_name = st.session_state.get('realtime_topic', topic_name)
                
                # 期間フィルタ適用
                if use_date_filter and tweets_df is not None:
                    tweets_df_filtered = tweets_df.copy()
                    if 'timestamp' in tweets_df_filtered.columns:
                        tweets_df_filtered['timestamp'] = pd.to_datetime(tweets_df_filtered['timestamp'])
                        mask = (tweets_df_filtered['timestamp'].dt.date >= filter_start) & \
                               (tweets_df_filtered['timestamp'].dt.date <= filter_end)
                        tweets_df = tweets_df_filtered[mask]
                        st.info(f"📅 フィルタ適用: {len(tweets_df)}件")
        
        st.markdown("---")
        
        # 予測実行
        if tweets_df is not None and len(tweets_df) > 0:
            st.subheader("🎯 炎上スコア予測")
            
            col1, col2 = st.columns(2)
            
            with col1:
                show_timeseries = st.checkbox("📈 時系列推移を表示", value=True)
            
            with col2:
                if st.button("🚀 予測を実行", type="primary"):
                    st.session_state['run_prediction'] = True
            
            if st.session_state.get('run_prediction', False):
                try:
                    with st.spinner("🔄 予測中... （初回はモデル読み込みに時間がかかります）"):
                        # 予測器を初期化（キャッシュを使用）
                        @st.cache_resource
                        def get_predictor():
                            from modules.flame_detection.realtime_predictor import FlamePredictor
                            return FlamePredictor()
                        
                        predictor = get_predictor()
                        
                        # 予測実行
                        if show_timeseries:
                            results = predictor.predict_timeseries(tweets_df, topic_name)
                        else:
                            result = predictor.predict_current(tweets_df, topic_name)
                            results = [result]
                    
                    st.session_state['prediction_results'] = results
                    st.session_state['run_prediction'] = False
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ 予測に失敗しました: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state['run_prediction'] = False
            
            # 結果表示
            if 'prediction_results' in st.session_state:
                results = st.session_state['prediction_results']
                
                if len(results) == 1:
                    # 単一スコア表示
                    result = results[0]
                    
                    st.markdown("### 🎯 現在の炎上スコア")
                    
                    # 大きなスコア表示
                    score = result['score']
                    status = result['status']
                    
                    # カラーを決定
                    if score >= 80:
                        color = "#ff4444"
                    elif score >= 60:
                        color = "#ff8c00"
                    elif score >= 40:
                        color = "#ffd700"
                    elif score >= 20:
                        color = "#90ee90"
                    else:
                        color = "#e0e0e0"
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div style="font-size: 80px; font-weight: bold; color: {color};">
                                {score}
                            </div>
                            <div style="font-size: 24px; margin-top: 10px;">
                                {status}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # プログレスバー
                    st.progress(score / 100)
                    
                    # 警告メッセージ
                    if score >= 60:
                        st.error("⚠️ 警告: 炎上リスクが高い状態です！即座の対応を検討してください。")
                    elif score >= 40:
                        st.warning("📢 注意: 炎上の兆候が見られます。監視を続けてください。")
                    else:
                        st.success("✅ 現在は平常な状態です。")
                
                else:
                    # 時系列表示
                    st.markdown("### 📈 時系列炎上スコア推移")
                    
                    # 最新スコアをハイライト
                    latest = results[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("最新スコア", f"{latest['score']}/100")
                    with col2:
                        avg_score = sum(r['score'] for r in results) / len(results)
                        st.metric("平均スコア", f"{avg_score:.1f}/100")
                    with col3:
                        max_score = max(r['score'] for r in results)
                        st.metric("最大スコア", f"{max_score}/100")
                    with col4:
                        st.metric("データ点数", len(results))
                    
                    st.markdown(f"**最新の状態:** {latest['status']}")
                    
                    # グラフ作成
                    results_df = pd.DataFrame(results)
                    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
                    
                    fig = go.Figure()
                    
                    # 炎上ゾーンを背景に
                    fig.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=60, y1=80, fillcolor="orange", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=40, y1=60, fillcolor="yellow", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=20, y1=40, fillcolor="green", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=0, y1=20, fillcolor="gray", opacity=0.1, line_width=0)
                    
                    # スコアライン
                    fig.add_trace(go.Scatter(
                        x=results_df['timestamp'],
                        y=results_df['score'],
                        mode='lines+markers',
                        name='炎上スコア',
                        line=dict(color='crimson', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # 閾値ライン
                    fig.add_hline(y=60, line_dash="dash", line_color="red", 
                                 annotation_text="危険ライン (60)")
                    fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                                 annotation_text="警戒ライン (40)")
                    
                    fig.update_layout(
                        title="炎上スコアの時系列推移",
                        xaxis_title="日時",
                        yaxis_title="炎上スコア (0-100)",
                        yaxis=dict(range=[0, 105]),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 投稿量とネガティブ率も表示
                    st.markdown("#### 📊 関連指標")
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=results_df['timestamp'],
                        y=results_df['volume'],
                        name='投稿量',
                        line=dict(color='blue')
                    ))
                    
                    fig2.add_trace(go.Scatter(
                        x=results_df['timestamp'],
                        y=results_df['negative_rate'] * 100,
                        name='ネガティブ率 (%)',
                        yaxis='y2',
                        line=dict(color='orange')
                    ))
                    
                    fig2.update_layout(
                        title="投稿量とネガティブ率の推移",
                        xaxis_title="日時",
                        yaxis=dict(title="投稿量", side="left"),
                        yaxis2=dict(title="ネガティブ率 (%)", side="right", overlaying="y"),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 詳細テーブル
                    with st.expander("📋 詳細データ"):
                        display_df = results_df[['timestamp', 'score', 'status', 'volume', 'negative_rate']].copy()
                        display_df['negative_rate'] = display_df['negative_rate'].apply(lambda x: f"{x*100:.1f}%")
                        display_df.columns = ['日時', 'スコア', '状態', '投稿量', 'ネガティブ率']
                        st.dataframe(display_df, use_container_width=True)
                
                # 結果クリアボタン
                if st.button("🗑️ 結果をクリア"):
                    if 'prediction_results' in st.session_state:
                        del st.session_state['prediction_results']
                    if 'realtime_tweets_df' in st.session_state:
                        del st.session_state['realtime_tweets_df']
                    st.rerun()


elif page == "🔄 パイプライン実行":
    st.title("🔄 パイプライン実行")
    
    if selected_topic:
        status = get_topic_status(selected_topic)
        original_info = get_original_data_info(selected_topic)
        
        st.subheader(f"📂 {selected_topic} の処理状況")
        
        # ステータス表示（7つに増やす）
        cols = st.columns(7)
        steps_status = [
            ("元データ", status['original']),
            ("標準化", status['standardized']),
            ("感情分析", status['sentiment']),
            ("立場検出", status['stance']),
            ("特徴量", status['feature']),
            ("ラベル設定", status['label_config']),
            ("ラベル付き", status['labeled']),
        ]
        
        for col, (name, done) in zip(cols, steps_status):
            with col:
                if done:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
        
        # 元データの情報を表示
        if original_info['has_folder']:
            with st.expander(f"📁 元データ情報（{original_info['total_files']}ファイル）", expanded=False):
                st.write(f"**フォルダ:** `data/original/{selected_topic}/`")
                st.write(f"**CSVファイル数:** {original_info['total_files']}")
                if original_info['csv_files']:
                    st.write("**ファイル一覧:**")
                    for f in original_info['csv_files'][:10]:  # 最大10件表示
                        st.write(f"  - {f}")
                    if len(original_info['csv_files']) > 10:
                        st.write(f"  ... 他 {len(original_info['csv_files']) - 10} ファイル")
        
        st.markdown("---")
        
        # 実行オプション
        st.subheader("⚙️ 実行オプション")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 利用可能なステップを動的に決定
            available_steps = []
            
            # combineは元データがある場合のみ
            if status['original']:
                available_steps.append("combine")
            
            # sentiment以降は標準化済み、または元データがある場合
            if status['standardized'] or status['original']:
                available_steps.extend(["sentiment", "stance", "feature", "label"])
            
            # デフォルト選択を決定
            default_steps = []
            if status['original'] and not status['standardized']:
                default_steps = ["combine", "sentiment", "stance", "feature"]
            elif status['standardized'] and not status['feature']:
                default_steps = ["sentiment", "stance", "feature"]
            
            selected_steps = st.multiselect(
                "実行するステップ",
                available_steps,
                default=default_steps
            )
            
            # ステップの説明
            st.markdown("""
            <small>
            
            | ステップ | 説明 |
            |----------|------|
            | combine | 元データを標準化形式に変換 |
            | sentiment | BERT感情分析 |
            | stance | スタンス検出 |
            | feature | 特徴量生成 |
            | label | ラベル付け |
            
            </small>
            """, unsafe_allow_html=True)
        
        with col2:
            force = st.checkbox("強制上書き (--force)", value=False)
            
            if not status['original'] and not status['standardized']:
                st.warning(f"⚠️ `data/original/{selected_topic}/` にCSVファイルを配置してください")
        
        # ステータス表示エリア
        status_area = st.empty()
        
        if st.button("🚀 パイプライン実行", type="primary", disabled=len(selected_steps)==0):
            steps_str = ",".join(selected_steps)
            
            st.info(f"⏳ 実行中: {steps_str}（ログはターミナルに出力されます）")
            
            # ステータス表示
            status_container = st.empty()
            success, stdout, _ = run_pipeline_step_with_log(
                selected_topic, steps_str, force, status_container
            )
            
            if success:
                st.success("✅ 完了しました！")
            else:
                st.error("❌ エラーが発生しました")
            
            # 最終ログを展開可能な形式で表示
            with st.expander("📋 実行ログ", expanded=False):
                st.code(stdout, language="bash")
            
            time.sleep(1)
            st.rerun()
    else:
        st.warning("サイドバーでトピックを選択してください")


elif page == "🏷️ ラベリング":
    st.title("🏷️ ラベリング設定")
    
    if selected_topic:
        # 特徴量データを表示
        df = load_feature_data(selected_topic)
        
        if df is not None:
            st.subheader("📈 特徴量データの可視化")
            
            # タイムスタンプをパース
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # グラフ作成
            fig = go.Figure()
            
            # 投稿量
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['volume'],
                name='投稿量',
                yaxis='y'
            ))
            
            # ネガティブ率
            if 'negative_rate' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['negative_rate'] * 100,
                    name='ネガティブ率 (%)',
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title=f"{selected_topic} - 時系列推移",
                xaxis_title="日時",
                yaxis=dict(title="投稿量", side="left"),
                yaxis2=dict(title="ネガティブ率 (%)", side="right", overlaying="y"),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("期間", f"{df['timestamp'].min().strftime('%Y-%m-%d')} 〜 {df['timestamp'].max().strftime('%Y-%m-%d')}")
            with col2:
                st.metric("時間数", len(df))
            with col3:
                st.metric("平均投稿量", f"{df['volume'].mean():.1f}")
            with col4:
                if 'negative_rate' in df.columns:
                    st.metric("平均ネガティブ率", f"{df['negative_rate'].mean()*100:.1f}%")
            
            st.markdown("---")
            
            # ラベル設定
            st.subheader("⚙️ 炎上期間の設定")
            
            # 既存設定を読み込み
            existing_config = load_label_config(selected_topic)
            
            if existing_config:
                st.info("既存の設定が見つかりました")
                with st.expander("現在の設定", expanded=False):
                    st.code(yaml.dump(existing_config, allow_unicode=True))
            
            # ラベル設定
            st.markdown("#### ⚙️ ラベル設定")
            
            # 炎上トピックかどうか
            is_flame_topic = st.radio(
                "このトピックは炎上事例ですか？",
                ["🔥 炎上事例", "✅ 非炎上事例"],
                index=0,
                horizontal=True
            )
            
            if is_flame_topic == "🔥 炎上事例":
                st.markdown("##### 炎上期間の設定")
                
                col1, col2 = st.columns(2)
                
                min_date = df['timestamp'].min().date()
                max_date = df['timestamp'].max().date()
                
                with col1:
                    st.markdown("**開始日時** (必須)")
                    start_date = st.date_input("開始日", value=min_date, min_value=min_date, max_value=max_date, key="start_date")
                    start_hour = st.selectbox("開始時刻", list(range(24)), index=0, key="start_hour")
                
                with col2:
                    st.markdown("**終了日時** (任意)")
                    has_end_date = st.checkbox("終了日を設定する", value=True)
                    if has_end_date:
                        end_date = st.date_input("終了日", value=max_date, min_value=min_date, max_value=max_date, key="end_date")
                        end_hour = st.selectbox("終了時刻", list(range(24)), index=23, key="end_hour")
                    else:
                        st.info("終了日未設定 → データ終了まで炎上とみなす")
                        end_date = max_date
                        end_hour = 23
                
                description = st.text_input("説明（任意）", placeholder="例: 文春報道後の炎上期間")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📝 設定を保存", type="primary"):
                        start_str = f"{start_date} {start_hour:02d}:00:00"
                        end_str = f"{end_date} {end_hour:02d}:00:00"
                        
                        new_period = {
                            'start': start_str,
                            'end': end_str,
                        }
                        if description:
                            new_period['description'] = description
                        
                        config = {
                            'topic': selected_topic,
                            'controversy_periods': [new_period]
                        }
                        
                        save_label_config(selected_topic, config)
                        st.success("✅ 設定を保存しました！")
                        st.rerun()
                
                with col2:
                    if st.button("🏷️ ラベリング実行"):
                        if not get_topic_status(selected_topic)['label_config']:
                            st.error("先に設定を保存してください")
                        else:
                            with st.spinner("ラベリング実行中..."):
                                success, stdout, stderr = run_pipeline_step(selected_topic, "label", force=True)
                            
                            if success:
                                st.success("✅ ラベリング完了！")
                                st.rerun()
                            else:
                                st.error("❌ エラーが発生しました")
                                st.code(stderr if stderr else stdout)
                
                with col3:
                    if st.button("🗑️ 設定を削除"):
                        config = {
                            'topic': selected_topic,
                            'controversy_periods': []
                        }
                        save_label_config(selected_topic, config)
                        st.success("✅ 設定を削除しました")
                        st.rerun()
            
            else:
                # 非炎上事例
                st.info("📋 非炎上事例として設定します（全期間 is_controversy=0）")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 非炎上として保存", type="primary"):
                        config = {
                            'topic': selected_topic,
                            'controversy_periods': []
                        }
                        save_label_config(selected_topic, config)
                        st.success("✅ 非炎上として設定を保存しました！")
                        st.rerun()
                
                with col2:
                    if st.button("🏷️ ラベリング実行"):
                        # 設定がなくても非炎上として実行
                        config = {
                            'topic': selected_topic,
                            'controversy_periods': []
                        }
                        save_label_config(selected_topic, config)
                        
                        with st.spinner("ラベリング実行中..."):
                            success, stdout, stderr = run_pipeline_step(selected_topic, "label", force=True)
                        
                        if success:
                            st.success("✅ ラベリング完了！（全て非炎上）")
                            st.rerun()
                        else:
                            st.error("❌ エラーが発生しました")
                            st.code(stderr if stderr else stdout)
            
            # ラベル付きデータの確認
            labeled_df = load_labeled_data(selected_topic)
            if labeled_df is not None:
                st.markdown("---")
                st.subheader("📊 ラベル付け結果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    controversy_count = (labeled_df['is_controversy'] == 1).sum()
                    st.metric("炎上ラベル (1)", controversy_count)
                with col2:
                    non_controversy_count = (labeled_df['is_controversy'] == 0).sum()
                    st.metric("非炎上ラベル (0)", non_controversy_count)
                with col3:
                    total = len(labeled_df)
                    flame_rate = controversy_count / total * 100 if total > 0 else 0
                    st.metric("炎上率", f"{flame_rate:.1f}%")
                
                # ラベル分布の可視化
                fig = px.pie(
                    values=[controversy_count, non_controversy_count],
                    names=['炎上', '非炎上'],
                    title='ラベル分布'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("特徴量データがありません。先にパイプラインを実行してください。")
    else:
        st.warning("サイドバーでトピックを選択してください")


elif page == "🤖 モデル学習":
    st.title("🤖 モデル学習")
    
    # 利用可能なトピック（ラベル付き済み）
    labeled_topics = [t for t in topics if get_topic_status(t)['labeled']]
    
    st.subheader("📂 ラベル付き済みトピック")
    
    if labeled_topics:
        # トピック情報
        topic_info = []
        for t in labeled_topics:
            df = load_labeled_data(t)
            if df is not None:
                topic_info.append({
                    'トピック': t,
                    'サンプル数': len(df),
                    '炎上(1)': (df['is_controversy'] == 1).sum(),
                    '非炎上(0)': (df['is_controversy'] == 0).sum(),
                    '炎上率': f"{df['is_controversy'].mean()*100:.1f}%"
                })
        
        st.dataframe(pd.DataFrame(topic_info), use_container_width=True)
        
        st.markdown("---")
        
        # 統合学習
        st.subheader("🚀 統合学習")
        
        selected_for_training = st.multiselect(
            "学習に使用するトピック",
            labeled_topics,
            default=labeled_topics
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 統合学習を実行", type="primary", disabled=len(selected_for_training) < 2):
                topics_str = ",".join(selected_for_training)
                
                st.info(f"⏳ 統合学習を実行中: {topics_str}（ログはターミナルに出力されます）")
                
                # ステータス表示
                status_container = st.empty()
                success, output = run_unified_training_with_log(topics_str, status_container)
                
                if success:
                    st.success("✅ 統合学習が完了しました！")
                else:
                    st.error("❌ エラーが発生しました")
                
                with st.expander("📋 実行ログ", expanded=False):
                    st.code(output, language="bash")
                
                time.sleep(1)
                st.rerun()
        
        with col2:
            if len(selected_for_training) < 2:
                st.warning("2つ以上のトピックを選択してください")
        
        # 現在のモデル情報
        if (OUTPUTS_DIR / "unified_model_v2" / "metadata.json").exists():
            st.markdown("---")
            st.subheader("📊 現在の統合モデル")
            
            with open(OUTPUTS_DIR / "unified_model_v2" / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("作成日時", metadata.get('created_at', '-')[:19])
            with col2:
                st.metric("使用トピック", ", ".join(metadata.get('topics', [])))
            with col3:
                metrics = metadata.get('metrics', {})
                st.metric("CV F1 Score", f"{metrics.get('cv_f1_mean', 0)*100:.1f}%")
    
    else:
        st.warning("ラベル付き済みのトピックがありません。先にラベリングを行ってください。")


elif page == "🔬 手法比較":
    st.title("🔬 感情分析手法の比較実験")
    
    st.markdown("""
    異なる感情分析手法を比較して、最適な手法を見つけます。
    
    ### 比較対象
    | 手法 | 説明 |
    |------|------|
    | **BERT のみ** | 事前学習済み深層学習モデルによる感情分析 |
    | **辞書ベース のみ** | PN辞書を使用した従来型の感情分析 |
    | **BERT + 辞書** | 両方の特徴量を組み合わせて使用 |
    """)
    
    st.markdown("---")
    
    # ラベル付き済みトピック
    labeled_topics = [t for t in topics if get_topic_status(t)['labeled']]
    
    if len(labeled_topics) >= 2:
        st.subheader("📂 実験設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_for_comparison = st.multiselect(
                "比較に使用するトピック",
                labeled_topics,
                default=labeled_topics
            )
        
        with col2:
            st.info(f"選択: {len(selected_for_comparison)}トピック")
        
        # 辞書分析の事前実行オプション
        st.markdown("---")
        st.subheader("📖 辞書ベース感情分析")
        
        st.markdown("""
        辞書ベースの比較を行うには、事前に辞書感情分析を実行する必要があります。
        """)
        
        # 辞書分析の状態確認
        dict_status = {}
        for t in selected_for_comparison:
            dict_path = PROCESSED_DIR / f"{t}_dict_sentiment_1h.csv"
            dict_status[t] = dict_path.exists()
        
        cols = st.columns(len(selected_for_comparison) if selected_for_comparison else 1)
        for i, t in enumerate(selected_for_comparison):
            with cols[i]:
                if dict_status[t]:
                    st.success(f"✅ {t}")
                else:
                    st.warning(f"❌ {t}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📖 辞書分析を実行", type="secondary"):
                for topic in selected_for_comparison:
                    if not dict_status[topic]:
                        st.info(f"⏳ {topic} の辞書分析を実行中...")
                        
                        # 標準化CSVから辞書分析（絶対パスを使用）
                        input_path = STANDARDIZED_DIR / f"{topic}.csv"
                        output_path = PROCESSED_DIR / f"{topic}_dict_sentiment_1h.csv"
                        
                        # BASE_DIRから実行し、絶対パスを渡す
                        cmd = f"cd {BASE_DIR} && python modules/sentiment_analysis/aggregate_dict_sentiment.py {input_path} -o {output_path}"
                        
                        print(f"\n{'='*60}")
                        print(f"🔥 実行: {cmd}")
                        print(f"{'='*60}")
                        sys.stdout.flush()
                        
                        import subprocess
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(BASE_DIR))
                        
                        if result.returncode == 0:
                            st.success(f"✅ {topic} 完了")
                        else:
                            st.error(f"❌ {topic} エラー")
                            st.code(result.stderr if result.stderr else result.stdout)
                
                time.sleep(1)
                st.rerun()
        
        # 比較実験実行
        st.markdown("---")
        st.subheader("🔬 比較実験")
        
        with col2:
            run_comparison = st.button("🚀 比較実験を実行", type="primary", 
                                       disabled=len(selected_for_comparison) < 2)
        
        if run_comparison:
            topics_str = ",".join(selected_for_comparison)
            
            st.info(f"⏳ 比較実験を実行中...（ログはターミナルに出力されます）")
            
            cmd = f"cd {BASE_DIR}/modules/flame_detection && python compare_sentiment_methods.py --topics {topics_str}"
            
            print(f"\n{'='*60}")
            print(f"🔥 実行: {cmd}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            import subprocess
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                print(line, end='')
                sys.stdout.flush()
            
            process.wait()
            
            if process.returncode == 0:
                st.success("✅ 比較実験が完了しました！")
            else:
                st.error("❌ エラーが発生しました")
            
            with st.expander("📋 実行ログ", expanded=True):
                st.code("".join(output_lines), language="bash")
            
            st.rerun()
        
        # 過去の結果を表示
        st.markdown("---")
        st.subheader("📊 比較結果")
        
        results_dir = OUTPUTS_DIR / "comparison_results"
        if results_dir.exists():
            result_files = sorted(results_dir.glob("sentiment_comparison_*.json"), reverse=True)
            
            if result_files:
                selected_result = st.selectbox(
                    "結果ファイルを選択",
                    result_files,
                    format_func=lambda x: x.stem
                )
                
                if selected_result:
                    with open(selected_result, 'r', encoding='utf-8') as f:
                        comparison_data = json.load(f)
                    
                    st.markdown(f"**実行日時:** {comparison_data['timestamp']}")
                    st.markdown(f"**使用トピック:** {', '.join(comparison_data['topics'])}")
                    st.markdown(f"**サンプル数:** {comparison_data['n_samples']}")
                    
                    # 結果テーブル
                    results_df = pd.DataFrame(comparison_data['results'])
                    
                    display_df = results_df[['method', 'cv_accuracy_mean', 'cv_f1_mean', 'cv_roc_auc_mean', 'n_features']].copy()
                    display_df.columns = ['手法', 'Accuracy', 'F1 Score', 'ROC-AUC', '特徴量数']
                    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
                    display_df['F1 Score'] = display_df['F1 Score'].apply(lambda x: f"{x*100:.1f}%")
                    display_df['ROC-AUC'] = display_df['ROC-AUC'].apply(lambda x: f"{x*100:.1f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # グラフ
                    fig = go.Figure()
                    
                    methods = results_df['method'].tolist()
                    
                    fig.add_trace(go.Bar(
                        name='Accuracy',
                        x=methods,
                        y=results_df['cv_accuracy_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in results_df['cv_accuracy_mean']],
                        textposition='auto',
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='F1 Score',
                        x=methods,
                        y=results_df['cv_f1_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in results_df['cv_f1_mean']],
                        textposition='auto',
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='ROC-AUC',
                        x=methods,
                        y=results_df['cv_roc_auc_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in results_df['cv_roc_auc_mean']],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title='感情分析手法の比較',
                        xaxis_title='手法',
                        yaxis_title='スコア (%)',
                        barmode='group',
                        height=400,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 最良手法
                    st.success(f"🏆 最良手法: **{comparison_data['best_method']}**")
            else:
                st.info("まだ比較実験の結果がありません。上のボタンから実行してください。")
        else:
            st.info("まだ比較実験の結果がありません。上のボタンから実行してください。")
        
        # ========================================
        # 立場検出の有無の比較
        # ========================================
        st.markdown("---")
        st.markdown("---")
        st.subheader("🎯 立場検出の有無の比較")
        
        st.markdown("""
        立場検出（Stance Detection）の有無がモデル性能に与える影響を比較します。
        
        | 手法 | 説明 |
        |------|------|
        | **Stance あり** | 感情分析 + 立場検出（賛成/反対/中立）を使用 |
        | **Stance なし** | 感情分析のみを使用（立場検出なし） |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            run_stance_comparison = st.button("🚀 立場検出比較を実行", type="primary",
                                              disabled=len(selected_for_comparison) < 2)
        
        if run_stance_comparison:
            topics_str = ",".join(selected_for_comparison)
            
            st.info(f"⏳ 立場検出比較を実行中...（ログはターミナルに出力されます）")
            
            cmd = f"cd {BASE_DIR} && python modules/flame_detection/compare_sentiment_methods.py --topics {topics_str} --type stance"
            
            print(f"\n{'='*60}")
            print(f"🔥 実行: {cmd}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            import subprocess
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                print(line, end='')
                sys.stdout.flush()
            
            process.wait()
            
            if process.returncode == 0:
                st.success("✅ 立場検出比較が完了しました！")
            else:
                st.error("❌ エラーが発生しました")
            
            with st.expander("📋 実行ログ", expanded=True):
                st.code("".join(output_lines), language="bash")
            
            st.rerun()
        
        # 立場検出比較の結果を表示
        st.markdown("#### 📊 立場検出比較の結果")
        
        if results_dir.exists():
            stance_result_files = sorted(results_dir.glob("stance_comparison_*.json"), reverse=True)
            
            if stance_result_files:
                selected_stance_result = st.selectbox(
                    "立場検出比較の結果ファイルを選択",
                    stance_result_files,
                    format_func=lambda x: x.stem,
                    key="stance_result_select"
                )
                
                if selected_stance_result:
                    with open(selected_stance_result, 'r', encoding='utf-8') as f:
                        stance_data = json.load(f)
                    
                    st.markdown(f"**実行日時:** {stance_data['timestamp']}")
                    st.markdown(f"**使用トピック:** {', '.join(stance_data['topics'])}")
                    
                    # 結果テーブル
                    stance_results_df = pd.DataFrame(stance_data['results'])
                    
                    stance_display_df = stance_results_df[['method', 'cv_accuracy_mean', 'cv_f1_mean', 'cv_roc_auc_mean', 'n_features']].copy()
                    stance_display_df.columns = ['手法', 'Accuracy', 'F1 Score', 'ROC-AUC', '特徴量数']
                    stance_display_df['Accuracy'] = stance_display_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
                    stance_display_df['F1 Score'] = stance_display_df['F1 Score'].apply(lambda x: f"{x*100:.1f}%")
                    stance_display_df['ROC-AUC'] = stance_display_df['ROC-AUC'].apply(lambda x: f"{x*100:.1f}%")
                    
                    st.dataframe(stance_display_df, use_container_width=True)
                    
                    # グラフ
                    fig2 = go.Figure()
                    
                    stance_methods = stance_results_df['method'].tolist()
                    
                    fig2.add_trace(go.Bar(
                        name='Accuracy',
                        x=stance_methods,
                        y=stance_results_df['cv_accuracy_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in stance_results_df['cv_accuracy_mean']],
                        textposition='auto',
                    ))
                    
                    fig2.add_trace(go.Bar(
                        name='F1 Score',
                        x=stance_methods,
                        y=stance_results_df['cv_f1_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in stance_results_df['cv_f1_mean']],
                        textposition='auto',
                    ))
                    
                    fig2.add_trace(go.Bar(
                        name='ROC-AUC',
                        x=stance_methods,
                        y=stance_results_df['cv_roc_auc_mean'] * 100,
                        text=[f"{v*100:.1f}%" for v in stance_results_df['cv_roc_auc_mean']],
                        textposition='auto',
                    ))
                    
                    fig2.update_layout(
                        title='立場検出の有無の比較',
                        xaxis_title='手法',
                        yaxis_title='スコア (%)',
                        barmode='group',
                        height=400,
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 効果の表示
                    with_stance = next((r for r in stance_data['results'] if r['method_key'] == 'with_stance'), None)
                    without_stance = next((r for r in stance_data['results'] if r['method_key'] == 'without_stance'), None)
                    
                    if with_stance and without_stance:
                        diff_f1 = (with_stance['cv_f1_mean'] - without_stance['cv_f1_mean']) * 100
                        diff_auc = (with_stance['cv_roc_auc_mean'] - without_stance['cv_roc_auc_mean']) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("F1 Score 改善", f"{'+' if diff_f1 >= 0 else ''}{diff_f1:.1f}%",
                                     delta=f"{diff_f1:.1f}%")
                        with col2:
                            st.metric("ROC-AUC 改善", f"{'+' if diff_auc >= 0 else ''}{diff_auc:.1f}%",
                                     delta=f"{diff_auc:.1f}%")
                    
                    st.success(f"🏆 最良手法: **{stance_data['best_method']}**")
            else:
                st.info("まだ立場検出比較の結果がありません。上のボタンから実行してください。")
        else:
            st.info("まだ立場検出比較の結果がありません。上のボタンから実行してください。")
    
    else:
        st.warning("比較実験には2つ以上のラベル付き済みトピックが必要です。先にラベリングを行ってください。")


elif page == "📈 結果分析":
    st.title("📈 結果分析")
    
    if selected_topic:
        labeled_df = load_labeled_data(selected_topic)
        
        if labeled_df is not None:
            st.subheader(f"📊 {selected_topic} の分析")
            
            # タイムスタンプをパース
            labeled_df['timestamp'] = pd.to_datetime(labeled_df['timestamp'])
            
            # 時系列グラフ
            fig = go.Figure()
            
            # 炎上期間をハイライト
            controversy_periods = labeled_df[labeled_df['is_controversy'] == 1]
            if len(controversy_periods) > 0:
                for _, row in controversy_periods.iterrows():
                    fig.add_vrect(
                        x0=row['timestamp'],
                        x1=row['timestamp'] + pd.Timedelta(hours=1),
                        fillcolor="red",
                        opacity=0.2,
                        line_width=0,
                    )
            
            # 投稿量
            fig.add_trace(go.Scatter(
                x=labeled_df['timestamp'],
                y=labeled_df['volume'],
                name='投稿量',
                line=dict(color='blue')
            ))
            
            # ネガティブ率
            if 'negative_rate' in labeled_df.columns:
                fig.add_trace(go.Scatter(
                    x=labeled_df['timestamp'],
                    y=labeled_df['negative_rate'] * 100,
                    name='ネガティブ率 (%)',
                    yaxis='y2',
                    line=dict(color='orange')
                ))
            
            fig.update_layout(
                title=f"{selected_topic} - 時系列推移（赤：炎上期間）",
                xaxis_title="日時",
                yaxis=dict(title="投稿量", side="left"),
                yaxis2=dict(title="ネガティブ率 (%)", side="right", overlaying="y"),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計
            st.markdown("---")
            st.subheader("📊 統計情報")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 炎上期間 (is_controversy=1)")
                controversy = labeled_df[labeled_df['is_controversy'] == 1]
                if len(controversy) > 0:
                    st.write(f"- サンプル数: {len(controversy)}")
                    st.write(f"- 平均投稿量: {controversy['volume'].mean():.1f}")
                    if 'negative_rate' in controversy.columns:
                        st.write(f"- 平均ネガティブ率: {controversy['negative_rate'].mean()*100:.1f}%")
                else:
                    st.write("データなし")
            
            with col2:
                st.markdown("#### 非炎上期間 (is_controversy=0)")
                non_controversy = labeled_df[labeled_df['is_controversy'] == 0]
                if len(non_controversy) > 0:
                    st.write(f"- サンプル数: {len(non_controversy)}")
                    st.write(f"- 平均投稿量: {non_controversy['volume'].mean():.1f}")
                    if 'negative_rate' in non_controversy.columns:
                        st.write(f"- 平均ネガティブ率: {non_controversy['negative_rate'].mean()*100:.1f}%")
                else:
                    st.write("データなし")
            
            # 特徴量の箱ひげ図
            st.markdown("---")
            st.subheader("📊 特徴量比較")
            
            feature_cols = ['volume', 'negative_rate', 'stance_against_rate']
            available_features = [f for f in feature_cols if f in labeled_df.columns]
            
            if available_features:
                selected_feature = st.selectbox("特徴量を選択", available_features)
                
                fig = px.box(
                    labeled_df,
                    x='is_controversy',
                    y=selected_feature,
                    labels={'is_controversy': 'ラベル', selected_feature: selected_feature},
                    title=f'{selected_feature} の分布比較'
                )
                fig.update_xaxes(ticktext=['非炎上 (0)', '炎上 (1)'], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("ラベル付きデータがありません。先にラベリングを行ってください。")
    else:
        st.warning("サイドバーでトピックを選択してください")


# フッター
st.sidebar.markdown("---")
st.sidebar.markdown("v2.0 | 🔥 炎上検知AI")
