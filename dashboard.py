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
STANDARDIZED_DIR = DATA_DIR / "standardized"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "modules" / "flame_detection" / "outputs"


def get_available_topics():
    """利用可能なトピック一覧を取得"""
    topics = set()
    
    # standardized
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


def get_topic_status(topic):
    """トピックの処理状況を確認"""
    status = {
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
    ["📊 ダッシュボード", "🔄 パイプライン実行", "🏷️ ラベリング", "🤖 モデル学習", "📈 結果分析"]
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
        
        topic_data.append({
            "トピック": t,
            "標準化": "✅" if status['standardized'] else "❌",
            "感情分析": "✅" if status['sentiment'] else "❌",
            "立場検出": "✅" if status['stance'] else "❌",
            "特徴量": "✅" if status['feature'] else "❌",
            "ラベル設定": "✅" if status['label_config'] else "❌",
            "ラベル付き": "✅" if status['labeled'] else "❌",
            "サンプル数": len(df) if df is not None else 0,
            "炎上率": f"{(df['is_controversy'].mean()*100):.1f}%" if df is not None and 'is_controversy' in df.columns else "-"
        })
    
    st.dataframe(pd.DataFrame(topic_data), use_container_width=True)
    
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


elif page == "🔄 パイプライン実行":
    st.title("🔄 パイプライン実行")
    
    if selected_topic:
        status = get_topic_status(selected_topic)
        
        st.subheader(f"📂 {selected_topic} の処理状況")
        
        # ステータス表示
        cols = st.columns(6)
        steps_status = [
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
        
        st.markdown("---")
        
        # 実行オプション
        st.subheader("⚙️ 実行オプション")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_steps = st.multiselect(
                "実行するステップ",
                ["combine", "sentiment", "stance", "feature", "label"],
                default=["sentiment", "stance", "feature"] if not status['feature'] else []
            )
        
        with col2:
            force = st.checkbox("強制上書き (--force)", value=False)
        
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
            
            # 新規設定
            st.markdown("#### 炎上期間を追加")
            
            col1, col2 = st.columns(2)
            
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            
            with col1:
                start_date = st.date_input("開始日", value=min_date, min_value=min_date, max_value=max_date)
                start_hour = st.selectbox("開始時刻", list(range(24)), index=0)
            
            with col2:
                end_date = st.date_input("終了日", value=max_date, min_value=min_date, max_value=max_date)
                end_hour = st.selectbox("終了時刻", list(range(24)), index=23)
            
            description = st.text_input("説明（任意）", placeholder="例: 文春報道後の炎上期間")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 設定を保存"):
                    # 設定を作成（常に上書き）
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
            
            # ラベル付きデータの確認
            labeled_df = load_labeled_data(selected_topic)
            if labeled_df is not None:
                st.markdown("---")
                st.subheader("📊 ラベル付け結果")
                
                col1, col2 = st.columns(2)
                with col1:
                    controversy_count = (labeled_df['is_controversy'] == 1).sum()
                    st.metric("炎上ラベル (1)", controversy_count)
                with col2:
                    non_controversy_count = (labeled_df['is_controversy'] == 0).sum()
                    st.metric("非炎上ラベル (0)", non_controversy_count)
                
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
