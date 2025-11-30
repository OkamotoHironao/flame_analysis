#!/bin/bash
# 全トピックの学習パイプライン実行スクリプト

set -e  # エラーが起きたら停止

# 色付け
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# プロジェクトルート
PROJECT_ROOT="/home/h-okamoto/デスクトップ/sotuken/flame_analysis"
cd "$PROJECT_ROOT"

# 仮想環境アクティベート
source .venv/bin/activate

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🔥 全トピック学習パイプライン${NC}"
echo -e "${BLUE}============================================================${NC}"

# トピック一覧
TOPICS=("松本人志" "寿司ペロ" "三苫" "aespa")

for TOPIC in "${TOPICS[@]}"; do
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📍 トピック: ${TOPIC}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # データディレクトリ確認
    DATA_DIR="data/original/${TOPIC}"
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}⚠️  スキップ: ${DATA_DIR} が見つかりません${NC}"
        continue
    fi
    
    # CSVファイル数確認
    CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" -type f | wc -l)
    if [ "$CSV_COUNT" -eq 0 ]; then
        echo -e "${RED}⚠️  スキップ: CSVファイルが見つかりません${NC}"
        continue
    fi
    
    echo -e "${BLUE}📂 データファイル数: ${CSV_COUNT}${NC}"
    
    # 出力ディレクトリ作成
    mkdir -p "modules/sentiment_analysis/dictionary_based/outputs/${TOPIC}"
    mkdir -p "modules/stance_detection/outputs/${TOPIC}"
    mkdir -p "modules/feature_engineering/outputs/${TOPIC}"
    mkdir -p "modules/flame_detection/outputs/${TOPIC}/model"
    
    # ----------------------------------------
    # Step 1: 感情分析
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 1/5] 感情分析（辞書ベース）${NC}"
    
    SENTIMENT_OUTPUT="modules/sentiment_analysis/dictionary_based/outputs/${TOPIC}/${TOPIC}_sentiment_1h.csv"
    
    if [ -f "$SENTIMENT_OUTPUT" ]; then
        echo -e "${GREEN}✓ スキップ: ${SENTIMENT_OUTPUT} は既に存在します${NC}"
    else
        python modules/sentiment_analysis/dictionary_based/analyze.py "$DATA_DIR"
        if [ -f "$SENTIMENT_OUTPUT" ]; then
            echo -e "${GREEN}✓ 感情分析完了${NC}"
        else
            echo -e "${RED}✗ 感情分析失敗${NC}"
            continue
        fi
    fi
    
    # ----------------------------------------
    # Step 2: 立場検出
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 2/5] 立場検出（BERT）${NC}"
    
    STANCE_OUTPUT="modules/stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv"
    
    if [ -f "$STANCE_OUTPUT" ]; then
        echo -e "${GREEN}✓ スキップ: ${STANCE_OUTPUT} は既に存在します${NC}"
    else
        python modules/stance_detection/predict.py "$DATA_DIR"
        if [ -f "$STANCE_OUTPUT" ]; then
            echo -e "${GREEN}✓ 立場検出完了${NC}"
        else
            echo -e "${RED}✗ 立場検出失敗${NC}"
            continue
        fi
    fi
    
    # ----------------------------------------
    # Step 3: 特徴量統合
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 3/5] 特徴量統合${NC}"
    
    FEATURE_OUTPUT="modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv"
    
    if [ -f "$FEATURE_OUTPUT" ]; then
        echo -e "${GREEN}✓ スキップ: ${FEATURE_OUTPUT} は既に存在します${NC}"
    else
        python modules/feature_engineering/feature_builder.py \
            --sentiment_csv "$SENTIMENT_OUTPUT" \
            --stance_csv "$STANCE_OUTPUT"
        
        if [ -f "$FEATURE_OUTPUT" ]; then
            echo -e "${GREEN}✓ 特徴量統合完了${NC}"
        else
            echo -e "${RED}✗ 特徴量統合失敗${NC}"
            continue
        fi
    fi
    
    # ----------------------------------------
    # Step 4: 可視化（炎上期間の確認用）
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 4/5] 特徴量可視化${NC}"
    
    VIS_OUTPUT="modules/flame_detection/outputs/${TOPIC}/${TOPIC}_feature_trends.png"
    
    python visualize_features.py "$FEATURE_OUTPUT" "$VIS_OUTPUT"
    
    if [ -f "$VIS_OUTPUT" ]; then
        echo -e "${GREEN}✓ 可視化完了: ${VIS_OUTPUT}${NC}"
    fi
    
    # ----------------------------------------
    # Step 5: ラベリング（手動→スキップ）
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 5/5] ラベリング確認${NC}"
    
    LABEL_CONFIG="modules/flame_detection/label_config_${TOPIC}.yaml"
    LABELED_OUTPUT="modules/flame_detection/outputs/${TOPIC}/${TOPIC}_labeled.csv"
    
    if [ ! -f "$LABEL_CONFIG" ]; then
        echo -e "${YELLOW}⚠️  ${LABEL_CONFIG} が見つかりません${NC}"
        echo -e "${YELLOW}   → ${VIS_OUTPUT} を確認して、炎上期間を設定してください${NC}"
        echo -e "${YELLOW}   → その後、以下のコマンドでラベリング:${NC}"
        echo -e "${BLUE}   python modules/flame_detection/label_windows.py \\${NC}"
        echo -e "${BLUE}     ${FEATURE_OUTPUT} \\${NC}"
        echo -e "${BLUE}     ${LABEL_CONFIG} \\${NC}"
        echo -e "${BLUE}     ${LABELED_OUTPUT}${NC}"
        continue
    fi
    
    if [ -f "$LABELED_OUTPUT" ]; then
        echo -e "${GREEN}✓ ラベル済みデータ: ${LABELED_OUTPUT}${NC}"
    else
        echo -e "${YELLOW}📋 ラベリング実行中...${NC}"
        python modules/flame_detection/label_windows.py \
            "$FEATURE_OUTPUT" \
            "$LABEL_CONFIG" \
            "$LABELED_OUTPUT"
        
        if [ -f "$LABELED_OUTPUT" ]; then
            echo -e "${GREEN}✓ ラベリング完了${NC}"
        else
            echo -e "${RED}✗ ラベリング失敗${NC}"
            continue
        fi
    fi
    
    # ----------------------------------------
    # Step 6: モデル学習
    # ----------------------------------------
    echo -e "\n${YELLOW}[Step 6/5] モデル学習${NC}"
    
    MODEL_OUTPUT="modules/flame_detection/outputs/${TOPIC}/model/model.pkl"
    
    if [ -f "$MODEL_OUTPUT" ]; then
        echo -e "${GREEN}✓ スキップ: モデルは既に存在します${NC}"
    else
        python modules/flame_detection/train_classifier.py \
            "$LABELED_OUTPUT" \
            "modules/flame_detection/outputs/${TOPIC}/model/"
        
        if [ -f "$MODEL_OUTPUT" ]; then
            echo -e "${GREEN}✓ モデル学習完了${NC}"
        else
            echo -e "${RED}✗ モデル学習失敗${NC}"
            continue
        fi
    fi
    
    echo -e "${GREEN}✅ ${TOPIC} の処理が完了しました！${NC}"
done

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}🎉 全トピックの処理が完了しました！${NC}"
echo -e "${BLUE}============================================================${NC}"

# 結果サマリー
echo -e "\n${GREEN}📊 結果サマリー:${NC}"
for TOPIC in "${TOPICS[@]}"; do
    MODEL_FILE="modules/flame_detection/outputs/${TOPIC}/model/model.pkl"
    if [ -f "$MODEL_FILE" ]; then
        echo -e "  ✅ ${TOPIC}"
        EVAL_FILE="modules/flame_detection/outputs/${TOPIC}/model/evaluation.txt"
        if [ -f "$EVAL_FILE" ]; then
            ACCURACY=$(grep "Accuracy:" "$EVAL_FILE" | awk '{print $2}')
            F1=$(grep "F1-Score:" "$EVAL_FILE" | awk '{print $2}')
            echo -e "      Accuracy: ${ACCURACY}, F1-Score: ${F1}"
        fi
    else
        echo -e "  ⏳ ${TOPIC} (未完了)"
    fi
done

echo ""
