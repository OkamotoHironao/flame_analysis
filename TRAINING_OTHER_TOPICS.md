# 他のトピックを学習させる手順

## 📋 手順書（トピックごとに実行）

### 対象トピック
- ✅ 松本人志（完了）
- ⏳ 寿司ペロ
- ⏳ 三苫
- ⏳ aespa

---

## 🔄 各トピックの学習手順

### **Step 0: データ確認**

```bash
# データがあることを確認
ls -l data/original/寿司ペロ/
ls -l data/original/三苫/
ls -l data/original/aespa/
```

---

### **Step 1: 感情分析（辞書ベース）**

```bash
# トピック名を設定
TOPIC="寿司ペロ"  # または "三苫", "aespa"

# 感情分析実行
python sentiment_analysis.py "data/original/${TOPIC}"
```

**出力先**: `data/processed/${TOPIC}_sentiment_1h.csv`

**確認**:
```bash
head data/processed/${TOPIC}_sentiment_1h.csv
wc -l data/processed/${TOPIC}_sentiment_1h.csv
```

---

### **Step 2: 立場検出（BERT）**

```bash
# 立場検出実行
python stance_predict.py "data/original/${TOPIC}"
```

**出力先**: `modules/stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv`

**確認**:
```bash
head modules/stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv
```

---

### **Step 3: 特徴量統合**

```bash
cd modules/feature_engineering

python feature_builder.py \
  --sentiment_csv ../../data/processed/${TOPIC}_sentiment_1h.csv \
  --stance_csv ../stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv

cd ../..
```

**出力先**: `modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv`

**確認**:
```bash
head modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv
wc -l modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv
```

---

### **Step 4: 可視化（炎上期間の特定）**

```bash
python visualize_features.py \
  modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv \
  modules/flame_detection/outputs/${TOPIC}_feature_trends.png
```

**出力**: グラフと推奨期間が表示される

**重要**: ターミナル出力の「推奨ラベリング期間」をメモする

---

### **Step 5: ラベル設定ファイル作成**

可視化結果を見て、炎上期間を設定ファイルに記述します。

```bash
cd modules/flame_detection

# 設定ファイルをコピー
cp label_config_matsumoto.yaml label_config_${TOPIC}.yaml

# 編集（viまたはVS Codeで）
vi label_config_${TOPIC}.yaml
```

**編集内容**:
```yaml
# 炎上期間設定ファイル - ${TOPIC}ケース
# 
# データ範囲: <可視化で確認した範囲>

controversy_periods:
  # 可視化結果から判明した炎上期間を記述
  - start: "2023-01-30 12:00:00"  # 例
    end: "2023-02-01 18:00:00"
    label: "メイン炎上期間"
    description: "投稿急増・ネガティブ率上昇"
  
  # 必要に応じて追加
  - start: "2023-02-03 00:00:00"
    end: "2023-02-03 12:00:00"
    label: "再燃"
    description: "2次炎上"
```

---

### **Step 6: ラベリング実行**

```bash
cd modules/flame_detection

python label_windows.py \
  ../feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv \
  label_config_${TOPIC}.yaml \
  outputs/${TOPIC}/${TOPIC}_labeled.csv
```

**確認**:
```bash
# ラベル分布を確認
python -c "
import pandas as pd
df = pd.read_csv('outputs/${TOPIC}/${TOPIC}_labeled.csv')
print(df['is_controversy'].value_counts())
print(f'炎上率: {df[\"is_controversy\"].mean()*100:.1f}%')
"
```

**理想的なバランス**:
- 炎上: 10〜30%
- 非炎上: 70〜90%

---

### **Step 7: モデル学習**

```bash
cd modules/flame_detection

# XGBoost
python train_classifier.py \
  outputs/${TOPIC}/${TOPIC}_labeled.csv \
  outputs/${TOPIC}/model/

# RandomForest（比較用）
python train_classifier.py \
  outputs/${TOPIC}/${TOPIC}_labeled.csv \
  outputs/${TOPIC}/model_rf/ \
  --model randomforest
```

**確認**:
```bash
cat outputs/${TOPIC}/model/evaluation.txt
```

---

## 🚀 実際のコマンド例（寿司ペロ）

```bash
# プロジェクトルートで実行
cd /home/h-okamoto/デスクトップ/sotuken/flame_analysis

# Step 1: 感情分析
python sentiment_analysis.py "data/original/寿司ペロ"

# Step 2: 立場検出
python stance_predict.py "data/original/寿司ペロ"

# Step 3: 特徴量統合
cd modules/feature_engineering
python feature_builder.py \
  --sentiment_csv ../../data/processed/寿司ペロ_sentiment_1h.csv \
  --stance_csv ../stance_detection/outputs/寿司ペロ/寿司ペロ_stance.csv
cd ../..

# Step 4: 可視化
python visualize_features.py \
  modules/feature_engineering/outputs/寿司ペロ/寿司ペロ_feature_table.csv \
  modules/flame_detection/outputs/寿司ペロ_feature_trends.png

# ここで可視化結果を確認 → 炎上期間を特定

# Step 5: 設定ファイル作成
cd modules/flame_detection
cp label_config_matsumoto.yaml label_config_寿司ペロ.yaml
# VS Codeで編集: label_config_寿司ペロ.yaml

# Step 6: ラベリング
python label_windows.py \
  ../feature_engineering/outputs/寿司ペロ/寿司ペロ_feature_table.csv \
  label_config_寿司ペロ.yaml \
  outputs/寿司ペロ/寿司ペロ_labeled.csv

# Step 7: 学習
python train_classifier.py \
  outputs/寿司ペロ/寿司ペロ_labeled.csv \
  outputs/寿司ペロ/model/
```

---

## 📝 全トピック用クイックコマンド

### 寿司ペロ
```bash
TOPIC="寿司ペロ"
python sentiment_analysis.py "data/original/${TOPIC}"
python stance_predict.py "data/original/${TOPIC}"
cd modules/feature_engineering && python feature_builder.py --sentiment_csv ../../data/processed/${TOPIC}_sentiment_1h.csv --stance_csv ../stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv && cd ../..
python visualize_features.py modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv modules/flame_detection/outputs/${TOPIC}_feature_trends.png
# → 可視化確認 → label_config編集 → ラベリング → 学習
```

### 三苫
```bash
TOPIC="三苫"
python sentiment_analysis.py "data/original/${TOPIC}"
python stance_predict.py "data/original/${TOPIC}"
cd modules/feature_engineering && python feature_builder.py --sentiment_csv ../../data/processed/${TOPIC}_sentiment_1h.csv --stance_csv ../stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv && cd ../..
python visualize_features.py modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv modules/flame_detection/outputs/${TOPIC}_feature_trends.png
```

### aespa
```bash
TOPIC="aespa"
python sentiment_analysis.py "data/original/${TOPIC}"
python stance_predict.py "data/original/${TOPIC}"
cd modules/feature_engineering && python feature_builder.py --sentiment_csv ../../data/processed/${TOPIC}_sentiment_1h.csv --stance_csv ../stance_detection/outputs/${TOPIC}/${TOPIC}_stance.csv && cd ../..
python visualize_features.py modules/feature_engineering/outputs/${TOPIC}/${TOPIC}_feature_table.csv modules/flame_detection/outputs/${TOPIC}_feature_trends.png
```

---

## ⚠️ トラブルシューティング

### エラー: ファイルが見つかりません
```bash
# データの存在確認
ls -R data/original/

# 出力ディレクトリ作成
mkdir -p data/processed
mkdir -p modules/stance_detection/outputs/${TOPIC}
mkdir -p modules/feature_engineering/outputs/${TOPIC}
mkdir -p modules/flame_detection/outputs/${TOPIC}/model
```

### エラー: タイムゾーンの問題
- データのタイムスタンプ形式を確認
- 必要に応じて `label_windows.py` のタイムゾーン処理を調整

### クラス不均衡（炎上率が極端）
- 炎上期間の設定を見直す
- 可視化グラフで明らかな急増期を含める
- 炎上率 10〜30% が理想

---

## 📊 進捗管理

| トピック | 感情分析 | 立場検出 | 特徴量 | 可視化 | ラベリング | 学習 |
|---------|---------|---------|--------|--------|----------|------|
| 松本人志 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 寿司ペロ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 三苫 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| aespa | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

チェックリストとして使ってください！
