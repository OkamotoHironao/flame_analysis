#!/usr/bin/env python3
"""
設定値表示スクリプト

config/presentation_config.json から主要な値を表示し、
README.md や発表資料の更新時の参考情報として使用

Usage:
    python scripts/show_config_values.py
"""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config" / "presentation_config.json"

def main():
    if not CONFIG_FILE.exists():
        print(f"❌ 設定ファイルが見つかりません: {CONFIG_FILE}")
        return
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("📊 プロジェクト主要指標")
    print("=" * 60)
    
    # メトリクス
    metrics = config['metrics']
    print(f"\n✅ 性能指標:")
    print(f"  - 最高F1スコア: {metrics['latest_best_f1']}%")
    print(f"  - クロストピックF1: {metrics['cross_topic_f1']}%")
    print(f"  - 特徴量数: {metrics['num_features']}")
    print(f"  - 比較モデル数: {metrics['num_models_compared']}")
    print(f"  - デフォルトF1: {metrics['default_best_f1']}%")
    print(f"  - デフォルトモデル: {metrics['default_best_model']}")
    
    # データセット
    dataset = config['dataset']
    print(f"\n📂 データセット:")
    print(f"  - トピック数: {dataset['num_topics']}")
    print(f"  - トピック一覧:")
    for topic in dataset['topics']:
        print(f"    • {topic['name']} ({topic['category']})")
    
    # README.md用のマークダウン生成
    print("\n" + "=" * 60)
    print("📝 README.md用テキスト")
    print("=" * 60)
    print(f"""
## 主要な成果

- ✅ **最高F1スコア達成**: **{metrics['latest_best_f1']}%** (SVM/Logistic Regression)
- ✅ **特徴量エンジニアリング**: {metrics['num_features']}特徴量（時系列・感情・立場）で汎化性能向上
- ✅ **クロストピック評価**: 未知トピックで{metrics['cross_topic_f1']}% F1（汎化性能実証）
- ✅ **{metrics['num_models_compared']}モデル比較**: SVM, Logistic Regression, XGBoost, LightGBM, Random Forest, CatBoost
    """)
    
    print("\n" + "=" * 60)
    print("💡 設定値の更新方法")
    print("=" * 60)
    print(f"""
設定ファイル: {CONFIG_FILE.relative_to(Path.cwd())}

このファイルを編集することで、以下のファイルが自動的に反映されます:
- presentation_site.py (研究発表サイト)
- このスクリプト (show_config_values.py)

README.mdは手動更新が必要です（上記のテキストをコピー）
    """)

if __name__ == "__main__":
    main()
