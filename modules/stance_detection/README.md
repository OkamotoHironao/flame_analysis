# Stance Detection（立場分類）モジュール

ツイートの立場を **AGAINST（反対）/ FAVOR（賛成）/ NEUTRAL（中立）** に分類するBERTベースのモデル。

## 📂 ディレクトリ構成

```
stance_detection/
├── dataset.py          # データセットクラス
├── train.py            # 学習スクリプト
├── predict.py          # 推論スクリプト
├── model/              # 学習済みモデル（自動生成）
│   ├── best_model.pth
│   ├── config.json
│   └── tokenizer_config.json
├── outputs/            # 推論結果（自動生成）
└── README.md           # このファイル
