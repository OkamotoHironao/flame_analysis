# インターネット公開型デモサイト - セットアップガイド

## 概要
ローカルのFlame Detection Dashboardをインターネット経由で公開します。  
会場のネットワーク設定に依存せず、QRコードまたはURLで誰でもアクセス可能です。

## 🚀 推奨方法の選択

### **大規模聴講（40人以上）→ Cloudflare Tunnel（無制限・無料）**
### **小規模聴講（40人未満）→ ngrok（シンプル・無料）**

---

## 方法1: Cloudflare Tunnel（接続数無制限・推奨）

### 特徴
| 項目 | 詳細 |
|------|------|
| **接続数制限** | **なし（無制限）** ✅ |
| **速度** | Cloudflare CDN経由で高速 |
| **セッション時間** | 無制限 |
| **HTTPS** | 自動対応 |
| **アカウント** | 不要 |
| **DDoS保護** | あり |

### インストール（1分）
```bash
# Ubuntu/Debian
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### 起動（1コマンド）
```bash
chmod +x setup_cloudflare_demo.sh
./setup_cloudflare_demo.sh
```

**これだけで完了！** アカウント登録も認証設定も不要です。

---

## 方法2: ngrok（小規模向け）

### 特徴
| 項目 | 詳細 |
|------|------|
| **接続数制限** | 40接続/分 ⚠️ |
| **速度** | やや遅い（米国経由） |
| **セッション時間** | 8時間 |
| **HTTPS** | 自動対応 |
| **アカウント** | 必要 |

### 事前準備（発表前日までに完了）

### 1. ngrokアカウント作成（無料）
1. https://ngrok.com/ にアクセス
2. 「Sign up」でアカウント作成（GitHubアカウントでも可）
3. ダッシュボードで「Authtoken」を取得

### 2. ngrokインストール

#### Linux (Ubuntu/Debian)
```bash
# 公式リポジトリから安定版をインストール
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

#### 別の方法（スナップ）
```bash
sudo snap install ngrok
```

### 3. ngrok認証設定
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```
※ `YOUR_AUTHTOKEN_HERE` を実際のトークンに置き換え

### 4. 必要なPythonパッケージインストール
```bash
source .venv/bin/activate
pip install qrcode pillow
```

---

## 発表当日の使用方法

### 起動手順（1コマンド）
```bash
chmod +x setup_ngrok_demo.sh
./setup_ngrok_demo.sh
```

**実行すると自動的に:**
1. Streamlitサーバーを起動（ポート8501）
2. ngrokトンネルを作成
3. 公開URLを取得
4. QRコードを生成（`demo_qr_code.png`）
5. URLとQRコードを画面表示

### 出力例
```
================================
✅ デモサイトが公開されました！
================================

🌐 公開URL: https://abc123.ngrok-free.app
📱 QRコード: demo_qr_code.png

📊 管理画面: http://localhost:4040
```

### QRコードの使用
1. `demo_qr_code.png` をプレゼン資料に挿入
2. 発表中にスライド表示
3. 聴講者がスマホでスキャン → 即アクセス

---

## 管理機能

### トラフィック監視
ブラウザで `http://localhost:4040` を開くと:
- リアルタイムアクセス数
- リクエスト/レスポンス詳細
- エラー監視

### サーバー停止
```bash
# Ctrl+C を押す（自動的に両方のプロセスを終了）
```

または手動で:
```bash
pkill -f streamlit
pkill -f ngrok
```

---

## サービス比較表

| 項目 | Cloudflare Tunnel | ngrok無料版 | ngrok有料版 |
|------|------------------|-------------|------------|
| **接続数** | **無制限** ✅ | 40接続/分 ⚠️ | 無制限 |
| **セッション時間** | 無制限 | 8時間 | 無制限 |
| **速度** | 高速（CDN） | 普通 | 普通 |
| **アカウント** | 不要 | 必要 | 必要 |
| **料金** | 無料 | 無料 | $8/月〜 |
| **DDoS保護** | あり | なし | あり |

**大規模発表では Cloudflare Tunnel 一択です！**

---

## トラブルシューティング

### Q1: ngrokがインストールできない
```bash
# 手動インストール（最新版）
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
ngrok --version
```

### Q2: QRコード生成が失敗する
```bash
# フォント不足の場合
sudo apt install fonts-dejavu-core
```

### Q3: URLが取得できない
```bash
# ngrok APIで確認
curl http://localhost:4040/api/tunnels | jq
```

### Q4: 発表中にURLが切れた
```bash
# 再起動すれば新しいURLが発行される
./setup_ngrok_demo.sh
```

### Q5: アクセスが遅い
- ngrok無料版は米国サーバー経由のため、やや遅延あり
- 事前にデモを軽量化（大きな画像を削減など）

---

## 有料版の検討（必要な場合）

発表で以下が必要なら有料版を検討:
- **固定URL**: `https://yourname.ngrok.app`（$8/月）
- **日本リージョン**: 低遅延（$8/月）
- **24時間稼働**: セッション制限なし

学会発表なら無料版で十分です。

---

## 発表当日チェックリスト

### 前日
- [ ] ngrok認証設定確認: `ngrok config check`
- [ ] スクリプト実行テスト: `./setup_ngrok_demo.sh`
- [ ] スマホでQRコード読み取りテスト
- [ ] 発表資料にQRコード画像を挿入

### 当日（発表30分前）
- [ ] スクリプト実行: `./setup_ngrok_demo.sh`
- [ ] URLアクセス確認（PCとスマホ両方）
- [ ] 管理画面起動: http://localhost:4040
- [ ] バックアップ用にURLをメモ

### 発表中
- [ ] QRコードスライド表示
- [ ] 「スマホでQRコードをスキャンしてください」とアナウンス
- [ ] 管理画面でアクセス数を確認（任意）

### 発表終了後
- [ ] Ctrl+Cでサーバー停止
- [ ] ログ確認（トラブルがあった場合）

---

## その他の代替案

### localtunnel（Node.js必要）
```bash
npm install -g localtunnel
lt --port 8501
```
※ 接続が不安定な場合あり

### serveo（SSH経由、シンプル）
```bash
ssh -R 80:localhost:8501 serveo.net
```
※ サービス停止中の場合あり

---

## まとめ

### 聴講者数に応じた推奨

| 聴講者数 | 推奨サービス | 理由 |
|---------|-------------|------|
| **50人以上** | **Cloudflare Tunnel** | 無制限接続、高速、無料 |
| **20〜50人** | Cloudflare Tunnel | 安全マージン確保 |
| **20人未満** | ngrok | シンプル、十分 |

### Cloudflare Tunnelの圧倒的メリット
- ✅ **接続数無制限**（最重要）
- ✅ アカウント不要（即利用可能）
- ✅ Cloudflare CDN経由で高速
- ✅ DDoS保護付き
- ✅ 1コマンドで即公開
- ✅ HTTPS自動対応
- ✅ QRコード自動生成

**大規模発表ならCloudflare Tunnel一択です！🚀**
