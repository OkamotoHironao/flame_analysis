#!/bin/bash
# Cloudflare Tunnelを使った発表用デモサイト公開スクリプト
# 接続数無制限・完全無料・高速

echo "🔧 Cloudflare Tunnel デモサイト起動"
echo "================================"

# cloudflaredがインストールされているか確認
if ! command -v cloudflared &> /dev/null; then
    echo "📥 cloudflaredをインストール中..."
    
    # アーキテクチャ判定
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared-linux-amd64.deb
        rm cloudflared-linux-amd64.deb
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
        sudo dpkg -i cloudflared-linux-arm64.deb
        rm cloudflared-linux-arm64.deb
    else
        echo "❌ サポートされていないアーキテクチャ: $ARCH"
        exit 1
    fi
    
    echo "✅ cloudflaredインストール完了"
fi

# 仮想環境の有効化
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ 仮想環境を有効化しました"
fi

# Streamlitをバックグラウンドで起動
echo ""
echo "🚀 Streamlitサーバーを起動中..."
streamlit run dashboard.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "   PID: $STREAMLIT_PID"

# Streamlitの起動待機
sleep 5

# Cloudflare Tunnelでトンネル作成
echo ""
echo "🌐 Cloudflare Tunnelを作成中..."
cloudflared tunnel --url http://localhost:8501 --no-autoupdate > /tmp/cloudflared.log 2>&1 &
CLOUDFLARED_PID=$!

# URL取得待機
echo "   URL取得中..."
sleep 3

# ログからURL抽出（最大30秒待機）
for i in {1..15}; do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -n1)
    if [ ! -z "$TUNNEL_URL" ]; then
        break
    fi
    sleep 2
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Cloudflare Tunnel URLの取得に失敗しました"
    echo "   ログを確認してください: /tmp/cloudflared.log"
    kill $STREAMLIT_PID $CLOUDFLARED_PID 2>/dev/null
    exit 1
fi

# QRコード生成
echo ""
echo "📱 QRコードを生成中..."
python3 << EOF
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

url = "$TUNNEL_URL"

# QRコード生成（高解像度）
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

qr_img = qr.make_image(fill_color="black", back_color="white")

# キャンバス作成
canvas_width = 700
canvas_height = 850
canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(canvas)

# QRコード配置
qr_img = qr_img.resize((550, 550))
canvas.paste(qr_img, (75, 80))

# テキスト追加
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    font_url = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except:
    font_title = ImageFont.load_default()
    font_url = ImageFont.load_default()
    font_info = ImageFont.load_default()

# タイトル
title = "🔥 Flame Detection AI Demo"
bbox = draw.textbbox((0, 0), title, font=font_title)
title_width = bbox[2] - bbox[0]
draw.text(((canvas_width - title_width) // 2, 20), title, fill="#FF6B35", font=font_title)

# URL
bbox = draw.textbbox((0, 0), url, font=font_url)
url_width = bbox[2] - bbox[0]
draw.text(((canvas_width - url_width) // 2, 660), url, fill="#0066CC", font=font_url)

# 説明文
instruction = "Scan QR code with your smartphone"
bbox = draw.textbbox((0, 0), instruction, font=font_info)
inst_width = bbox[2] - bbox[0]
draw.text(((canvas_width - inst_width) // 2, 710), instruction, fill="#333333", font=font_info)

# Cloudflareバッジ
powered = "Powered by Cloudflare Tunnel (Unlimited Access)"
bbox = draw.textbbox((0, 0), powered, font=font_info)
powered_width = bbox[2] - bbox[0]
draw.text(((canvas_width - powered_width) // 2, 750), powered, fill="#666666", font=font_info)

# 保存
output_path = "demo_qr_code.png"
canvas.save(output_path, quality=95)
print(f"✅ QRコードを保存しました: {output_path}")
EOF

# 結果表示
echo ""
echo "========================================"
echo "✅ デモサイトが公開されました！"
echo "========================================"
echo ""
echo "🌐 公開URL: $TUNNEL_URL"
echo "📱 QRコード: demo_qr_code.png"
echo ""
echo "🎯 Cloudflare Tunnelの特徴:"
echo "   ✅ 接続数無制限（ngrokの40接続/分制限なし）"
echo "   ✅ 完全無料"
echo "   ✅ Cloudflare CDN経由で高速"
echo "   ✅ HTTPS自動対応"
echo "   ✅ DDoS保護機能付き"
echo ""
echo "⚠️  注意事項:"
echo "   - URLは一時的です（発表終了後に無効化推奨）"
echo "   - 同時接続数の制限はありません"
echo ""
echo "🛑 停止方法:"
echo "   - Ctrl+C を押すか、以下を実行:"
echo "   kill $STREAMLIT_PID $CLOUDFLARED_PID"
echo ""
echo "📝 ログファイル:"
echo "   - Streamlit: /tmp/streamlit.log"
echo "   - Cloudflare: /tmp/cloudflared.log"
echo ""
echo "🎉 大規模聴講者でも安心してご利用いただけます！"

# トラップ設定
trap "echo ''; echo '🛑 サーバーを停止中...'; kill $STREAMLIT_PID $CLOUDFLARED_PID 2>/dev/null; echo '✅ 停止しました'; exit" INT

# プロセス監視
wait
