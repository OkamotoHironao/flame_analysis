#!/bin/bash
# Cloudflare Tunnelを使った研究発表サイト公開スクリプト

echo "🔧 研究発表サイト起動（Cloudflare Tunnel）"
echo "=========================================="

# cloudflaredインストール確認
if ! command -v cloudflared &> /dev/null; then
    echo "📥 cloudflaredをインストール中..."
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

# Streamlit起動（研究発表サイト）
echo ""
echo "🚀 研究発表サイトを起動中..."
streamlit run presentation_site.py --server.port 8502 --server.headless true > /tmp/streamlit_presentation.log 2>&1 &
STREAMLIT_PID=$!
echo "   PID: $STREAMLIT_PID"

sleep 5

# Cloudflare Tunnel作成
echo ""
echo "🌐 Cloudflare Tunnelを作成中..."
cloudflared tunnel --url http://localhost:8502 --no-autoupdate > /tmp/cloudflared_presentation.log 2>&1 &
CLOUDFLARED_PID=$!

echo "   URL取得中..."
sleep 3

# URL抽出
for i in {1..15}; do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared_presentation.log | head -n1)
    if [ ! -z "$TUNNEL_URL" ]; then
        break
    fi
    sleep 2
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Cloudflare Tunnel URLの取得に失敗しました"
    echo "   ログ: /tmp/cloudflared_presentation.log"
    kill $STREAMLIT_PID $CLOUDFLARED_PID 2>/dev/null
    exit 1
fi

# QRコード生成
echo ""
echo "📱 QRコードを生成中..."
python3 << EOF
import qrcode
from PIL import Image, ImageDraw, ImageFont

url = "$TUNNEL_URL"

# QRコード生成
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
canvas_height = 900
canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(canvas)

# QRコード配置
qr_img = qr_img.resize((550, 550))
canvas.paste(qr_img, (75, 100))

# テキスト
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_url = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except:
    font_title = ImageFont.load_default()
    font_subtitle = ImageFont.load_default()
    font_url = ImageFont.load_default()
    font_info = ImageFont.load_default()

# タイトル
title = "🔥 炎上検知AI"
bbox = draw.textbbox((0, 0), title, font=font_title)
title_width = bbox[2] - bbox[0]
draw.text(((canvas_width - title_width) // 2, 20), title, fill="#FF6B35", font=font_title)

# サブタイトル
subtitle = "研究発表サイト"
bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
subtitle_width = bbox[2] - bbox[0]
draw.text(((canvas_width - subtitle_width) // 2, 70), subtitle, fill="#004E89", font=font_subtitle)

# URL
bbox = draw.textbbox((0, 0), url, font=font_url)
url_width = bbox[2] - bbox[0]
draw.text(((canvas_width - url_width) // 2, 680), url, fill="#0066CC", font=font_url)

# 説明
instruction1 = "スマホでQRコードをスキャン"
bbox = draw.textbbox((0, 0), instruction1, font=font_info)
inst_width = bbox[2] - bbox[0]
draw.text(((canvas_width - inst_width) // 2, 730), instruction1, fill="#333333", font=font_info)

instruction2 = "または上記URLにアクセス"
bbox = draw.textbbox((0, 0), instruction2, font=font_info)
inst_width = bbox[2] - bbox[0]
draw.text(((canvas_width - inst_width) // 2, 760), instruction2, fill="#666666", font=font_info)

# フッター
powered = "Powered by Cloudflare Tunnel (接続数無制限)"
bbox = draw.textbbox((0, 0), powered, font=font_info)
powered_width = bbox[2] - bbox[0]
draw.text(((canvas_width - powered_width) // 2, 820), powered, fill="#999999", font=font_info)

# 保存
output_path = "presentation_qr_code.png"
canvas.save(output_path, quality=95)
print(f"✅ QRコード保存: {output_path}")
EOF

# 結果表示
echo ""
echo "=========================================="
echo "✅ 研究発表サイトが公開されました！"
echo "=========================================="
echo ""
echo "🌐 公開URL: $TUNNEL_URL"
echo "📱 QRコード: presentation_qr_code.png"
echo ""
echo "📄 サイト内容:"
echo "   - 研究概要"
echo "   - システムアーキテクチャ"
echo "   - モデル比較実験（6モデル）"
echo "   - 特徴量分析（SHAP）"
echo "   - 主要な知見"
echo "   - 今後の課題"
echo ""
echo "🎯 特徴:"
echo "   ✅ 接続数無制限"
echo "   ✅ インタラクティブなグラフ"
echo "   ✅ スマホ対応"
echo "   ✅ 完全無料"
echo ""
echo "🛑 停止: Ctrl+C または kill $STREAMLIT_PID $CLOUDFLARED_PID"
echo ""
echo "📝 ログ:"
echo "   - Streamlit: /tmp/streamlit_presentation.log"
echo "   - Cloudflare: /tmp/cloudflared_presentation.log"
echo ""
echo "🎉 発表を頑張ってください！"

# トラップ設定
trap "echo ''; echo '🛑 サーバー停止中...'; kill $STREAMLIT_PID $CLOUDFLARED_PID 2>/dev/null; echo '✅ 停止完了'; exit" INT

wait
