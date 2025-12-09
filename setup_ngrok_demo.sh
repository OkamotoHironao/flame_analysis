#!/bin/bash
# ngrokを使った発表用デモサイト公開スクリプト

echo "🔧 発表用デモサイト起動スクリプト"
echo "================================"

# ngrokがインストールされているか確認
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrokがインストールされていません"
    echo ""
    echo "📥 インストール方法:"
    echo "1. https://ngrok.com/ でアカウント作成（無料）"
    echo "2. Linux用のngrokをダウンロード:"
    echo "   curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
    echo "   echo \"deb https://ngrok-agent.s3.amazonaws.com buster main\" | sudo tee /etc/apt/sources.list.d/ngrok.list"
    echo "   sudo apt update && sudo apt install ngrok"
    echo "3. 認証トークン設定:"
    echo "   ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

# 仮想環境の有効化
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ 仮想環境を有効化しました"
else
    echo "⚠️  仮想環境が見つかりません"
fi

# Streamlitをバックグラウンドで起動
echo ""
echo "🚀 Streamlitサーバーを起動中..."
streamlit run dashboard.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "   PID: $STREAMLIT_PID"

# Streamlitの起動待機
sleep 5

# ngrokでトンネル作成
echo ""
echo "🌐 ngrokトンネルを作成中..."
ngrok http 8501 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# ngrok URLの取得（最大30秒待機）
echo "   URL取得中..."
sleep 3

for i in {1..10}; do
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | cut -d'"' -f4 | head -n1)
    if [ ! -z "$NGROK_URL" ]; then
        break
    fi
    sleep 2
done

if [ -z "$NGROK_URL" ]; then
    echo "❌ ngrok URLの取得に失敗しました"
    kill $STREAMLIT_PID $NGROK_PID 2>/dev/null
    exit 1
fi

# QRコードを生成
echo ""
echo "📱 QRコードを生成中..."
python3 << EOF
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

url = "$NGROK_URL"

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

# キャンバス作成（QR + テキスト）
canvas_width = 600
canvas_height = 750
canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(canvas)

# QRコードを中央配置
qr_img = qr_img.resize((500, 500))
canvas.paste(qr_img, (50, 50))

# テキスト追加
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    font_url = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_url = ImageFont.load_default()

# タイトル
title = "Flame Detection Demo"
bbox = draw.textbbox((0, 0), title, font=font_title)
title_width = bbox[2] - bbox[0]
draw.text(((canvas_width - title_width) // 2, 570), title, fill="black", font=font_title)

# URL
bbox = draw.textbbox((0, 0), url, font=font_url)
url_width = bbox[2] - bbox[0]
draw.text(((canvas_width - url_width) // 2, 620), url, fill="blue", font=font_url)

# 説明文
instruction = "QR code scan or access the URL above"
bbox = draw.textbbox((0, 0), instruction, font=font_url)
inst_width = bbox[2] - bbox[0]
draw.text(((canvas_width - inst_width) // 2, 680), instruction, fill="gray", font=font_url)

# 保存
output_path = "demo_qr_code.png"
canvas.save(output_path)
print(f"✅ QRコードを保存しました: {output_path}")
EOF

# 結果表示
echo ""
echo "================================"
echo "✅ デモサイトが公開されました！"
echo "================================"
echo ""
echo "🌐 公開URL: $NGROK_URL"
echo "📱 QRコード: demo_qr_code.png"
echo ""
echo "📊 管理画面: http://localhost:4040 (トラフィック確認用)"
echo ""
echo "⚠️  注意事項:"
echo "   - このURLは発表終了後に無効化されます"
echo "   - ngrok無料版は8時間で自動切断されます"
echo "   - 同時アクセス数の制限があります（40接続/分）"
echo ""
echo "🛑 停止方法:"
echo "   - Ctrl+C を押すか、以下を実行:"
echo "   kill $STREAMLIT_PID $NGROK_PID"
echo ""
echo "📝 ログファイル:"
echo "   - Streamlit: /tmp/streamlit.log"
echo "   - ngrok: /tmp/ngrok.log"
echo ""
echo "発表を頑張ってください！🎉"

# トラップ設定（Ctrl+Cで両方のプロセスを終了）
trap "echo ''; echo '🛑 サーバーを停止中...'; kill $STREAMLIT_PID $NGROK_PID 2>/dev/null; echo '✅ 停止しました'; exit" INT

# プロセスが生きている限り待機
wait
