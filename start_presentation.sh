#!/bin/bash
# プレゼンサイト起動スクリプト

echo "========================================="
echo "🔥 炎上検知AI プレゼンサイト"
echo "========================================="
echo ""

# 既存プロセスを停止
pkill -f "streamlit.*presentation_site" 2>/dev/null || true
sleep 1

# Streamlitを起動
echo "起動中..."
streamlit run presentation_site.py \
  --server.port 8502 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false

echo ""
echo "========================================="
echo "✅ プレゼンサイトが起動しました"
echo ""
echo "📱 ローカルアクセス:"
echo "   http://localhost:8502"
echo ""
echo "🌐 ネットワークアクセス:"
echo "   http://$(hostname -I | awk '{print $1}'):8502"
echo ""
echo "⚠️  停止するには Ctrl+C を押してください"
echo "========================================="
