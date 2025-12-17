#!/usr/bin/env python3
"""
QRコード生成スクリプト
プレゼンサイトのURLからQRコードを生成
"""

import qrcode
from pathlib import Path

def generate_qr_code(url: str, output_path: str = "presentation_qr.png"):
    """
    URLからQRコードを生成
    
    Args:
        url: QRコード化するURL
        output_path: 出力ファイルパス
    """
    # QRコードインスタンス作成
    qr = qrcode.QRCode(
        version=1,  # 1-40、大きいほど容量大
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 高い誤り訂正
        box_size=10,  # ピクセルサイズ
        border=4,  # 余白
    )
    
    # データ追加
    qr.add_data(url)
    qr.make(fit=True)
    
    # 画像生成
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 保存
    img.save(output_path)
    print(f"✅ QRコード生成完了: {output_path}")
    print(f"📱 URL: {url}")
    print(f"📏 サイズ: {img.size}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用法: python generate_qr.py <URL>")
        print("")
        print("例:")
        print("  python generate_qr.py https://your-app.streamlit.app")
        print("  python generate_qr.py http://192.168.1.100:8502")
        sys.exit(1)
    
    url = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "presentation_qr.png"
    
    generate_qr_code(url, output)
