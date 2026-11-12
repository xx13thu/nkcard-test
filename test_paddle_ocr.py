import os
import requests
import re
import difflib
import io
from PIL import Image
import cv2
import numpy as np

# ==========================================
# 【极度关键】：必须在 import paddleocr 之前设置此环境变量！
# 关闭 Paddle 3.0+ 默认启用的新 PIR 引擎，退回稳定的 AST 执行模式
# 这是解决 ConvertPirAttribute2RuntimeAttribute 报错的官方解法
# ==========================================
os.environ["FLAGS_enable_pir_api"] = "0"

import logging
# 隐藏 PaddleOCR 烦人的底层警告日志
logging.getLogger("ppocr").setLevel(logging.ERROR)
from paddleocr import PaddleOCR

# --- 字典配置 ---
TOPIC_MAP = {
    "ItoRiria": 2, "IwamotoRenka": 4, "YoshidaAyanoChristie": 6, "EndoSakura": 8,
    "KakiHaruka": 10, "KanagawaSaya": 12, "ShibataYuna": 14, "TamuraMayu": 16,
    "TsutsuiAyame": 18, "KuromiHaruka": 20, "HayashiRuna": 22, "YumikiNao": 24,
    "IokiMao": 26, "IkedaTeresa": 28, "IchinoseMiku": 30, "InoueNagi": 32,
    "OkamotoHina": 34, "OgawaAya": 36, "OkudaIroha": 38, "KawasakiSakura": 40,
    "SugawaraSatsuki": 42, "TomisatoNao": 44, "NakanishiAruno": 46,
    "AtagoKokone": 48, "OkoshiHinano": 50, "OzuReina": 52, "KaibeAkari": 54, 
    "KawabataHina": 56, "SuzukiYuuna": 58, "SetoguchiMitsuki": 60, "NagashimaRio": 62, 
    "MasudaMirine": 64, "MorihiraUrumi": 66, "YadaMoeka": 68
}
LOWER_TO_REAL_MAP = {k.lower(): k for k in TOPIC_MAP.keys()}

print("正在加载 PaddleOCR 3.0+ 模型 (英文/罗马音专用)...")
# 【极度关键 2】：加上 use_mkldnn=False，彻底封堵 CPU 指令集报错
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

def paddle_vision_ocr(image_url):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        w, h = img.size
        # 裁剪掉上半部分背景，只留底部的名字区域
        cropped_img = img.crop((0, int(h * 0.7), w, h))
        
        img_array = np.array(cropped_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 最新版 API 直接传图即可，不需要 cls=True
        result = ocr.ocr(img_bgr)
        
        if not result or not result[0]:
            print("  [底层提取] 未识别到任何文字。")
            return []

        raw_texts = [line[1][0] for line in result[0]]
        raw_text_joined = " ".join(raw_texts)
        print(f"  [底层提取] 原始提取拼接文本: '{raw_text_joined}'")
        
        clean_str = re.sub(r'[^a-zA-Z\s]', '', raw_text_joined).lower()
        words = clean_str.split()
        if not words: return []
        
        en_tags = []
        candidate_chunks = []
        max_len = min(4, len(words))
        for n in range(1, max_len + 1):
            for i in range(len(words) - n + 1):
                candidate_chunks.append("".join(words[i:i+n]))

        valid_keys = list(LOWER_TO_REAL_MAP.keys())
        for chunk in candidate_chunks:
            # 80% 的高匹配阈值，防止烂字母硬凑成成员名
            matches = difflib.get_close_matches(chunk, valid_keys, n=1, cutoff=0.8)
            if matches:
                en_tags.append(LOWER_TO_REAL_MAP[matches[0]])
                
        if en_tags:
            return list(dict.fromkeys(en_tags))
        else:
            safe_tag = "".join(words).capitalize()
            return [safe_tag] if safe_tag else []
            
    except Exception as e:
        print(f"  [异常] 识别报错: {e}")
        return []
test_urls = [
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/17013961.jpg", # 铃木/增田 双人
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/37014691.jpg", # 超乃木恋部排版
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/17014567.jpg", # 井上和单人
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/38114704.jpg", # 林瑠奈单人
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/2507953.jpg"
]

print("\n" + "="*50)
for url in test_urls:
    print(f"开始测试 URL: {url}")
    tags = paddle_vision_ocr(url)
    print(f"  [最终输出] 准备推送到频道的标签: {tags}")
    print("="*50)
