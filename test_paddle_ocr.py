import os

# 跳过模型源连通性检查(启动更快);PIR 环境变量保留作为兜底
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"

import io
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests
from PIL import Image
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
    "MasudaMirine": 64, "MorihiraUrumi": 66, "YadaMoeka": 68,
}

FUZZY_CUTOFF = 0.85   # 模糊匹配阈值
CROP_TOP_RATIO = 0.7  # 先只识别底部 30%,没结果再识别整图


def build_variants(key):
    """把 CamelCase 名字拆成片段,生成各种语序(姓名/名姓/反序)的无空格小写形式。"""
    parts = [p.lower() for p in re.findall(r"[A-Z][a-z]*", key)]
    variants = set()
    for i in range(len(parts)):
        rot = parts[i:] + parts[:i]
        variants.add("".join(rot))
        variants.add("".join(reversed(rot)))
    return variants


NAME_VARIANTS = {k: build_variants(k) for k in TOPIC_MAP}

print("正在加载 PaddleOCR 3.x 模型 ...")
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,  # 卡面是正的,关掉文档方向分类
    use_doc_unwarping=False,             # 关掉文档矫正,否则可能把卡面扭曲
    use_textline_orientation=False,      # 名字都是横排
    enable_mkldnn=False,                 # 3.x 参数名是 enable_mkldnn,不是 use_mkldnn
)


def run_ocr(img_bgr):
    """3.x 用 predict();返回识别出的文本行列表。"""
    texts = []
    for res in ocr.predict(img_bgr):
        texts.extend(res["rec_texts"])
    return texts


def squash(texts):
    """去掉所有非字母和空格 -> 字母间距很大的排版(I N O U E)也能还原。"""
    return re.sub(r"[^a-z]", "", " ".join(texts).lower())


def find_members(squashed):
    hits = []  # (score, start, end, key)
    for key, variants in NAME_VARIANTS.items():
        best = None
        for v in variants:
            idx = squashed.find(v)
            if idx >= 0:
                cand = (1.0, idx, idx + len(v), key)
            else:
                cand = None
                L = len(v)
                for size in (L - 1, L, L + 1):
                    for i in range(0, len(squashed) - size + 1):
                        r = SequenceMatcher(None, v, squashed[i:i + size]).ratio()
                        if r >= FUZZY_CUTOFF and (cand is None or r > cand[0]):
                            cand = (r, i, i + size, key)
            if cand and (best is None or cand[0] > best[0]):
                best = cand
        if best:
            hits.append(best)

    # 去掉与更高分命中重叠的弱命中(避免 KakiHaruka / KuromiHaruka 之类互相误报)
    hits.sort(reverse=True)
    kept = []
    for h in hits:
        if all(min(h[2], k[2]) - max(h[1], k[1]) <= 0.5 * (h[2] - h[1]) for k in kept):
            kept.append(h)
    return [k[3] for k in sorted(kept, key=lambda x: x[1])]


def paddle_vision_ocr(image_url):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = img.size

        regions = [
            ("底部裁剪", img.crop((0, int(h * CROP_TOP_RATIO), w, h))),
            ("整图", img),
        ]
        for label, region in regions:
            bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
            texts = run_ocr(bgr)
            print(f"  [{label}] 原始文本行: {texts}")
            tags = find_members(squash(texts))
            if tags:
                return tags

        print("  [未匹配] 没有找到成员名,建议人工检查")
        return []
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

if __name__ == "__main__":
    print("\n" + "=" * 50)
    for url in test_urls:
        print(f"开始测试 URL: {url}")
        print(f"  [最终输出] 准备推送到频道的标签: {paddle_vision_ocr(url)}")
        print("=" * 50)
