import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"

import io
import re
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests
from PIL import Image

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

FUZZY_CUTOFF = 0.85   # 名字模糊匹配阈值
CROP_TOP_RATIO = 0.7  # 名字在卡面底部,先只处理底部 30%
DEBUG_DIR = os.environ.get("OCR_DEBUG_DIR")  # 设了就把每一遍的预处理图存下来

# 预处理 pass 列表:先便宜的(原图),不行再上"抠白字"。
# 白字 pass 参数 = (白色阈值 T, 放大倍数, 加粗核, 横向闭运算核)
#   T:      min(B,G,R) 达到多少算"白"。卡面名字是纯白,围巾的奶油色/金色 min 通道明显偏低
#   放大:   罗马音只有十几像素高,放大 4 倍检测器才看得见
#   加粗:   细衬线笔画(尤其 I、i、l 这种竖线)放大后仍偏细,膨胀后识别更稳
WHITE_PASSES = [
    (220, 4, 3, 7),
    (210, 4, 4, 7),
    (230, 4, 4, 7),
]

_ocr = None


def get_ocr():
    """延迟加载模型,方便单独测试预处理/匹配逻辑。"""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        print("正在加载 PaddleOCR 3.x 模型 ...")
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,  # 卡面是正的
            use_doc_unwarping=False,             # 关掉矫正,避免扭曲小字
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr


# ------------------------------------------------------------------
# 预处理
# ------------------------------------------------------------------
def white_text_image(bgr, T, up, thick, hclose):
    """只保留近白色像素 -> 黑字白底。去掉花纹背景,让检测器只看到文字。"""
    g = bgr.min(axis=2)  # 三通道最小值:只有真正的白才高
    # 软阈值:T-40 ~ T 之间线性过渡,保留细笔画的抗锯齿边缘,比硬二值化稳
    m = np.clip((g.astype(np.float32) - (T - 40)) / 40.0, 0, 1)
    m = (m * 255).astype(np.uint8)
    m = cv2.resize(m, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    if thick:
        m = cv2.dilate(m, np.ones((thick, thick), np.uint8))
    if hclose:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (hclose, 1)))
    return cv2.cvtColor(255 - m, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------
# OCR + 阅读顺序
# ------------------------------------------------------------------
def run_ocr(img_bgr):
    """返回 [(text, (x1,y1,x2,y2)), ...]"""
    items = []
    for res in get_ocr().predict(img_bgr):
        texts = res["rec_texts"]
        if "rec_boxes" in res and len(res["rec_boxes"]):
            boxes = [tuple(map(float, b)) for b in res["rec_boxes"]]
        else:
            boxes = []
            for poly in res["rec_polys"]:
                p = np.asarray(poly)
                boxes.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
        items.extend(zip(texts, boxes))
    return items


def reading_order(items):
    """按"行"聚类后从左到右排序。检测框顺序不保证是阅读顺序(实测 Nagi 会排在 Inoue 前面)。"""
    if not items:
        return []
    items = sorted(items, key=lambda it: (it[1][1] + it[1][3]) / 2)
    lines, cur = [], [items[0]]
    for it in items[1:]:
        cy = (it[1][1] + it[1][3]) / 2
        ref = (cur[0][1][1] + cur[0][1][3]) / 2
        h = max(cur[0][1][3] - cur[0][1][1], 1)
        if abs(cy - ref) <= 0.6 * h:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    out = []
    for ln in lines:
        out.extend(sorted(ln, key=lambda it: it[1][0]))
    return [t for t, _ in out]


# ------------------------------------------------------------------
# 名字匹配
# ------------------------------------------------------------------
def build_variants(key):
    parts = [p.lower() for p in re.findall(r"[A-Z][a-z]*", key)]
    variants = set()
    for i in range(len(parts)):
        rot = parts[i:] + parts[:i]
        variants.add("".join(rot))
        variants.add("".join(reversed(rot)))
    return variants


NAME_VARIANTS = {k: build_variants(k) for k in TOPIC_MAP}


def find_members(texts):
    # 去掉所有非字母 -> 字母间距很大的排版(I n o u e  N a g i)也能还原
    squashed = re.sub(r"[^a-z]", "", " ".join(texts).lower())
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
    # 去掉与更高分命中重叠的弱命中(KakiHaruka / KuromiHaruka 之类)
    hits.sort(reverse=True)
    kept = []
    for h in hits:
        if all(min(h[2], k[2]) - max(h[1], k[1]) <= 0.5 * (h[2] - h[1]) for k in kept):
            kept.append(h)
    return [k[3] for k in sorted(kept, key=lambda x: x[1])]


# ------------------------------------------------------------------
# 主流程:逐级尝试,命中就停(大多数卡第一遍就够,不会多花时间)
# ------------------------------------------------------------------
def candidate_passes(img_rgb):
    w, h = img_rgb.size
    bottom = cv2.cvtColor(np.array(img_rgb.crop((0, int(h * CROP_TOP_RATIO), w, h))), cv2.COLOR_RGB2BGR)
    yield "底部原图", bottom
    for T, up, thick, hclose in WHITE_PASSES:
        yield f"底部白字 T={T} x{up}", white_text_image(bottom, T, up, thick, hclose)
    full = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    yield "整图原图", full


def recognize(img_rgb, name="card"):
    all_texts = []
    for i, (label, im) in enumerate(candidate_passes(img_rgb)):
        if DEBUG_DIR:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_{i}.png"), im)
        texts = reading_order(run_ocr(im))
        print(f"  [{label}] 文本: {texts}")
        all_texts.extend(texts)
        tags = find_members(texts)
        if tags:
            return tags
    # 所有 pass 都单独没命中 -> 把各 pass 的文本合起来再试一次(Inoue / Nagi 可能分散在不同 pass)
    tags = find_members(all_texts)
    if tags:
        print("  [合并所有 pass 文本后命中]")
    return tags


def paddle_vision_ocr(image_url):
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        tags = recognize(img, name=re.sub(r"\W+", "_", image_url)[-20:])
        if not tags:
            print("  [未匹配] 没有找到成员名,建议人工检查")
        return tags
    except Exception as e:
        print(f"  [异常] 识别报错: {e}")
        return []


test_urls = [
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/17014567.jpg",  # 井上和:白色细衬线罗马音压在围巾上
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/17013961.jpg",  # 铃木/增田 双人合影卡
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/37014691.jpg",  # 超!乃木恋部 字母大间距排版
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/38114693.jpg",  # 井上和 单人卡
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/38114704.jpg",  # 林瑠奈 单人卡
    "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/27014700.jpg",  # 五百城
]

if __name__ == "__main__":
    print("\n" + "=" * 50)
    for url in test_urls:
        print(f"开始测试 URL: {url}")
        print(f"  [最终输出] 准备推送到频道的标签: {paddle_vision_ocr(url)}")
        print("=" * 50)
