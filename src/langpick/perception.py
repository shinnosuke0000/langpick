"""知覚: オープン語彙物体検出（OWLv2）と、比較用の色ベースライン.

課題要件の「AI の活用」はここ。学習はせず、事前学習済み VLM の推論だけを使う。
``ColorBaseline`` は「VLM を使う必然性」を示すための対照実験であり、
色だけで決め打ちするため *形* を問う指示には原理的に答えられない。
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np

# 環境変数で差し替え可能にしてある。社内ミラーや、
# 事前にダウンロードしたローカルディレクトリを指したいときに使う。
DEFAULT_MODEL = os.environ.get("LANGPICK_MODEL",
                               "google/owlv2-base-patch16-ensemble")


@dataclasses.dataclass
class Detection:
    """1 件の検出結果."""

    query: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1（画素）
    source: str = "owlv2"

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def contains(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.box
        return (x0 - margin <= point[0] <= x1 + margin
                and y0 - margin <= point[1] <= y1 + margin)


class OwlV2Detector:
    """OWLv2 によるオープン語彙物体検出.

    「赤い缶」のような未知クラスでも、テキストを与えるだけで検出できる。
    物体クラスを事前に定義しない＝指示文の語彙に縛りがない、のが利点。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.model_name = model_name

    def detect(self, rgb: np.ndarray, queries: list[str],
               threshold: float = 0.10) -> list[Detection]:
        """``queries`` のそれぞれについて検出を行い、全件を返す."""
        from PIL import Image

        image = Image.fromarray(rgb)
        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        sizes = self.torch.tensor([[image.height, image.width]]).to(self.device)
        # transformers のバージョンによって後処理の名前が違う
        post = getattr(self.processor, "post_process_grounded_object_detection",
                       None) or self.processor.post_process_object_detection
        res = post(outputs, threshold=threshold, target_sizes=sizes)[0]

        # 新しい版は "text_labels"（文字列）、古い版は "labels"（クエリの添字）。
        # "text_labels" はキー自体は存在して中身が None のことがあるので、
        # get の既定値では拾えない点に注意。
        labels = res.get("text_labels")
        if labels is None:
            labels = res["labels"]
        out = []
        for score, label, box in zip(res["scores"], labels, res["boxes"]):
            query = label if isinstance(label, str) else queries[int(label)]
            out.append(Detection(
                query=query,
                score=float(score),
                box=tuple(float(v) for v in box),
            ))
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    def best(self, rgb: np.ndarray, query: str,
             threshold: float = 0.10) -> Detection | None:
        """1 つの語句について最もスコアの高い検出を返す."""
        hits = [d for d in self.detect(rgb, [query], threshold) if d.query == query]
        return hits[0] if hits else None


# --- 比較用ベースライン ------------------------------------------------------

_HSV_RANGES = {
    # 色名 -> (色相の下限, 色相の上限)  ※ 0-1 正規化
    "red": (0.94, 0.06),
    "green": (0.25, 0.45),
    "blue": (0.55, 0.72),
    "purple": (0.72, 0.85),
    "orange": (0.05, 0.10),
    "yellow": (0.12, 0.20),
}

_COLOR_WORDS = {
    "red": "red", "赤": "red", "green": "green", "緑": "green",
    "blue": "blue", "青": "blue", "purple": "purple", "紫": "purple",
    "orange": "orange", "オレンジ": "orange", "yellow": "yellow", "黄": "yellow",
}


class ColorBaseline:
    """HSV の閾値で最大色塊を取るだけの古典的ベースライン.

    VLM との比較対象。色語を含まない指示や、同色・異形の区別はできない。
    """

    source = "color"

    def detect(self, rgb: np.ndarray, queries: list[str],
               threshold: float = 0.0) -> list[Detection]:
        return [d for q in queries if (d := self.best(rgb, q)) is not None]

    def best(self, rgb: np.ndarray, query: str, threshold: float = 0.0):
        from matplotlib.colors import rgb_to_hsv
        from scipy import ndimage

        color = self._color_of(query)
        if color is None:
            return None

        hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = _HSV_RANGES[color]
        hue_ok = (h >= lo) | (h <= hi) if lo > hi else (h >= lo) & (h <= hi)
        mask = hue_ok & (s > 0.45) & (v > 0.25)

        labels, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        best = int(np.argmax(sizes)) + 1
        if sizes[best - 1] < 60:
            return None
        ys, xs = np.where(labels == best)
        return Detection(query=query, score=float(sizes[best - 1] / mask.size),
                         box=(float(xs.min()), float(ys.min()),
                              float(xs.max()), float(ys.max())),
                         source=self.source)

    @staticmethod
    def _color_of(query: str) -> str | None:
        q = query.lower()
        for word, color in _COLOR_WORDS.items():
            if re.search(word, q):
                return color
        return None


def verify_placement(detector, rgb_after: np.ndarray,
                     target_query: str, destination_query: str,
                     threshold: float = 0.08) -> tuple[bool, str, list[Detection]]:
    """実行後の画像だけを見て「置けたか」を判定する.

    対象物の中心が置き先の枠の内側にあれば成功とみなす。
    シミュレータの状態ではなく、実行後の *見え* で判定するのが要点。
    """
    dets = detector.detect(rgb_after, [target_query, destination_query], threshold)
    tgt = next((d for d in dets if d.query == target_query), None)
    dst = next((d for d in dets if d.query == destination_query), None)

    if dst is None:
        return False, f"置き先『{destination_query}』を検出できず判定不能", dets
    if tgt is None:
        return False, f"対象『{target_query}』が見つからない（トレイに隠れた可能性）", dets
    inside = dst.contains(tgt.center, margin=4.0)
    msg = ("対象の中心が置き先の内側にある" if inside
           else "対象が置き先の外にある")
    return inside, msg, dets
