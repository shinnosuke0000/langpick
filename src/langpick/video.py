"""動画の書き出しと、フレームへの情報の焼き込み.

提出動画は「編集なしでそのまま出せる」ことを狙っている。
指示文・検出結果・フェーズ・自己検証の判定をすべてフレームに描き込む。
"""

from __future__ import annotations

import dataclasses
import pathlib

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 日本語が出せるフォントを順に探す（Colab / Windows / mac を想定）
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
            except OSError:
                continue
    _font_cache[size] = ImageFont.load_default(size)
    return _font_cache[size]


ACCENT = (255, 214, 10)
OK_COLOR = (64, 214, 122)
NG_COLOR = (255, 99, 92)
PALETTE = ((255, 214, 10), (64, 214, 122), (108, 190, 255), (255, 138, 216))


@dataclasses.dataclass
class Banner:
    """フレーム上部に出す情報."""

    instruction_ja: str = ""
    instruction_en: str = ""
    phase: str = ""
    note: str = ""
    verdict: str | None = None  # "SUCCESS" / "FAILURE" / None


def _box(draw, xy, fill, radius=8):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def top_per_query(detections: list) -> list:
    """クエリごとに最高スコアの 1 件だけを残す（画面が枠だらけになるのを防ぐ）."""
    best: dict[str, object] = {}
    for det in detections:
        if det.query not in best or det.score > best[det.query].score:
            best[det.query] = det
    return list(best.values())


def annotate(frame: np.ndarray, banner: Banner,
             detections: list | None = None,
             scale: float = 1.0,
             marker: tuple[float, float] | None = None) -> np.ndarray:
    """1 フレームに指示文・検出枠・フェーズを描き込む.

    ``marker`` を渡すと、その画素位置に十字を描く（推定した把持点の可視化）。
    """
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    s = scale

    # --- 検出枠 ---
    for i, det in enumerate(detections or []):
        x0, y0, x1, y1 = det.box
        color = PALETTE[i % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(3 * s)))
        tag = f"{det.query}  {det.score:.2f}"
        f = font(int(17 * s))
        tw = d.textlength(tag, font=f)
        ty = max(0, y0 - int(23 * s))
        _box(d, [x0, ty, x0 + tw + 12 * s, ty + int(22 * s)], color + (235,), radius=5)
        d.text((x0 + 6 * s, ty + 2 * s), tag, font=f, fill=(15, 15, 18))

    # --- 推定した把持点の十字 ---
    if marker is not None:
        mx, my = marker
        r, lw = int(16 * s), max(2, int(2.5 * s))
        d.line([mx - r, my, mx + r, my], fill=ACCENT + (255,), width=lw)
        d.line([mx, my - r, mx, my + r], fill=ACCENT + (255,), width=lw)
        d.ellipse([mx - r * 0.45, my - r * 0.45, mx + r * 0.45, my + r * 0.45],
                  outline=ACCENT + (255,), width=lw)

    # --- 上部バナー ---
    bh = int(72 * s)
    d.rectangle([0, 0, w, bh], fill=(10, 12, 18, 215))
    f_main, f_sub = font(int(26 * s)), font(int(16 * s))
    d.text((int(18 * s), int(10 * s)), banner.instruction_ja, font=f_main, fill=(255, 255, 255))
    d.text((int(18 * s), int(44 * s)), banner.instruction_en, font=f_sub, fill=(150, 160, 175))

    if banner.phase:
        f = font(int(18 * s))
        tw = d.textlength(banner.phase, font=f)
        _box(d, [w - tw - int(34 * s), int(14 * s), w - int(16 * s), int(42 * s)],
             ACCENT + (240,))
        d.text((w - tw - int(25 * s), int(17 * s)), banner.phase, font=f, fill=(15, 15, 18))

    # --- 下部の注記 ---
    if banner.note:
        f = font(int(18 * s))
        d.rectangle([0, h - int(38 * s), w, h], fill=(10, 12, 18, 200))
        d.text((int(18 * s), h - int(31 * s)), banner.note, font=f, fill=(215, 220, 230))

    # --- 自己検証の判定 ---
    if banner.verdict:
        ok = banner.verdict.upper().startswith("S")
        color = OK_COLOR if ok else NG_COLOR
        f = font(int(30 * s))
        text = ("✔ " if ok else "✘ ") + banner.verdict
        tw = d.textlength(text, font=f)
        _box(d, [w / 2 - tw / 2 - 20 * s, h - int(104 * s),
                 w / 2 + tw / 2 + 20 * s, h - int(56 * s)], color + (240,), radius=10)
        d.text((w / 2 - tw / 2, h - int(99 * s)), text, font=f, fill=(12, 14, 18))

    return np.asarray(img)


def title_card(size: tuple[int, int], lines: list[tuple[str, int]],
               bg=(12, 14, 20)) -> np.ndarray:
    """タイトル・章立て用の静止画を作る. ``lines`` は (文字列, フォントサイズ)."""
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    total = sum(sz + 16 for _, sz in lines)
    y = (h - total) / 2
    for text, sz in lines:
        f = font(sz)
        tw = d.textlength(text, font=f)
        color = ACCENT if sz >= 44 else (225, 230, 240)
        d.text(((w - tw) / 2, y), text, font=f, fill=color)
        y += sz + 16
    return np.asarray(img)


def table_card(size: tuple[int, int], title: str,
               headers: list[str], rows: list[list[str]],
               marks: list[list[bool | None]] | None = None,
               footer: str = "", bg=(12, 14, 20)) -> np.ndarray:
    """実験結果などを表として描いた静止画を作る."""
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    f_title, f_head, f_cell, f_foot = font(30), font(18), font(19), font(17)

    d.text((40, 28), title, font=f_title, fill=ACCENT)

    n_col = len(headers)
    left, right = 40, w - 40
    col_w = (right - left) / n_col
    y = 86
    for i, head in enumerate(headers):
        d.text((left + i * col_w + 10, y), head, font=f_head, fill=(150, 160, 178))
    y += 30
    d.line([left, y, right, y], fill=(60, 68, 84), width=2)
    y += 12

    row_h = min(46, max(30, (h - y - 90) // max(1, len(rows))))
    for r, row in enumerate(rows):
        for i, cell in enumerate(row):
            color = (235, 238, 245)
            if marks is not None and marks[r][i] is not None:
                color = OK_COLOR if marks[r][i] else NG_COLOR
            d.text((left + i * col_w + 10, y + 6), cell, font=f_cell, fill=color)
        y += row_h
        d.line([left, y - 6, right, y - 6], fill=(36, 42, 54), width=1)

    if footer:
        for k, line in enumerate(footer.split("\n")):
            d.text((40, h - 70 + k * 24), line, font=f_foot, fill=(185, 195, 210))
    return np.asarray(img)


class Clip:
    """フレームを受け取り、その場で mp4 へ書き出していく.

    溜めてから書くと 960x540 × 2000 フレームで 1.5GB 以上をメモリに抱えることになり、
    Colab では落ちる。エンコーダへ逐次流し込んでメモリを一定に保つ。
    """

    def __init__(self, path: str | pathlib.Path, fps: int = 30, quality: int = 8):
        self.path = pathlib.Path(path)
        self.fps = fps
        self.quality = quality
        self.n_frames = 0
        self._writer = None
        self._size: tuple[int, int] | None = None

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._size = frame.shape[:2]
            self._writer = imageio.get_writer(
                str(self.path), fps=self.fps, quality=self.quality,
                macro_block_size=None, codec="libx264", pixelformat="yuv420p")
        elif frame.shape[:2] != self._size:
            raise ValueError(
                f"フレームのサイズが揃っていません: {frame.shape[:2]} != {self._size}")
        return self._writer

    def add(self, frame: np.ndarray, repeat: int = 1) -> None:
        w = self._ensure_writer(frame)
        frame = np.ascontiguousarray(frame)
        for _ in range(repeat):
            w.append_data(frame)
            self.n_frames += 1

    def hold(self, frame: np.ndarray, seconds: float) -> None:
        self.add(frame, repeat=max(1, int(self.fps * seconds)))

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def close(self) -> pathlib.Path:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.path

    # 後方互換: save() でも閉じられるようにしておく
    def save(self, path: str | pathlib.Path | None = None, quality: int = 8):
        if path is not None and pathlib.Path(path) != self.path:
            raise ValueError("Clip の出力先は生成時に決まります")
        return self.close()


class SimRecorder:
    """シミュレーション中、一定間隔でフレームを拾う."""

    def __init__(self, camera, clip: Clip, view_name: str = "cine"):
        self.camera, self.clip, self.view_name = camera, clip, view_name
        self.banner = Banner()
        self.detections: list = []
        self._next_t = 0.0

    def maybe_capture(self, data, force: bool = False) -> None:
        if not force and data.time < self._next_t:
            return
        self._next_t = data.time + 1.0 / self.clip.fps
        frame = self.camera.render_rgb(data, self.view_name)
        self.clip.add(annotate(frame, self.banner, self.detections,
                               scale=frame.shape[1] / 640.0))
