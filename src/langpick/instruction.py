"""指示文を「対象」と「置き先」のスロットに分解する.

ここは意図的に **規則ベース** にしてある。
難しいのは「どの語がどの画素か」という接地（grounding）の方で、
そこを OWLv2 に任せている。語彙を増やしたければ ``VOCAB`` に足すだけでよく、
検出器側は再学習不要（オープン語彙検出の利点）。
"""

from __future__ import annotations

import dataclasses
import re

# 日本語・英語の表記 -> (英語クエリ用の語, 日本語表示用の語)
COLORS: dict[str, tuple[str, str]] = {
    "赤": ("red", "赤い"), "red": ("red", "赤い"),
    "緑": ("green", "緑の"), "green": ("green", "緑の"),
    "青": ("blue", "青い"), "blue": ("blue", "青い"),
    "紫": ("purple", "紫の"), "purple": ("purple", "紫の"),
    "黄": ("yellow", "黄色の"), "yellow": ("yellow", "黄色の"),
    "きいろ": ("yellow", "黄色の"), "キイロ": ("yellow", "黄色の"),
}

NOUNS: dict[str, tuple[str, str, str]] = {
    # 表記 -> (英語クエリ用の語, 日本語表示用の語, 種別)
    "ブロック": ("cube", "ブロック", "object"),
    "block": ("cube", "ブロック", "object"),
    "cube": ("cube", "ブロック", "object"),
    "キューブ": ("cube", "ブロック", "object"),
    "立方体": ("cube", "ブロック", "object"),
    "缶": ("can", "缶", "object"),
    "can": ("can", "缶", "object"),
    "円柱": ("can", "缶", "object"),
    "ボトル": ("can", "缶", "object"),
    "トレイ": ("box", "トレイ", "place"),
    "tray": ("box", "トレイ", "place"),
    "箱": ("box", "トレイ", "place"),
    "bin": ("box", "トレイ", "place"),
}

ARTICLE = {"a", "an"}


@dataclasses.dataclass
class Slot:
    color: str          # 英語の色名
    noun: str           # 英語の名詞
    kind: str           # "object" | "place"
    label_ja: str       # 日本語の表示名

    @property
    def query(self) -> str:
        """OWLv2 に渡す英語クエリ."""
        art = "an" if self.color[0] in "aeiou" else "a"
        return f"{art} {self.color} {self.noun}"


@dataclasses.dataclass
class ParsedInstruction:
    raw: str
    target: Slot
    destination: Slot

    @property
    def text_ja(self) -> str:
        return f"{self.target.label_ja}を{self.destination.label_ja}に入れて"

    @property
    def text_en(self) -> str:
        return f"Put {self.target.query} in {self.destination.query}"


def _scan(text: str) -> list[Slot]:
    """文中に現れる「色＋名詞」の組を出現順に拾う."""
    lower = text.lower()
    hits: list[tuple[int, str, str]] = []
    for table, tag in ((COLORS, "color"), (NOUNS, "noun")):
        for word in table:
            for m in re.finditer(re.escape(word.lower()), lower):
                hits.append((m.start(), tag, word))
    hits.sort()

    slots, pending_color = [], None
    for _, tag, word in hits:
        if tag == "color":
            pending_color = COLORS[word][0]
            pending_ja = COLORS[word][1]
        else:
            noun_en, noun_ja, kind = NOUNS[word]
            if pending_color is None:
                continue
            slots.append(Slot(pending_color, noun_en, kind, pending_ja + noun_ja))
            pending_color = None
    return slots


def parse(text: str) -> ParsedInstruction:
    """指示文から対象と置き先を取り出す.

    「赤いブロックを紫のトレイに入れて」/ "put the red block in the purple tray"
    のどちらでも動く。
    """
    slots = _scan(text)
    targets = [s for s in slots if s.kind == "object"]
    places = [s for s in slots if s.kind == "place"]
    if not targets or not places:
        raise ValueError(
            f"指示を解釈できません: {text!r}\n"
            "「<色><物体>を<色>のトレイに入れて」の形で書いてください。"
            f"\n使える色: {sorted({v[0] for v in COLORS.values()})}"
            f"\n使える物体: {sorted({v[0] for v in NOUNS.values()})}"
        )
    return ParsedInstruction(raw=text, target=targets[0], destination=places[0])
