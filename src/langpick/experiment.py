"""対照実験: オープン語彙検出 (OWLv2) vs 色ベースライン.

「VLM を使う必然性があるのか」を数字で示すための実験。
シーンには **同じ赤色で形の違う 2 物体**（赤いブロック / 赤い缶）を置いてある。
色だけを手がかりにするベースラインは、この 2 つを原理的に区別できない。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import scene as S
from .camera import CameraView
from .grasp import estimate_grasp
from .instruction import parse

# 指示の語 -> シーン中のボディ名（採点にのみ使う対応表）
NAME_OF = {
    ("red", "cube"): "red_cube",
    ("red", "can"): "red_can",
    ("green", "cube"): "green_cube",
    ("blue", "can"): "blue_can",
    ("purple", "box"): "purple_tray",
    ("yellow", "box"): "yellow_tray",
}

DEFAULT_INSTRUCTIONS = (
    "赤いブロックを紫のトレイに入れて",
    "赤い缶を黄色のトレイに入れて",
    "緑のブロックを黄色のトレイに入れて",
    "青い缶を紫のトレイに入れて",
)


@dataclasses.dataclass
class Trial:
    instruction: str
    detector: str
    query: str
    expected: str
    identified: str | None
    score: float
    grasp_error_mm: float

    @property
    def correct(self) -> bool:
        return self.identified == self.expected


def project(view: CameraView, p) -> tuple[float, float]:
    """ワールド座標を画素へ（採点用の射影）."""
    pc = view.rot.T @ (np.asarray(p, float) - view.pos)
    z = -pc[2]
    h, w = view.shape
    f = view.focal
    return pc[0] * f / z + w / 2.0, -pc[1] * f / z + h / 2.0


def identify(view: CameraView, model, data, box) -> str | None:
    """検出枠が実際にどの物体を囲っているかを、真値の射影で判定する."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    best, best_d = None, 1e9
    for obj in S.OBJECTS:
        u, v = project(view, S.body_pos(model, data, obj.name))
        d = float(np.hypot(u - cx, v - cy))
        if d < best_d:
            best, best_d = obj.name, d
    return best if best_d < 60 else None


def run(env, detectors: dict, instructions=DEFAULT_INSTRUCTIONS,
        threshold: float = 0.10) -> list[Trial]:
    """同一の画像に対して各検出器を走らせ、対象の同定精度を比べる."""
    env.reset()
    view = env.look()
    trials: list[Trial] = []

    for text in instructions:
        ins = parse(text)
        expected = NAME_OF[(ins.target.color, ins.target.noun)]
        truth = S.body_pos(env.model, env.data, expected)

        for name, det in detectors.items():
            hit = det.best(view.rgb, ins.target.query, threshold)
            if hit is None:
                trials.append(Trial(text, name, ins.target.query, expected,
                                    None, 0.0, float("nan")))
                continue
            identified = identify(view, env.model, env.data, hit.box)
            g = estimate_grasp(view, hit.box)
            err = float(np.linalg.norm(g.position[:2] - truth[:2]) * 1000)
            trials.append(Trial(text, name, ins.target.query, expected,
                                identified, hit.score, err))
    return trials


def to_table(trials: list[Trial]) -> str:
    """結果を Markdown の表にする（レポートへそのまま貼れる形）."""
    detectors = sorted({t.detector for t in trials})
    lines = ["| 指示（対象） | " + " | ".join(detectors) + " |",
             "|---|" + "---|" * len(detectors)]
    for text in dict.fromkeys(t.instruction for t in trials):
        row = [text.split("を")[0]]
        for d in detectors:
            t = next(x for x in trials if x.instruction == text and x.detector == d)
            if t.identified is None:
                row.append("検出なし ✗")
            elif t.correct:
                # 色閾値のスコアは「色塊の面積比」で信頼度ではないので出さない
                conf = f"{t.score:.2f}, " if t.score > 0.01 else ""
                row.append(f"{t.identified} ✓ ({conf}{t.grasp_error_mm:.0f}mm)")
            else:
                conf = f" ({t.score:.2f})" if t.score > 0.01 else ""
                row.append(f"{t.identified} ✗{conf}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    for d in detectors:
        sub = [t for t in trials if t.detector == d]
        acc = sum(t.correct for t in sub) / len(sub)
        errs = [t.grasp_error_mm for t in sub if t.correct and np.isfinite(t.grasp_error_mm)]
        mean_err = f"{np.mean(errs):.1f}mm" if errs else "—"
        lines.append(f"- **{d}**: 対象同定 {sum(t.correct for t in sub)}/{len(sub)} "
                     f"({acc:.0%})、正解時の把持位置誤差 平均 {mean_err}")
    return "\n".join(lines)
