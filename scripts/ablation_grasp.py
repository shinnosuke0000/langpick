"""把持位置推定のアブレーション.

レポートで「19mm → 7mm → 3mm」と主張している改善を、誰でも再現できるようにする。
検出器の性能に左右されないよう、ここでは真値から作った理想的な bbox を使い、
**物体軸の決め方だけ**を 3 通り比べる。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from langpick import scene as S  # noqa: E402
from langpick.camera import Camera  # noqa: E402
from langpick.experiment import project  # noqa: E402

VARIANTS = ("点群全体の中央値", "天面の点の中央値", "天面の点の両端中点")


def center_of(above: np.ndarray, variant: str) -> np.ndarray:
    top_z = float(np.percentile(above[:, 2], 92))
    if variant == "点群全体の中央値":
        return np.median(above[:, :2], axis=0)
    top = above[above[:, 2] > top_z - 0.008]
    src = top if top.shape[0] >= 25 else above
    if variant == "天面の点の中央値":
        return np.median(src[:, :2], axis=0)
    return 0.5 * (np.percentile(src[:, :2], 2, axis=0)
                  + np.percentile(src[:, :2], 98, axis=0))


def bbox_of(view, center, half) -> tuple[float, float, float, float]:
    pts = np.array([project(view, np.asarray(center) + np.asarray(half) * [sx, sy, sz])
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def main() -> None:
    model, data = S.make_model("assets", offwidth=960, offheight=540)
    cam = Camera(model, 540, 960)
    view = cam.capture(data, "percept")

    errors: dict[str, list[float]] = {v: [] for v in VARIANTS}
    rows = []
    for obj in S.OBJECTS:
        truth = S.body_pos(model, data, obj.name)
        half = (obj.size[0], obj.size[0],
                obj.size[2] if obj.kind == "box" else obj.size[1])
        pts = view.point_cloud(bbox_of(view, truth, half), stride=1)
        above = pts[pts[:, 2] > S.MAT_TOP_Z + 0.008]
        for radius in (0.06, 0.035):
            c = np.median(above[:, :2], axis=0)
            above = above[np.linalg.norm(above[:, :2] - c, axis=1) < radius]

        row = [obj.name]
        for v in VARIANTS:
            err = float(np.linalg.norm(center_of(above, v) - truth[:2]) * 1000)
            errors[v].append(err)
            row.append(f"{err:.1f}")
        rows.append(row)

    print("| 物体 | " + " | ".join(VARIANTS) + " |")
    print("|---|" + "---|" * len(VARIANTS))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("| **平均 [mm]** | "
          + " | ".join(f"**{np.mean(errors[v]):.1f}**" for v in VARIANTS) + " |")
    print("\n（単位 mm。物体の一辺は 40mm、グリッパの最大開き幅は 80mm）")


if __name__ == "__main__":
    main()
