"""検出枠（2D）から把持姿勢（3D）を推定する.

入力は「バウンディングボックス」と「そのフレームの RGB-D」だけ。
シミュレータの物体座標には触れない。

手順
----
1. 枠の内側の画素を点群に戻す
2. 作業台の高さより上にある点だけを残す（＝机を捨てる）
3. 枠内に別の物体が入り込むことがあるので、中央値から外れた点を 2 段階で捨てる
4. 残った点の Z の上位パーセンタイルを天面高さとみなす
5. **天面付近の点だけ**を使い、その両端の中点を物体の軸とする
   （単眼だと手前の面しか見えず、点群全体の重心は手前へ約 2cm 偏るため）
6. 天面から ``grip_depth`` だけ下を掴む
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .camera import CameraView
from .scene import MAT_TOP_Z


@dataclasses.dataclass
class GraspPose:
    """推定した把持位置と、その信頼度の材料."""

    position: np.ndarray    # 掴みにいく指先中心のワールド座標
    top_z: float            # 推定した天面の高さ
    n_points: int           # 推定に使った点の数
    spread: float           # 点群の水平方向の広がり [m]（物体サイズの目安）

    @property
    def ok(self) -> bool:
        return self.n_points >= 30 and self.spread < 0.12


def estimate_grasp(view: CameraView,
                   box: tuple[float, float, float, float],
                   table_z: float = MAT_TOP_Z,
                   clearance: float = 0.008,
                   grip_depth: float = 0.015) -> GraspPose:
    """矩形 ``box`` が囲む物体の把持位置を推定する."""
    pts = view.point_cloud(box, stride=1)
    if pts.size == 0:
        return GraspPose(np.zeros(3), table_z, 0, np.inf)

    above = pts[pts[:, 2] > table_z + clearance]
    if above.shape[0] < 10:
        return GraspPose(np.zeros(3), table_z, int(above.shape[0]), np.inf)

    # 枠に紛れ込んだ別物体・背景を落とす（中央値から 6cm → 3.5cm の 2 段階）
    for radius in (0.06, 0.035):
        center = np.median(above[:, :2], axis=0)
        dist = np.linalg.norm(above[:, :2] - center, axis=1)
        trimmed = above[dist < radius]
        if trimmed.shape[0] < 10:
            break
        above = trimmed

    top_z = float(np.percentile(above[:, 2], 92))

    # 単眼なのでカメラ側の面だけが見え、点群全体の重心は手前へ 2cm ほど偏る。
    # 凸物体を上から見ると **天面は全部見えている** ので、天面の点だけを使えば
    # 偏りのない物体軸が取れる（実測の平均誤差 18.4mm → 7.3mm）。
    top_pts = above[above[:, 2] > top_z - 0.008]
    src = top_pts if top_pts.shape[0] >= 25 else above
    # 天面も斜めから見るので手前側ほど点が密になる。中央値だと密な側へ寄るため、
    # 外れ値を切った上での「両端の中点」を物体軸とする（7.3mm → 3.2mm）。
    # 数値は scripts/ablation_grasp.py で再現できる。
    center = 0.5 * (np.percentile(src[:, :2], 2, axis=0)
                    + np.percentile(src[:, :2], 98, axis=0))
    spread = float(np.percentile(np.linalg.norm(above[:, :2] - center, axis=1), 90) * 2)

    grasp_z = max(top_z - grip_depth, table_z + 0.012)
    return GraspPose(
        position=np.array([center[0], center[1], grasp_z]),
        top_z=top_z,
        n_points=int(above.shape[0]),
        spread=spread,
    )


def estimate_place_point(view: CameraView,
                         box: tuple[float, float, float, float],
                         table_z: float = MAT_TOP_Z) -> np.ndarray:
    """トレイなど「置き先」の中心を推定する.

    縁の壁が一番高いので、天面ではなく **内側の底** を狙いたい。
    枠内の点の XY 中央値と、高さの中央値（壁より低い底面側）を使う。
    """
    pts = view.point_cloud(box, stride=1)
    if pts.size == 0:
        return np.array([0.5, 0.0, table_z])
    above = pts[pts[:, 2] > table_z - 0.002]
    if above.shape[0] < 10:
        above = pts
    center = np.median(above[:, :2], axis=0)
    rim_z = float(np.percentile(above[:, 2], 90))
    return np.array([center[0], center[1], rim_z])
