"""カメラモデルと 2D→3D 逆投影.

本プロジェクトの中核。検出器が返す「画像上の点」を、深度バッファと
カメラ内部パラメータだけを使ってワールド座標へ戻す。
シミュレータの物体座標 (``mjData.xpos``) は一切参照しない。
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np


@dataclasses.dataclass
class CameraView:
    """ある瞬間のカメラ画像一式（RGB・深度・外部パラメータ）."""

    rgb: np.ndarray        # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32 [m]
    pos: np.ndarray        # (3,) カメラ位置（ワールド）
    rot: np.ndarray        # (3, 3) カメラ姿勢（列が x_right, y_up, z_back）
    fovy: float            # 垂直画角 [deg]

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]

    @property
    def focal(self) -> float:
        """ピクセル単位の焦点距離（正方ピクセル前提）."""
        h = self.rgb.shape[0]
        return 0.5 * h / np.tan(0.5 * np.deg2rad(self.fovy))

    def unproject(self, u: float, v: float, depth: float | None = None) -> np.ndarray:
        """画素 ``(u, v)`` をワールド座標へ戻す.

        ``depth`` 未指定時はその画素の深度値を使う。MuJoCo の深度バッファは
        カメラ光軸方向の距離 [m] なので、ピンホールモデルをそのまま適用できる。
        """
        h, w = self.shape
        if depth is None:
            depth = float(self.depth[int(round(v)), int(round(u))])
        f = self.focal
        cx, cy = w / 2.0, h / 2.0
        # MuJoCo のカメラ座標系: +x 右 / +y 上 / -z 視線方向
        p_cam = np.array([(u - cx) * depth / f,
                          -(v - cy) * depth / f,
                          -depth])
        return self.pos + self.rot @ p_cam

    def point_cloud(self, box: tuple[float, float, float, float] | None = None,
                    stride: int = 2) -> np.ndarray:
        """画像（または矩形領域）全体を点群 ``(N, 3)`` としてワールド座標に戻す.

        ピクセルごとに ``unproject`` を呼ぶと遅いので、行列演算で一括変換する。
        """
        h, w = self.shape
        if box is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = (int(max(0, box[0])), int(max(0, box[1])),
                              int(min(w, box[2] + 1)), int(min(h, box[3] + 1)))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 3))

        vs = np.arange(y0, y1, stride)
        us = np.arange(x0, x1, stride)
        uu, vv = np.meshgrid(us, vs)
        zz = self.depth[y0:y1:stride, x0:x1:stride].astype(np.float64)

        f, cx, cy = self.focal, w / 2.0, h / 2.0
        p_cam = np.stack([(uu - cx) * zz / f,
                          -(vv - cy) * zz / f,
                          -zz], axis=-1).reshape(-1, 3)
        keep = np.isfinite(p_cam).all(axis=1) & (zz.reshape(-1) < 1e3)
        return self.pos + p_cam[keep] @ self.rot.T

    def median_depth(self, box: tuple[float, float, float, float],
                     shrink: float = 0.4) -> float:
        """バウンディングボックス中心付近の深度の中央値.

        箱の縁は背景を拾うので内側に縮めてから取る。外れ値に強い中央値を使う。
        """
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) * shrink / 2.0, (y1 - y0) * shrink / 2.0
        h, w = self.shape
        a = max(0, int(cy - hh)), min(h, int(cy + hh) + 1)
        b = max(0, int(cx - hw)), min(w, int(cx + hw) + 1)
        patch = self.depth[a[0]:a[1], b[0]:b[1]]
        patch = patch[np.isfinite(patch)]
        if patch.size == 0:
            return float(self.depth[int(cy), int(cx)])
        return float(np.median(patch))


class Camera:
    """MuJoCo のオフスクリーン描画をまとめた薄いラッパ."""

    def __init__(self, model, height: int = 480, width: int = 640):
        self.model = model
        self.height, self.width = height, width
        self._rgb = mujoco.Renderer(model, height, width)
        self._depth = mujoco.Renderer(model, height, width)
        self._depth.enable_depth_rendering()

    def render_rgb(self, data, camera: str) -> np.ndarray:
        self._rgb.update_scene(data, camera=camera)
        return self._rgb.render().copy()

    def capture(self, data, camera: str = "percept") -> CameraView:
        """RGB と深度を同時に取得し、外部パラメータを添えて返す."""
        rgb = self.render_rgb(data, camera)
        self._depth.update_scene(data, camera=camera)
        depth = self._depth.render().copy()

        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        return CameraView(
            rgb=rgb,
            depth=depth,
            pos=data.cam_xpos[cid].copy(),
            rot=data.cam_xmat[cid].reshape(3, 3).copy(),
            fovy=float(self.model.cam_fovy[cid]),
        )

    def close(self) -> None:
        self._rgb.close()
        self._depth.close()
