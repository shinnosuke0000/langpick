"""指示 1 件を最初から最後まで実行する.

    指示文 → 言語解析 → VLM で接地 → RGB-D から 3D 把持位置 → IK で実行
          → 実行後の画像で自己検証

この関数は ``mjData`` の物体座標を読まない。唯一の例外は ``ground_truth``
（評価のためだけに最後に記録する値）で、制御には一切使っていない。
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import scene as S
from .camera import Camera
from .controller import PickPlace
from .grasp import GraspPose, estimate_grasp, estimate_place_point
from .instruction import ParsedInstruction, parse
from .perception import Detection, verify_placement
from .video import Banner, Clip, SimRecorder, annotate, title_card, top_per_query


@dataclasses.dataclass
class EpisodeResult:
    instruction: ParsedInstruction
    target_det: Detection | None
    dest_det: Detection | None
    grasp: GraspPose | None
    place_point: np.ndarray | None
    motion: dict | None
    verified: bool
    verify_msg: str
    verify_dets: list
    percept_rgb: np.ndarray
    after_rgb: np.ndarray | None
    ground_truth: dict          # 実行後の真値（評価専用）
    ground_truth_before: dict   # 実行前の真値（把持位置の誤差評価に使う）
    seconds: float

    @property
    def ok(self) -> bool:
        return self.verified

    def summary(self) -> str:
        t = self.instruction
        head = f"[{'成功' if self.verified else '失敗'}] {t.text_ja}"
        if self.target_det is None:
            return head + "  — 対象を検出できなかった"
        g = "" if self.grasp is None else f" grasp={np.round(self.grasp.position, 3)}"
        return (head + f"  検出 {self.target_det.score:.2f}" + g
                + f"  ({self.verify_msg})")


class Env:
    """モデル・カメラ・検出器をひとまとめにした実行環境.

    知覚用と動画用でカメラの *画角* は違うが **解像度は同じ** にしてある。
    動画の全フレームを同じサイズに揃えないと mp4 にまとめられないため。
    """

    def __init__(self, detector, assets_dir: str = "assets",
                 width: int = 960, height: int = 540):
        self.model, self.data = S.make_model(assets_dir,
                                             offwidth=width, offheight=height)
        self.cam = Camera(self.model, height, width)
        self.detector = detector

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(幅, 高さ)``。title_card などに渡す."""
        return self.cam.width, self.cam.height

    def reset(self) -> None:
        S.reset(self.model, self.data)

    def look(self) -> "object":
        return self.cam.capture(self.data, "percept")

    def ground_truth(self, names: list[str]) -> dict:
        """評価専用。制御には使わない物体の真の位置."""
        return {n: S.body_pos(self.model, self.data, n).tolist() for n in names}


def run_episode(env: Env, text: str, clip: Clip | None = None,
                threshold: float = 0.08, speed_note: str = "") -> EpisodeResult:
    """指示 1 件を実行し、``clip`` が渡されていれば動画も作る."""
    t_start = time.time()
    ins = parse(text)
    banner = Banner(instruction_ja=ins.text_ja, instruction_en=ins.text_en)

    if clip is not None:
        clip.hold(title_card(env.frame_size, [
            ("指示", 22), (ins.text_ja, 40), (ins.text_en, 20),
        ]), 1.6)

    # --- 1. 知覚: 指示文の語を画素に接地する -----------------------------
    view = env.look()
    dets = env.detector.detect(view.rgb, [ins.target.query, ins.destination.query],
                               threshold)
    target = next((d for d in dets if d.query == ins.target.query), None)
    dest = next((d for d in dets if d.query == ins.destination.query), None)

    scale = view.rgb.shape[1] / 640.0
    if clip is not None:
        note = (f"OWLv2: 『{ins.target.query}』 "
                + (f"score={target.score:.2f}" if target else "検出なし"))
        clip.hold(annotate(view.rgb, Banner(ins.text_ja, ins.text_en,
                                            phase="PERCEPTION", note=note),
                           top_per_query(dets), scale=scale), 2.4)

    gt = env.ground_truth([o.name for o in S.OBJECTS] + [t.name for t in S.TRAYS])

    if target is None or dest is None:
        missing = "対象" if target is None else "置き先"
        if clip is not None:
            clip.hold(annotate(view.rgb, Banner(ins.text_ja, ins.text_en,
                                                phase="ABORT",
                                                note=f"{missing}を検出できず中止",
                                                verdict="FAILURE"),
                               dets, scale=scale), 2.0)
        return EpisodeResult(ins, target, dest, None, None, None, False,
                             f"{missing}を検出できなかった", dets, view.rgb, None,
                             gt, gt, time.time() - t_start)

    # --- 2. 2D の枠を 3D の把持位置へ ------------------------------------
    grasp = estimate_grasp(view, target.box)
    place = estimate_place_point(view, dest.box)

    if clip is not None:
        note = (f"RGB-D から推定: 把持点 ({grasp.position[0]:.3f}, "
                f"{grasp.position[1]:.3f}, {grasp.position[2]:.3f}) m  /  "
                f"点群 {grasp.n_points} 点")
        pc = view.rot.T @ (grasp.position - view.pos)
        f, (vh, vw) = view.focal, view.shape
        marker = (pc[0] * f / -pc[2] + vw / 2.0, -pc[1] * f / -pc[2] + vh / 2.0)
        clip.hold(annotate(view.rgb, Banner(ins.text_ja, ins.text_en,
                                            phase="3D ESTIMATE", note=note),
                           [target, dest], scale=scale, marker=marker), 2.4)

    # --- 3. 実行 ----------------------------------------------------------
    recorder = None
    if clip is not None:
        recorder = SimRecorder(env.cam, clip, "cine")
        recorder.banner = Banner(ins.text_ja, ins.text_en, note=speed_note)

    ctrl = PickPlace(env.model, env.data, recorder)
    motion = ctrl.execute(grasp.position, place)

    # --- 4. 自己検証: 実行後の画像だけで成否を判定 ------------------------
    after = env.look()
    ok, msg, vdets = verify_placement(env.detector, after.rgb,
                                      ins.target.query, ins.destination.query,
                                      threshold=max(0.06, threshold - 0.02))

    if clip is not None:
        clip.hold(annotate(after.rgb, Banner(ins.text_ja, ins.text_en,
                                             phase="VERIFY", note=msg,
                                             verdict="SUCCESS" if ok else "FAILURE"),
                           top_per_query(vdets), scale=scale), 2.6)

    return EpisodeResult(ins, target, dest, grasp, place, motion, ok, msg,
                         vdets, view.rgb, after.rgb,
                         env.ground_truth([o.name for o in S.OBJECTS]
                                          + [t.name for t in S.TRAYS]),
                         gt, time.time() - t_start)


def evaluate_against_truth(result: EpisodeResult) -> dict:
    """真値を使った後追い評価（制御には使っていない指標）."""
    ins = result.instruction
    name_of = {
        ("red", "cube"): "red_cube", ("red", "can"): "red_can",
        ("green", "cube"): "green_cube", ("blue", "can"): "blue_can",
        ("purple", "box"): "purple_tray", ("yellow", "box"): "yellow_tray",
    }
    tname = name_of.get((ins.target.color, ins.target.noun))
    dname = name_of.get((ins.destination.color, ins.destination.noun))
    out: dict = {"target": tname, "destination": dname}
    if tname is None or dname is None:
        return out

    tp = np.array(result.ground_truth[tname])
    dp = np.array(result.ground_truth[dname])
    tray_half = next(t.half for t in S.TRAYS if t.name == dname)
    out["distance_to_tray_center"] = float(np.linalg.norm(tp[:2] - dp[:2]))
    out["truly_in_tray"] = bool(out["distance_to_tray_center"] < tray_half)
    if result.grasp is not None:
        # 把持位置の誤差は「掴みにいった時点」の真値と比べる
        out["grasp_xy_error_mm"] = float(
            np.linalg.norm(result.grasp.position[:2] - np.array(
                result.ground_truth_before[tname])[:2]) * 1000)
    return out
