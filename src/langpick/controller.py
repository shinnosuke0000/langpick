"""pick & place の動作生成.

Panda のアクチュエータは位置サーボ（``ctrl`` = 目標関節角）なので、
IK で求めた関節角を ``ctrl`` に入れて積分するだけで追従する。
直線的に降ろしたい区間があるので、関節空間ではなく **手先の直線軌道** を
細かく刻み、各点で IK を解く。
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from .ik import DOWNWARD, ArmKinematics

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def _smoothstep(t: float) -> float:
    """0→1 を滑らかに補間（始点と終点で速度 0）."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclasses.dataclass
class MotionLog:
    """実行時の追従誤差。レポートの定量評価に使う."""

    max_pos_err: float = 0.0
    max_ik_err: float = 0.0
    ik_failures: int = 0


class PickPlace:
    """手先の直線運動とグリッパ開閉を並べて pick & place を作る."""

    def __init__(self, model, data, recorder=None):
        self.model, self.data = model, data
        self.kin = ArmKinematics(model)
        self.recorder = recorder
        self.log = MotionLog()
        self._q_seed = self.kin.arm_qpos(data)

    # --- 低レベル -------------------------------------------------------
    def _tick(self, phase: str | None = None) -> None:
        if phase is not None and self.recorder is not None:
            self.recorder.banner.phase = phase
        mujoco.mj_step(self.model, self.data)
        if self.recorder is not None:
            self.recorder.maybe_capture(self.data)

    def settle(self, seconds: float = 0.4, phase: str | None = None) -> None:
        """目標を変えずに時間を進める（接触の落ち着き待ち）."""
        n = int(seconds / self.model.opt.timestep)
        for _ in range(n):
            self._tick(phase)

    # --- 手先の直線移動 -------------------------------------------------
    def move_to(self, goal: np.ndarray, duration: float = 1.5,
                rot: np.ndarray = DOWNWARD, phase: str = "MOVE",
                waypoints: int = 40) -> None:
        """現在の TCP から ``goal`` まで直線で動かす."""
        start, _ = self.kin.tcp(self.data)
        goal = np.asarray(goal, float)
        steps_total = max(1, int(duration / self.model.opt.timestep))
        per_seg = max(1, steps_total // waypoints)

        for k in range(1, waypoints + 1):
            a = _smoothstep(k / waypoints)
            target = start + (goal - start) * a
            q, pe, re = self.kin.solve(self.data, target, rot, q_init=self._q_seed)
            self._q_seed = q
            self.log.max_ik_err = max(self.log.max_ik_err, pe)
            if pe > 5e-3:
                self.log.ik_failures += 1
            self.data.ctrl[:7] = q
            for _ in range(per_seg):
                self._tick(phase)

        tcp, _ = self.kin.tcp(self.data)
        self.log.max_pos_err = max(self.log.max_pos_err,
                                   float(np.linalg.norm(tcp - goal)))

    def move_home(self, duration: float = 1.6, phase: str = "HOME") -> None:
        """キーフレーム ``home`` の関節角へ戻す."""
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        q_home = self.model.key_ctrl[key][:7].copy()
        q0 = self.data.ctrl[:7].copy()
        n = max(1, int(duration / self.model.opt.timestep))
        for i in range(n):
            self.data.ctrl[:7] = q0 + (q_home - q0) * _smoothstep((i + 1) / n)
            self._tick(phase)
        self._q_seed = self.kin.arm_qpos(self.data)

    # --- グリッパ -------------------------------------------------------
    def gripper(self, value: float, duration: float = 0.7,
                phase: str = "GRIPPER") -> None:
        v0 = float(self.data.ctrl[7])
        n = max(1, int(duration / self.model.opt.timestep))
        for i in range(n):
            self.data.ctrl[7] = v0 + (value - v0) * _smoothstep((i + 1) / n)
            self._tick(phase)

    def gripper_width(self) -> float:
        """現在の指の開き幅 [m]（把持できているかの判定に使う）."""
        j1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        j2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        return float(self.data.qpos[self.model.jnt_qposadr[j1]]
                     + self.data.qpos[self.model.jnt_qposadr[j2]])

    # --- 一連の動作 -----------------------------------------------------
    def execute(self, grasp_pos: np.ndarray, place_pos: np.ndarray,
                approach: float = 0.13, lift: float = 0.20,
                place_clearance: float = 0.07) -> dict:
        """掴んで運んで置く。戻り値は各段階の記録."""
        grasp_pos = np.asarray(grasp_pos, float)
        place_pos = np.asarray(place_pos, float)
        up = np.array([0.0, 0.0, 1.0])

        self.gripper(GRIPPER_OPEN, 0.3, "OPEN")
        self.move_to(grasp_pos + up * approach, 1.6, phase="APPROACH")
        self.move_to(grasp_pos, 1.1, phase="DESCEND")
        self.gripper(GRIPPER_CLOSED, 0.8, "GRASP")
        self.settle(0.25, "GRASP")
        width_after_close = self.gripper_width()

        # 持ち上げは「指令した把持位置」ではなく **実際の TCP** から真上に行う。
        # サーボの残差（数 mm）を目標にすると、その分だけ横に押す動きが混ざり、
        # 円柱がせん断されて指から飛び出す（立方体は面接触なので耐えてしまい、
        # 円柱でだけ再現する厄介な不具合だった）。
        tcp_now, _ = self.kin.tcp(self.data)
        self.move_to(tcp_now + up * lift, 1.1, phase="LIFT")
        width_after_lift = self.gripper_width()

        above_tray = place_pos + up * (place_clearance + 0.10)
        self.move_to(above_tray, 2.0, phase="TRANSFER")
        self.move_to(place_pos + up * place_clearance, 0.9, phase="PLACE")
        self.gripper(GRIPPER_OPEN, 0.6, "RELEASE")
        self.settle(0.35, "RELEASE")
        self.move_to(place_pos + up * (place_clearance + 0.14), 0.9, phase="RETREAT")
        self.move_home(1.5)

        return {
            "grasp_pos": grasp_pos.tolist(),
            "place_pos": place_pos.tolist(),
            "width_after_close": width_after_close,
            "width_after_lift": width_after_lift,
            "held": width_after_lift > 0.012,
            "max_ik_pos_err": self.log.max_ik_err,
            "ik_failures": self.log.ik_failures,
        }
