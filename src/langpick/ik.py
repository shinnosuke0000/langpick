"""減衰最小二乗法（DLS）による逆運動学.

Panda の 7 自由度に対して、手先位置 + 姿勢の 6 次元誤差を解く。
特異点近傍で解が発散しないよう、擬似逆行列ではなく

    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e

を使う。``mujoco.mj_jac`` は任意の点まわりのヤコビアンを返すので、
TCP 用の site を MJCF に足さずに済む。
"""

from __future__ import annotations

import mujoco
import numpy as np

# フランジ面から指の間（TCP）までの距離。Panda の公称値
TCP_OFFSET = 0.1034

# 真下を向いた手先姿勢。列が順に手先の x / y / z 軸（ワールド系）
# z 軸 = (0,0,-1) すなわち「下向きに近づく」、指の開閉方向はワールド y 軸
DOWNWARD = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])


class ArmKinematics:
    """Panda のアーム 7 関節に関する運動学ユーティリティ."""

    def __init__(self, model):
        self.model = model
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.dof = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.joint_names
        ])
        self.qadr = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.joint_names
        ])
        self.lo = np.array([
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
            for n in self.joint_names
        ])
        self.hi = np.array([
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
            for n in self.joint_names
        ])
        self._scratch = mujoco.MjData(model)

    # --- 順運動学 ---------------------------------------------------------
    def tcp(self, data) -> tuple[np.ndarray, np.ndarray]:
        """TCP のワールド座標と回転行列を返す."""
        rot = data.xmat[self.hand_id].reshape(3, 3)
        pos = data.xpos[self.hand_id] + rot @ np.array([0.0, 0.0, TCP_OFFSET])
        return pos.copy(), rot.copy()

    def arm_qpos(self, data) -> np.ndarray:
        return data.qpos[self.qadr].copy()

    # --- 逆運動学 ---------------------------------------------------------
    def solve(self, data, target_pos, target_rot=DOWNWARD, q_init=None,
              iters: int = 120, damping: float = 0.08,
              pos_tol: float = 1e-3, rot_tol: float = 2e-2,
              step_limit: float = 0.25) -> tuple[np.ndarray, float, float]:
        """``target_pos`` / ``target_rot`` を満たす関節角を返す.

        Returns
        -------
        (q, pos_err, rot_err)
            収束しなかった場合も、その時点で最良の関節角を返す（呼び出し側が
            誤差を見て打ち切れるようにする）。
        """
        s = self._scratch
        s.qpos[:] = data.qpos
        s.qvel[:] = 0.0
        q = self.arm_qpos(data) if q_init is None else np.asarray(q_init, float).copy()

        target_pos = np.asarray(target_pos, float)
        quat_des = np.zeros(4)
        mujoco.mju_mat2Quat(quat_des, np.asarray(target_rot, float).flatten())

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        quat_cur, quat_inv, quat_err = np.zeros(4), np.zeros(4), np.zeros(4)
        w = np.zeros(3)
        pos_err = rot_err = np.inf

        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)

            rot = s.xmat[self.hand_id].reshape(3, 3)
            tcp = s.xpos[self.hand_id] + rot @ np.array([0.0, 0.0, TCP_OFFSET])

            e_pos = target_pos - tcp
            mujoco.mju_mat2Quat(quat_cur, rot.flatten())
            mujoco.mju_negQuat(quat_inv, quat_cur)
            mujoco.mju_mulQuat(quat_err, quat_des, quat_inv)
            mujoco.mju_quat2Vel(w, quat_err, 1.0)

            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(w))
            if pos_err < pos_tol and rot_err < rot_tol:
                break

            mujoco.mj_jac(self.model, s, jacp, jacr, tcp, self.hand_id)
            J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])   # (6, 7)
            err = np.concatenate([e_pos, w])

            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            n = np.linalg.norm(dq)
            if n > step_limit:
                dq *= step_limit / n
            q = np.clip(q + dq, self.lo + 1e-4, self.hi - 1e-4)

        return q, pos_err, rot_err
