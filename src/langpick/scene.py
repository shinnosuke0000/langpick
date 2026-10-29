"""LangPick のシミュレーション環境（MuJoCo）を組み立てるモジュール.

設計方針
--------
* ロボットは mujoco_menagerie の Franka Emika Panda をそのまま使う（再発明しない）。
* 卓上のオブジェクトとトレイ、カメラだけを MJCF 文字列として生成し、
  ``<include file="panda.xml"/>`` で Panda を取り込む。
* MuJoCo の include / meshdir は「メインの MJCF が置かれたディレクトリ」を基準に
  解決されるため、生成した scene XML は menagerie の ``franka_emika_panda/``
  直下に書き出してからロードする。

``OBJECTS`` に書かれたワールド座標は **評価と初期配置にのみ** 使う。
制御側（知覚→IK→動作）は一切参照しない。これが本課題の肝である。
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import mujoco
import numpy as np

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
PANDA_DIR_NAME = "franka_emika_panda"
SCENE_FILE_NAME = "langpick_scene.xml"

# 卓上の作業領域（ワールド座標 [m]）
WORKSPACE_CENTER = (0.50, 0.0)
MAT_HALF = (0.26, 0.42)
MAT_TOP_Z = 0.004


@dataclasses.dataclass(frozen=True)
class SceneObject:
    """掴む対象（または妨害物）の定義."""

    name: str
    kind: str  # "box" | "cylinder" | "sphere"
    size: tuple[float, ...]  # MuJoCo の geom size
    rgba: tuple[float, float, float, float]
    xy: tuple[float, float]
    label_ja: str
    graspable: bool = True

    @property
    def height(self) -> float:
        """接地時の中心高さ（= 高さの半分）."""
        if self.kind == "box":
            return self.size[2]
        if self.kind == "cylinder":
            return self.size[1]
        return self.size[0]  # sphere

    @property
    def spawn_z(self) -> float:
        return MAT_TOP_Z + self.height + 0.001


@dataclasses.dataclass(frozen=True)
class SceneTray:
    """置き先のトレイ."""

    name: str
    rgba: tuple[float, float, float, float]
    xy: tuple[float, float]
    label_ja: str
    half: float = 0.085
    wall_h: float = 0.022


# --- 掴む対象 ----------------------------------------------------------------
# red_cube と red_can は「同じ色・違う形」。色ベースラインでは区別できず、
# オープン語彙検出（VLM）でのみ区別できる。この対比が実験の中心。
OBJECTS: tuple[SceneObject, ...] = (
    SceneObject(
        name="red_cube",
        kind="box",
        size=(0.020, 0.020, 0.030),
        rgba=(0.85, 0.11, 0.13, 1.0),
        xy=(0.40, -0.11),
        label_ja="赤いブロック",
    ),
    SceneObject(
        name="red_can",
        kind="cylinder",
        size=(0.024, 0.034),
        rgba=(0.88, 0.16, 0.18, 1.0),
        xy=(0.54, 0.09),
        label_ja="赤い缶",
    ),
    SceneObject(
        name="green_cube",
        kind="box",
        size=(0.020, 0.020, 0.030),
        rgba=(0.13, 0.62, 0.22, 1.0),
        xy=(0.58, -0.13),
        label_ja="緑のブロック",
    ),
    SceneObject(
        name="blue_can",
        kind="cylinder",
        size=(0.024, 0.034),
        rgba=(0.12, 0.33, 0.78, 1.0),
        xy=(0.42, 0.13),
        label_ja="青い缶",
    ),
)

TRAYS: tuple[SceneTray, ...] = (
    SceneTray(
        name="purple_tray",
        rgba=(0.45, 0.22, 0.62, 1.0),
        xy=(0.47, 0.32),
        label_ja="紫のトレイ",
    ),
    SceneTray(
        name="yellow_tray",
        rgba=(0.95, 0.78, 0.08, 1.0),
        xy=(0.47, -0.32),
        label_ja="黄色のトレイ",
    ),
)


def ensure_menagerie(assets_dir: str | pathlib.Path) -> pathlib.Path:
    """Panda のモデルを取得して ``franka_emika_panda`` ディレクトリを返す.

    menagerie 全体は 2GB 近くあるので、sparse checkout で Panda だけを取る
    （実測 44MB・数秒）。すでにあれば何もしない。
    """
    assets_dir = pathlib.Path(assets_dir)
    repo = assets_dir / "_menagerie"
    panda_dir = repo / PANDA_DIR_NAME
    if (panda_dir / "panda.xml").exists():
        return panda_dir

    assets_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MENAGERIE_URL, str(repo)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", PANDA_DIR_NAME],
        cwd=str(repo), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not (panda_dir / "panda.xml").exists():
        sys.exit("Panda のモデル取得に失敗しました")
    return panda_dir


def _tray_xml(tray: SceneTray) -> str:
    h, w = tray.half, tray.wall_h
    t = 0.006  # 板厚
    rgba = " ".join(f"{c:g}" for c in tray.rgba)
    walls = []
    for dx, dy, sx, sy in (
        (h, 0.0, t, h), (-h, 0.0, t, h),
        (0.0, h, h, t), (0.0, -h, h, t),
    ):
        walls.append(
            f'      <geom type="box" size="{sx:g} {sy:g} {w:g}" '
            f'pos="{dx:g} {dy:g} {w:g}" rgba="{rgba}" '
            f'contype="1" conaffinity="1"/>'
        )
    return (
        f'    <body name="{tray.name}" pos="{tray.xy[0]:g} {tray.xy[1]:g} {MAT_TOP_Z:g}">\n'
        f'      <geom type="box" size="{h:g} {h:g} {t:g}" pos="0 0 {t:g}" rgba="{rgba}"/>\n'
        + "\n".join(walls)
        + "\n    </body>\n"
    )


PHYSICS = ('condim="4" friction="1.5 0.03 0.003" '
           'solref="0.008 1" solimp="0.95 0.99 0.001"')


def _object_xml(obj: SceneObject) -> str:
    """物体 1 個分の MJCF.

    円柱には **箱の衝突プロキシ** を被せている。
    平行平板グリッパで滑らかな円柱を掴むのは MuJoCo では安定せず、
    指のパッドが円柱に線接触するため、持ち上げ中に少しずつ横へ滑り出てしまう。
    摩擦係数を上げると今度は接触が軟らかくなりすぎて指が物体にめり込み、
    「掴めている」ように見えて実際には非物理的な状態になった。

    そこで、見た目は円柱のまま、衝突形状だけを外接する箱に置き換えた
    （ロボティクスのシミュレーションでよく使われる衝突プロキシ）。
    箱の半幅は円柱の半径と同じなので、指が接触する位置は見た目と一致し、
    はみ出す角は 45 度方向＝グリッパから見えない向きにある。

    知覚側はレンダリング結果しか見ないので、この置き換えは検出には影響しない。
    """
    rgba = " ".join(f"{c:g}" for c in obj.rgba)

    if obj.kind == "cylinder":
        r, half_h = obj.size[0], obj.size[1]
        return (
            f'    <body name="{obj.name}" pos="{obj.xy[0]:g} {obj.xy[1]:g} {obj.spawn_z:g}">\n'
            f"      <freejoint/>\n"
            f'      <geom name="{obj.name}_visual" type="cylinder" size="{r:g} {half_h:g}" '
            f'rgba="{rgba}" contype="0" conaffinity="0" group="2" density="0"/>\n'
            f'      <geom name="{obj.name}_geom" type="box" size="{r:g} {r:g} {half_h:g}" '
            f'rgba="{rgba}" group="3" {PHYSICS} density="400"/>\n'
            f"    </body>\n"
        )

    size = " ".join(f"{s:g}" for s in obj.size)
    return (
        f'    <body name="{obj.name}" pos="{obj.xy[0]:g} {obj.xy[1]:g} {obj.spawn_z:g}">\n'
        f"      <freejoint/>\n"
        f'      <geom name="{obj.name}_geom" type="{obj.kind}" size="{size}" '
        f'rgba="{rgba}" {PHYSICS} density="400"/>\n'
        f"    </body>\n"
    )


def build_scene_xml(objects=OBJECTS, trays=TRAYS, offwidth=1280, offheight=720) -> str:
    """シーン全体の MJCF を文字列で組み立てる."""
    mx, my = WORKSPACE_CENTER
    bodies = "".join(_object_xml(o) for o in objects)
    tray_bodies = "".join(_tray_xml(t) for t in trays)
    return f"""<mujoco model="langpick scene">
  <include file="panda.xml"/>

  <statistic center="{mx:g} 0 0.2" extent="1.1"/>

  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}" azimuth="140" elevation="-25"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.38 0.38 0.38" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="12"/>
    <rgba haze="0.16 0.2 0.26 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.26 0.32 0.42" rgb2="0.05 0.07 0.1"
             width="512" height="3072"/>
    <texture type="2d" name="floortex" builtin="checker" mark="edge"
             rgb1="0.22 0.24 0.28" rgb2="0.16 0.18 0.21" markrgb="0.4 0.42 0.46"
             width="300" height="300"/>
    <material name="floormat" texture="floortex" texuniform="true" texrepeat="6 6" reflectance="0.08"/>
    <material name="matmat" rgba="0.80 0.78 0.74 1" specular="0.15" shininess="0.1"/>
  </asset>

  <worldbody>
    <light pos="0.5 -0.6 1.6" dir="-0.1 0.35 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0.2 0.8 1.4" dir="0.2 -0.5 -1" directional="true" diffuse="0.28 0.28 0.3"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floormat"/>

    <!-- 作業台（天板のみ）。この上に物体とトレイが載る -->
    <geom name="workmat" type="box" material="matmat"
          size="{MAT_HALF[0]:g} {MAT_HALF[1]:g} {MAT_TOP_Z:g}"
          pos="{mx:g} {my:g} {MAT_TOP_Z:g}"/>

    <!-- カメラの注視点にだけ使う空 body（geom を持たないので描画されない） -->
    <body name="look_at" pos="{mx:g} {my:g} 0.06"/>
    <body name="look_at_wide" pos="{mx * 0.55:g} {my:g} 0.26"/>

    <!-- 知覚用カメラ。ここから撮った RGB + 深度だけで把持位置を決める -->
    <camera name="percept" pos="{mx:g} -0.78 0.74" mode="targetbody" target="look_at" fovy="52"/>
    <!-- 動画用カメラ（ロボット全体が入る斜め視点） -->
    <camera name="cine" pos="1.45 -1.15 1.05" mode="targetbody" target="look_at_wide" fovy="45"/>

{tray_bodies}{bodies}  </worldbody>
</mujoco>
"""


def make_model(assets_dir: str | pathlib.Path = "assets", **kwargs):
    """シーンを書き出して ``MjModel`` / ``MjData`` を返す."""
    panda_dir = ensure_menagerie(assets_dir)
    scene_path = panda_dir / SCENE_FILE_NAME
    scene_path.write_text(build_scene_xml(**kwargs), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def reset(model, data) -> None:
    """``home`` キーフレームへ戻し、物体を初期配置に置き直す."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)

    for obj in OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.name)
        adr = model.jnt_qposadr[model.body_jntadr[bid]]
        data.qpos[adr:adr + 3] = (obj.xy[0], obj.xy[1], obj.spawn_z)
        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)


def body_pos(model, data, name: str) -> np.ndarray:
    """評価専用: ボディのワールド座標（制御では使わない）."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()
