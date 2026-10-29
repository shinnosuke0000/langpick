"""提出用の mp4 を 1 本作る（編集なしでそのまま出せる形にする）.

使い方::

    python scripts/make_video.py --account YOUR_OMNICAMPUS_ID

GPU があれば数分、CPU でも 10 分程度で終わる。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from langpick import experiment, scene as S  # noqa: E402
from langpick.episode import Env, evaluate_against_truth, run_episode  # noqa: E402
from langpick.perception import ColorBaseline, DEFAULT_MODEL, OwlV2Detector  # noqa: E402
from langpick.video import Banner, Clip, annotate, table_card, title_card  # noqa: E402

TITLE = "LangPick"
SUBTITLE = "言語指示で動く、視覚ベースの pick & place"

DEMO_INSTRUCTIONS = (
    "赤いブロックを紫のトレイに入れて",
    "青い缶を黄色のトレイに入れて",
    "赤い缶を紫のトレイに入れて",
)


def build(account: str, out_dir: pathlib.Path, model_name: str = DEFAULT_MODEL,
          instructions=DEMO_INSTRUCTIONS, fps: int = 30,
          with_experiment: bool = True, detector=None) -> dict:
    """動画を 1 本組み立てる。``detector`` を渡せば読み込み済みのものを使い回す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if detector is None:
        print(f"[1/5] 検出器を読み込み: {model_name}")
        detector = OwlV2Detector(model_name)
    else:
        print("[1/5] 読み込み済みの検出器を使用")
        model_name = getattr(detector, "model_name", model_name)
    env = Env(detector)
    size = env.frame_size
    video_path = out_dir / f"PAI最終課題_{account}.mp4"
    clip = Clip(video_path, fps=fps)

    # --- オープニング ---
    clip.hold(title_card(size, [
        (TITLE, 62), (SUBTITLE, 26), ("", 10),
        ("Physical AI 応用編1  最終課題", 20),
        (f"omnicampus: {account}", 18),
    ]), 3.2)

    clip.hold(title_card(size, [
        ("やっていること", 30), ("", 6),
        ("① 指示文を対象と置き先に分解", 23),
        ("② OWLv2 が語を画素に接地（オープン語彙検出）", 23),
        ("③ RGB-D から 3D 把持位置を推定", 23),
        ("④ IK で把持・搬送し、実行後の画像で自己検証", 23),
        ("", 8),
        ("シミュレータ内の物体座標は一切参照していない", 20),
    ]), 5.0)

    # --- シーン紹介 ---
    env.reset()
    view = env.look()
    clip.hold(annotate(view.rgb, Banner(
        instruction_ja="シーン: 卓上に 4 物体 + 2 つの置き先",
        instruction_en="Scene: a red cube and a red can share one colour",
        phase="SETUP",
        note="赤いブロックと赤い缶は同じ色・違う形 — 色だけを見る手法では区別できない"),
        scale=view.rgb.shape[1] / 640.0), 4.2)

    results, records = [], []
    for i, text in enumerate(instructions, 1):
        print(f"[2/5] エピソード {i}/{len(instructions)}: {text}")
        r = run_episode(env, text, clip)
        ev = evaluate_against_truth(r)
        results.append(r)
        records.append({
            "instruction": text,
            "target_query": r.instruction.target.query,
            "detected": None if r.target_det is None else {
                "score": r.target_det.score, "box": list(r.target_det.box)},
            "grasp": None if r.grasp is None else list(map(float, r.grasp.position)),
            "verified_by_vlm": r.verified,
            "verify_message": r.verify_msg,
            "motion": r.motion,
            "truth_check": ev,
            "seconds": r.seconds,
        })
        print("      " + r.summary())

    # --- 対照実験 ---
    table_md = ""
    if with_experiment:
        print("[3/5] 対照実験: OWLv2 vs 色ベースライン")
        trials = experiment.run(env, {"OWLv2": detector, "色閾値": ColorBaseline()})
        table_md = experiment.to_table(trials)
        headers = ["指示の対象", "OWLv2（VLM）", "色閾値ベースライン"]
        rows, marks = [], []
        for text in dict.fromkeys(t.instruction for t in trials):
            row, mark = [text.split("を")[0]], [None]
            for dname in ("OWLv2", "色閾値"):
                t = next(x for x in trials
                         if x.instruction == text and x.detector == dname)
                row.append("検出なし" if t.identified is None
                           else f"{t.identified}" + ("" if t.correct else "（誤り）"))
                mark.append(t.correct)
            rows.append(row)
            marks.append(mark)
        acc = {d: sum(t.correct for t in trials if t.detector == d)
               for d in ("OWLv2", "色閾値")}
        n = len(trials) // 2
        clip.hold(table_card(size, "対照実験: VLM を使う必然性", headers, rows, marks,
                             footer=(f"対象の同定: OWLv2 {acc['OWLv2']}/{n}  /  "
                                     f"色閾値 {acc['色閾値']}/{n}\n"
                                     "色閾値は『赤いブロック』と『赤い缶』に"
                                     "同じ領域を返す — 形を問う指示に原理的に答えられない")),
                  7.0)

    # --- クロージング ---
    n_ok = sum(r.verified for r in results)
    truth_ok = sum(rec["truth_check"].get("truly_in_tray", False) for rec in records)
    clip.hold(title_card(size, [
        ("結果", 34), ("", 6),
        (f"指示 {len(results)} 件中 {n_ok} 件を自己検証で成功と判定", 24),
        (f"（真値による確認: {truth_ok}/{len(results)} 件がトレイ内）", 20),
        ("", 10),
        ("コードは 1 行も変えていない。変えたのは指示文だけ。", 22),
    ]), 5.0)

    path = clip.close()
    print(f"[4/5] 書き出し完了 … {clip.duration:.1f} 秒 / {clip.n_frames} フレーム")

    summary = {
        "account": account,
        "model": model_name,
        "episodes": records,
        "video_seconds": clip.duration,
        "comparison_table_markdown": table_md,
        "build_seconds": time.time() - t0,
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5/5] 完了: {path}  ({time.time() - t0:.0f} 秒)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--account", default="anonymous",
                   help="omnicampus のアカウント名（動画タイトルに入れる）")
    p.add_argument("--out", default="out")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-experiment", action="store_true")
    a = p.parse_args()
    build(a.account, pathlib.Path(a.out), a.model,
          fps=a.fps, with_experiment=not a.no_experiment)


if __name__ == "__main__":
    main()
