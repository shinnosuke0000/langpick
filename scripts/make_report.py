"""提出用レポートを組み立てる.

``out/results.json`` の実測値をテンプレートに差し込み、
Markdown と、A4 で印刷すると整う HTML を書き出す。
HTML をブラウザで開き Ctrl+P →「PDF に保存」で提出用 PDF になる。

使い方::

    python scripts/make_report.py --account YOUR_ID \
        --repo https://github.com/you/langpick --video https://youtu.be/xxxx
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sys
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Yu Gothic", "Hiragino Kaku Gothic ProN", "Noto Sans JP",
               "Meiryo", sans-serif;
  font-size: 10.2pt; line-height: 1.75; color: #16181d;
  max-width: 182mm; margin: 0 auto; padding: 8mm 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 19pt; margin: 0 0 4pt; letter-spacing: -0.01em; }
h2 { font-size: 13.5pt; margin: 20pt 0 7pt; padding-bottom: 4pt;
     border-bottom: 1.5px solid #d5d9e0; }
h3 { font-size: 11.4pt; margin: 14pt 0 5pt; color: #2b3a52; }
p, li { margin: 5pt 0; }
hr { border: 0; border-top: 1px solid #dde1e8; margin: 12pt 0; }
code { font-family: "Consolas", "SF Mono", monospace; font-size: 9pt;
       background: #f1f3f7; padding: 1px 4px; border-radius: 3px; }
pre { background: #f7f8fb; border: 1px solid #e2e6ee; border-left: 3px solid #5b7cfa;
      border-radius: 4px; padding: 8pt 10pt; overflow-x: auto;
      font-size: 8.6pt; line-height: 1.5; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 9.2pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #dde1e8; padding: 4.5pt 7pt; text-align: left;
         vertical-align: top; }
th { background: #eef1f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfd; }
blockquote { margin: 9pt 0; padding: 7pt 12pt; background: #f4f7ff;
             border-left: 3px solid #5b7cfa; page-break-inside: avoid; }
blockquote p { margin: 0; }
strong { color: #0f1115; }
h2, h3 { page-break-after: avoid; }
.meta { color: #5a6376; font-size: 9pt; margin-bottom: 14pt; }
"""


def md_to_html(md: str) -> str:
    """提出レポートに必要な範囲だけの小さな Markdown 変換."""
    out, lines = [], md.split("\n")
    i, in_code, in_table, in_list = 0, False, False, None

    def inline(s: str) -> str:
        s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    # Markdown では改行しただけの行は同じ段落。1 行 1 段落にすると
    # 行間が空きすぎて読みにくいので、連続する平文はまとめる。
    para: list[str] = []

    def join_lines(parts: list[str]) -> str:
        """折り返した行をつなぐ. 日本語どうしの境目には空白を入れない."""
        text = parts[0]
        for nxt in parts[1:]:
            sep = "" if (ord(text[-1]) > 0x2E80 and ord(nxt[0]) > 0x2E80) else " "
            text += sep + nxt
        return text

    def flush_para():
        if para:
            out.append(f"<p>{inline(join_lines(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if (not line.strip() or line.startswith(("```", "|", "#", "> ", "- "))
                or re.match(r"^\d+\.\s+", line) or line.strip() == "---"):
            flush_para()
        # コードブロックや表が始まるときは、開きっぱなしのリストを先に閉じる
        if not in_code and (line.startswith("```") or line.startswith("|")):
            close_list()
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre><code>")
            in_code = not in_code
            i += 1
            continue
        if in_code:
            out.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-: |]+\|$", lines[i + 1]):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{inline(c)}</th>" for c in cells)
                       + "</tr></thead><tbody>")
            i += 2
            in_table = True
            continue
        if in_table:
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
                continue
            out.append("</tbody></table>")
            in_table = False

        ordered = re.match(r"^(\d+)\.\s+(.*)", line)
        bullet = line.startswith("- ")
        want = "ol" if ordered else ("ul" if bullet else None)
        if in_list and in_list != want:
            close_list()
        if want and in_list != want:
            out.append(f"<{want}>")
            in_list = want

        if m := re.match(r"^(#{1,4})\s+(.*)", line):
            lv = len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
        elif line.startswith("> "):
            out.append(f"<blockquote><p>{inline(line[2:])}</p></blockquote>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif ordered:
            out.append(f"<li>{inline(ordered.group(2))}</li>")
        elif bullet:
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip():
            para.append(line.strip())
        i += 1

    flush_para()
    close_list()
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


def episode_table(results: dict) -> str:
    rows = ["| 指示 | 対象の検出スコア | 把持位置の誤差 | 自己検証 | 真値での確認 |",
            "|---|---|---|---|---|"]
    for ep in results["episodes"]:
        det = ep.get("detected")
        tc = ep.get("truth_check", {})
        score = f"{det['score']:.2f}" if det else "検出なし"
        err = (f"{tc['grasp_xy_error_mm']:.1f} mm"
               if "grasp_xy_error_mm" in tc else "—")
        verdict = "成功" if ep["verified_by_vlm"] else "失敗"
        truth = ("トレイ内 ✓" if tc.get("truly_in_tray")
                 else f"トレイ外（中心から {tc.get('distance_to_tray_center', 0) * 100:.1f} cm）")
        rows.append(f"| {ep['instruction']} | {score} | {err} | {verdict} | {truth} |")

    n = len(results["episodes"])
    ok = sum(e["verified_by_vlm"] for e in results["episodes"])
    truth_ok = sum(e.get("truth_check", {}).get("truly_in_tray", False)
                   for e in results["episodes"])
    rows.append("")
    rows.append(f"指示 {n} 件のうち、自己検証で成功と判定したのが {ok} 件、"
                f"真値で確認してトレイ内にあったのが {truth_ok} 件。")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--account", required=True)
    p.add_argument("--repo", default="（提出時に記入）")
    p.add_argument("--video", default="（提出時に記入）")
    p.add_argument("--results", default="out/results.json")
    p.add_argument("--template", default="report/report_template.md")
    p.add_argument("--out-dir", default="report")
    a = p.parse_args()

    results = json.loads(pathlib.Path(a.results).read_text(encoding="utf-8"))

    print("アブレーションを実行中 …")
    import ablation_grasp
    buf = io.StringIO()
    with redirect_stdout(buf):
        ablation_grasp.main()
    grasp_table = buf.getvalue().strip()

    md = pathlib.Path(a.template).read_text(encoding="utf-8")
    md = (md.replace("{{ACCOUNT}}", a.account)
            .replace("{{REPO_URL}}", a.repo)
            .replace("{{VIDEO_URL}}", a.video)
            .replace("{{GRASP_TABLE}}", grasp_table)
            .replace("{{COMPARISON_TABLE}}",
                     results.get("comparison_table_markdown") or "（未取得）")
            .replace("{{EPISODE_TABLE}}", episode_table(results)))

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"説明資料_PAI最終課題_{a.account}"
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")

    html = (f"<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
            f"<title>{stem}</title><style>{CSS}</style></head><body>"
            f"{md_to_html(md)}</body></html>")
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"書き出し: {out_dir / f'{stem}.md'}")
    print(f"書き出し: {html_path}")
    print("\nHTML をブラウザで開き、Ctrl+P →「送信先: PDF に保存」で提出用 PDF になります。")
    print(f"ファイル名は {stem}.pdf にしてください。")


if __name__ == "__main__":
    main()
