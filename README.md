# LangPick — 言語指示で動く、視覚ベースの pick & place

Physical AI 応用編1 最終課題。
**指示文を書き換えるだけ**で、ロボットが掴む物も置き先も変わる。コードは 1 行も変えない。

```
「赤いブロックを紫のトレイに入れて」
   │
   ├─ ① 指示解析   instruction.py  対象と置き先のスロットに分解
   ├─ ② 接地       perception.py   OWLv2（オープン語彙物体検出）が語を画素に結びつける
   ├─ ③ 3D 化      camera.py       深度と内部パラメータで逆投影
   │               grasp.py        天面の点群から把持位置を推定
   ├─ ④ 動作       ik.py           減衰最小二乗 IK
   │               controller.py   approach → grasp → lift → transfer → release
   └─ ⑤ 自己検証   perception.py   実行後の画像を再び VLM に通して成否を判定
```

## この実装の立ち位置

**シミュレータ内部の物体座標を制御に一切使っていない。**
把持位置は、知覚用カメラの RGB 画像と深度バッファだけから決めている。
`mjData` の真値を読むのは、実行後に精度を採点するときだけ（`evaluate_against_truth`）。

> 「シミュレータだから座標が分かっただけ」を自分で潰しておかないと、
> 実機に持っていける話にならない、と考えたため。

AI の使い方は **推論のみ**（追加学習なし）。
オープン語彙検出なので、語彙を増やしても再学習が要らない。

## 動かす

### Google Colab（推奨・GPU 不要の PC でも可）

`notebooks/LangPick_Colab.ipynb` を Colab で開き、上から実行するだけ。
ランタイムを **T4 GPU** にすると 5 分、CPU だと 30 分程度。

### ローカル

```bash
pip install -r requirements.txt
python scripts/make_video.py --account YOUR_OMNICAMPUS_ID
```

事前にダウンロードした重みやミラーを使いたい場合は、環境変数で差し替えられる。

```bash
LANGPICK_MODEL=/path/to/owlv2 python scripts/make_video.py --account YOUR_ID
```

`out/PAI最終課題_<アカウント名>.mp4` と `out/results.json` が出力される。
動画にはタイトル・指示文・検出枠・フェーズ・判定がすべて焼き込まれているので、
**動画編集は不要**。

初回だけ `mujoco_menagerie` から Franka Emika Panda を sparse checkout で取得する
（Panda のみ・約 44MB・数秒）。

## 構成

| ファイル | 役割 |
|---|---|
| `src/langpick/scene.py` | MJCF の生成。卓上の 4 物体と 2 トレイ、2 つのカメラ |
| `src/langpick/camera.py` | カメラ内部パラメータ、RGB-D の取得、点群への逆投影 |
| `src/langpick/grasp.py` | バウンディングボックス → 3D 把持位置の推定 |
| `src/langpick/perception.py` | OWLv2 ラッパ、色閾値ベースライン、自己検証 |
| `src/langpick/instruction.py` | 指示文のスロット分解（日本語 / 英語） |
| `src/langpick/ik.py` | 減衰最小二乗法による逆運動学 |
| `src/langpick/controller.py` | 手先の直線軌道とグリッパ制御 |
| `src/langpick/episode.py` | 指示 1 件の実行と評価 |
| `src/langpick/experiment.py` | OWLv2 vs 色閾値の対照実験 |
| `src/langpick/video.py` | 情報を焼き込んだ mp4 の生成 |

## シーン設計の意図

卓上には **赤いブロック** と **赤い缶** が置いてある。同じ色で形が違う。
色の閾値処理だけのベースラインは、この 2 つに同じ領域を返してしまい、
「赤いブロック」と「赤い缶」を区別できない。
`experiment.py` はこの差を数字にする。

## 使える語彙

| 種別 | 使える語 |
|---|---|
| 色 | 赤 / 緑 / 青 / 紫 / 黄 |
| 物体 | ブロック（cube） / 缶（can） |
| 置き先 | トレイ（box） |

`「<色><物体>を<色>のトレイに入れて」` の形。英語でも動く。
語彙は `src/langpick/instruction.py` の表に足すだけで増やせる。

## ライセンス

MIT。Franka Emika Panda のモデルは
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0）、
検出器は [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)（Apache-2.0）。
