# Paul trap simulation

## 現在の状態

2026-10-06時点で、Phase 1（Python参照計算、`scripts/floquet_reference.py`）、Phase 2（C++/OpenMPのFloquet走査、`src/floquet_scan.cpp`）、Phase 3（境界追跡、`src/floquet_boundary.cpp`）、Phase 4の3次元軌道とアニメーション（`scripts/trajectory.py`、`scripts/animate_trap.py`）、Phase 5の捕獲確率（`src/trajectory.cpp`、`src/capture_scan.cpp`、`scripts/capture_map.py`）、Phase 6の描画分離・2D/3D GIF・一括再生成（`scripts/make_figures.py`、README画像は`docs/images/`）を実装済みです。インタラクティブHTMLは保留中。走査結果は暫定的に`.npy`とmetadata JSONで出力し、正式な結果形式はPhase 3で決める。

## 読み順

1. [既存ノートの確認結果](prior-work.md)
2. [物理モデルと無次元化](model.md)
3. [安定領域の数値計算法](stability.md)
4. [ソフトウェア構成](architecture.md)
5. [検証計画](validation.md)
6. [開発ロードマップ](roadmap.md)

## 目標

- 空気抵抗と重力を含む単一粒子の2D・3D軌道を計算する。
- 無次元パラメータ`a`、`q`、`b`におけるFloquet安定領域を高精度に求める。
- 有限電極、有限観測時間、初期条件分布を含む捕獲確率を求める。
- C++で大規模走査を高速化し、Pythonで解析・可視化する。
- 静止画、GIF、MP4、操作可能な3D HTMLを同じ計算結果から生成する。

## 対象外

初期段階では、粒子間相互作用、空間電荷、衝突の確率過程、実電極の有限要素電場、フィードバック制御は対象外とする。必要になった場合は、理想四重極モデルと混同せず別モデルとして追加する。
