# ソフトウェア構成

## 基本方針

数値計算の速度と、解析・可視化の柔軟性を分離する。

```text
設定・ジョブ作成・可視化: Python
              |
              v
       安定判定・大規模走査: C++
              |
              v
      数値結果と再現用metadata
```

## C++の責務

- モノドロミー行列の積分
- Floquet乗数と安定分類
- `(a, q, b)`の並列走査
- 3次元軌道と脱出イベント
- 初期条件のMonte Carlo走査
- 収束診断に必要な量の出力

数式、許容値、境界分類をCLIや描画コードへ分散させない。

## Pythonの責務

- 設定ファイルの作成と検証
- C++ジョブの起動
- 小規模な独立参照計算
- 安定領域、境界、捕獲確率の描画
- 2D/3D軌道、GIF、MP4の生成
- Plotlyによる回転・ズーム・時間スライダー付きHTML

PythonとC++の同名関数を増やさず、参照実装と本番実装の目的を明示する。

## インターフェース候補

初期段階はC++実行ファイルと設定ファイルを使用し、境界を安定させる。反復的な対話性能が必要になった場合にpybind11を検討する。

結果形式は次の条件で選ぶ。

- 大規模な密格子を効率よく部分読み込みできる。
- 単位、軸、solver、刻み、許容値、Git revisionを保持できる。
- Pythonから標準的に読める。
- 長期保存で特定の実行環境に依存しない。

ROOTは使用しない（2026-10-06決定）。HDF5、Zarr、NumPy形式などを候補とするが、実装前に代表データ量で比較する。CSVは小さい境界曲線やデバッグ用に限定する。

2026-10-06時点の暫定形式: 密な走査（`floquet_scan`）は`.npy`とmetadata JSON、境界点（`floquet_boundary`）はCSV（`b, polyline, index, closed, a_z, q_z, limiting, region, edge_axis, edge_i, edge_j`）とmetadata JSON。新しい依存を増やさないため、HDF5やZarrは大きな結果が必要になった段階で再検討する。

軌道ファイル（2026-10-06、Phase 6a）: `scripts/trajectory.py`の`save_run`/`load_run`が読み書きする`.npz`（`paultrap-trajectory-v1`）。描画（`animate_trap.py`）は計算と分離され、この形式を入力にできる。

```text
metadata    JSON文字列: format, created, command, git, python/numpy/scipy版, solver,
            setup（Setupの全フィールド、SI単位）, particles（初期条件）,
            trajectories（label, escaped, escape_time [s] または null, electrode）
config      元のTOML全文（無い場合は空文字列）
t_eval      要求したサンプル時刻 [s]（任意。アニメーションのフレーム時刻）
p{k}_t, p{k}_position [n, 3] m, p{k}_velocity [n, 3] m/s
            粒子kの軌道。脱出した場合は衝突点を最終サンプルとして含む
```

## 想定ディレクトリ

実装が始まった時点で必要なものだけ追加する。

```text
src/          C++実装とCLI
scripts/      Pythonの実行・描画スクリプト
configs/      version管理する小さな設定例
tests/        数式、境界、入出力、再現性のテスト
data/         Git管理外の入力
results/      Git管理外の計算結果と動画
docs/         設計と検証記録
```

## 並列化と再現性

- OpenMPを第一候補とし、スレッド数を実行時に指定可能にする。
- 並列順序に依存しない決定論的な結果を優先する。
- Monte Carloではseedとサンプル番号から各乱数系列を決定する。
- 出力にGit revision、dirty状態、コンパイラ、依存関係、実行コマンド、設定を残す。
