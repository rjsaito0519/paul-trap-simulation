# Paul Trap Simulation

空気抵抗と重力を含むポールトラップの粒子軌道、Mathieu方程式の安定領域、有限電極内での捕獲可能性を数値計算するためのプロジェクトです。

2026-10-06時点ではPhase 3（C++/OpenMPによるFloquet安定判定と境界追跡）、Phase 4の3次元軌道・アニメーション、Phase 5の捕獲確率（C++/OpenMP）まで実装済みです。実装を始める前に、[docs/simulation/README.md](docs/simulation/README.md)から設計文書を確認してください。

## 想定する構成

- C++17: Floquet判定、パラメータ走査、Monte Carlo計算
- Python: 設定、結果解析、2D/3D可視化、GIF・MP4・インタラクティブHTML
- CMake: C++ビルド

CERN ROOTは使用しない（2026-10-06決定）。

## 使い方

```bash
./build.sh                                    # C++ビルド（OpenMP必須）
(cd .build && ctest)                          # C++セルフテスト
python3 -m unittest discover -s tests         # Python参照計算とC++回帰テスト
python3 scripts/floquet_reference.py --b 0    # Python参照計算: 安定領域を results/ に出力
.build/bin/floquet_scan --output results/scan_b0 --b 0 --na 401 --nq 401 --threads 8
.build/bin/floquet_boundary --output results/boundary --threads 8   # 安定境界の追跡
python3 scripts/plot_stability.py results/boundary                  # 境界の図
python3 scripts/explainer_figures.py                                # 解説図 results/explainer/
python3 scripts/stability_overview.py                               # b ごとの安定領域全体
python3 scripts/trajectory.py configs/example_microparticle.toml    # 粒子軌道と捕獲/脱出
python3 scripts/animate_trap.py configs/example_microparticle.toml --rf-cycles 20   # 3D GIF（.mp4 も可）
python3 scripts/capture_map.py configs/example_microparticle.toml                   # 捕獲確率マップ（[capture]節）
```

共有ログインノードでは、Pythonの数値ライブラリが全コアを使わないよう`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`などを設定して実行してください。

入力データや生成結果は`data/`と`results/`に置き、Gitでは管理しません。プロジェクトの索引は[docs/README.md](docs/README.md)、AI向けの作業規則は[AGENTS.md](AGENTS.md)にあります。
