# Paul Trap Simulation

空気抵抗と重力を含むポールトラップの粒子軌道、Mathieu方程式の安定領域、有限電極内での捕獲可能性を数値計算するためのプロジェクトです。

2026-10-06時点では設計・検証計画だけを管理しており、シミュレーション実装はまだありません。実装を始める前に、[docs/simulation/README.md](docs/simulation/README.md)から設計文書を確認してください。

## 想定する構成

- C++17: Floquet判定、パラメータ走査、Monte Carlo計算
- Python: 設定、結果解析、2D/3D可視化、GIF・MP4・インタラクティブHTML
- CMake: C++ビルド
- ROOT: 大規模結果の保存候補。採用は実装前に再検討する

入力データや生成結果は`data/`と`results/`に置き、Gitでは管理しません。プロジェクトの索引は[docs/README.md](docs/README.md)、AI向けの作業規則は[AGENTS.md](AGENTS.md)にあります。
