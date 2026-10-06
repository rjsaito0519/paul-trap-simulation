# Documentation

このリポジトリの設計判断、数理モデル、検証条件、開発計画を管理します。

## Paul trap simulation

- [概要と読み順](simulation/README.md)
- [既存ノートの確認結果](simulation/prior-work.md)
- [物理モデルと無次元化](simulation/model.md)
- [安定領域の数値計算法](simulation/stability.md)
- [ソフトウェア構成](simulation/architecture.md)
- [検証計画](simulation/validation.md)
- [開発ロードマップ](simulation/roadmap.md)

2026-10-06時点ではPhase 3（C++/OpenMPによるFloquet安定判定と境界追跡）、Phase 4の3次元軌道・アニメーション、Phase 5の捕獲確率（C++/OpenMP）まで実装済みです。コード、依存ライブラリ、ファイル形式の追加は、上記文書の前提と未決事項を確認してから行います。
