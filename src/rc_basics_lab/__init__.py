"""rc_basics_lab — リザバー計算の基礎編ラボ.

連載記事 01〜05 で共有する実験基盤。サブパッケージ構成:

- ``config`` / ``seeds`` / ``metrics`` / ``meta``: 実験の土台 (設定・乱数・計量)
- ``diagnostics``: 状態系列 ``X`` だけを入力に取る診断層 (``reservoir`` に依存しない)
- ``reservoir``: 入力系列から状態系列を作る側 (ESN)
- ``readout``: 設計行列の構築とリッジ回帰 (3手法の差は ``FeatureSpec`` だけ)
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
