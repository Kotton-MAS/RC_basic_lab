"""実験05 の掃引の格子 (5-C プロトコル感度 / 5-D N と性能) の設定 (D-13).

``anomaly05.py`` から分けたのは行数のためである —— config package は
1モジュール**非空 300 行**が上限 (``tests/test_config_package_layout.py``) で、
T3 完了時点の ``anomaly05.py`` は非空 285 行だった。分ける単位としても素直で、
ここにあるのは「1本の実験の条件」ではなく**条件をどう振るか**である。

格子は ``AnomalyPreprocessConfig`` / ``AnomalyReservoirConfig`` の葉と1対1に
対応する。**既定の格子はそれぞれの既定値を含む**が、その一致は定数を
書き写して保つ (逆向きに import すると ``anomaly05`` との循環になる) ——
書き写しが崩れていないことは
``tests/test_experiment_anomaly_sweep.py::test_the_default_grids_contain_the_headline_condition``
が実測する。掃引側は既定値を含まなければ ``ValueError`` になる (D-79) ので、
崩れたまま静かに回ることはない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnomalyProtocolSweepConfig:
    """5-C の格子 (前処理・スコア整形のプロトコルを振る3軸)。

    3軸は ``AnomalyPreprocessConfig`` の ``normalize`` / ``input_window`` /
    ``score_smoothing`` に1対1で対応し、格子点はその**全組合せ**である
    (仕様 §7 リスク3 が予算超過時の退避として挙げている「軸ごとに1つずつ
    振る」形は、交互作用が見えなくなるので採らない)。

    Attributes:
        normalize_grid: ``preprocess.normalize`` に入れる値
            (``NORMALIZE_METHODS`` の4値から選ぶ)。
        input_window_grid: ``preprocess.input_window`` に入れる値。
        score_smoothing_grid: ``preprocess.score_smoothing`` に入れる値。
            ``1`` は平滑化なし。
    """

    normalize_grid: tuple[str, ...] = ("zscore", "minmax", "robust")
    input_window_grid: tuple[int, ...] = (8, 16, 32)
    score_smoothing_grid: tuple[int, ...] = (1, 8, 16)


@dataclass(frozen=True, slots=True)
class AnomalySizeSweepConfig:
    """5-D の格子 (リザバーの大きさ N を振る軸)。

    劣化点を測る割合 (基準 N の 90%) は設定の葉にしない —— 報告する量の名前
    (``n_units_at_90pct``) がその値そのものなので、葉にすると列名が嘘になる
    (``experiment/anomaly_sweep.py`` の ``DEGRADATION_FRACTION``)。

    Attributes:
        n_units_grid: ``reservoir.n_units`` に入れる値。**基準 N
            (``reservoir.n_units``) を必ず含む**こと (含まないと 5-D の基準行が
            5-A と別条件になるため ``ValueError``、D-79)。
    """

    n_units_grid: tuple[int, ...] = (25, 50, 100, 200)


__all__ = [
    "AnomalyProtocolSweepConfig",
    "AnomalySizeSweepConfig",
]
