"""05 の成果物の行 dataclass と CSV 列 (``anomaly.csv`` / ``anomaly_threshold.csv``).

計算をせず、**成果物の形だけ**を持つ。分けてあるのは 05 の実験層が1ファイル
600 行を上限にしているため (D-63) で、行の docstring (列の意味) が重いぶんを
``anomaly.py`` から切り出した形である。

このモジュールが構造で守っているのは D-55 の1点:

    PA-F1 の列名を作れるのは ``pa_columns`` だけで、その関数は
    ``(pa_f1_k*, pa_f1_random_k*)`` を**必ず対で**返す。

``pa_f1`` 単独の列を作るには関数を書き換えるしかなく、書き換えれば
``tests/test_experiment_anomaly.py::
test_point_adjust_is_never_reported_without_the_random_control`` が
列の対応を数えて落とす。

``f1_test_optimal`` (D-56 の**別列**) と PA%K の列は設定で増減するので、
列順の単一の真実は「``AnomalyRow`` の宣言順」+「``anomaly_csv_columns`` の
規則」の2段になる。``experiment/threshold.py`` の ``threshold_csv_columns`` /
``threshold_row_as_dict`` と同じ形である。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

from rc_basics_lab.config import Anomaly05Config
from rc_basics_lab.metrics_detection import PointAdjustReport

ANOMALY_CSV = "anomaly.csv"
ANOMALY_THRESHOLD_CSV = "anomaly_threshold.csv"


@dataclass(frozen=True, slots=True)
class AnomalyRow:
    """``anomaly.csv`` の1行 (1系列 x 1系統 x 1レプリケート)。

    ``f1_test_optimal`` と ``point_adjust`` だけは**設定によって列になったり
    ならなかったり**するので、列順の単一の真実 ``ANOMALY_SCALAR_COLUMNS`` から
    外してある (``anomaly_csv_columns`` / ``anomaly_row_as_dict`` が扱う)。

    Attributes:
        dataset: ``dataset.source``。
        series: 系列名。
        method: ``ANOMALY_METHODS`` のいずれか。
        replicate: レプリケート番号。
        seed_reservoir: リザバーの基底シード。
        seed_task: 系列生成の基底シード。
        seed_split: 分割の基底シード。
        seed_control: 一様乱数対照の基底シード (D-61)。
        normalize: 前処理の方式 (来歴)。
        preprocessor_id: 前処理係数の指紋。**(系列, レプリケート) 内で単一値**
            であることが D-57 の成果物側の証拠になる。
        selected_alpha: 検証で選ばれた alpha (D-04)。学習しない系統は ``nan``。
        auprc: **主指標**。point-adjust を一切通していない (D-54 / D-55)。
        auprc_random: 同じ評価点での一様乱数対照の AUPRC (D-61)。異常率付近に
            張り付く —— 行の中に基準線を置くことで、図を見ない読者にも
            「その差は雑音か」が判定できる。
        anomaly_rate: 評価点のうち異常だった割合。
        threshold: 較正区間から決めた運用閾値 (D-56)。
        f1_calibrated: その閾値での F1。
        precision_calibrated: その閾値での適合率。
        recall_calibrated: その閾値での再現率。
        far_test: テスト区間での実測誤報率。
        n_evaluated: 評価に使った点数 (``ignore`` を落とした後)。
        n_train: 学習行数。
        n_calibration: 較正行数。
        n_test: テスト行数。
        t0: 全系統共通の基準行 (D-05)。
        split_offset: ``seeds.split`` が選んだオフセット。
        wall_time_s: この行の計算時間。
        f1_test_optimal: テスト側で閾値を最適化したときの F1 (**参考値**)。
            報告しない設定では ``nan`` で、列そのものが出ない。
        point_adjust: PA%K の報告 (D-55)。``PointAdjustReport`` は
            ``pa_f1`` と ``pa_f1_random`` を同時にしか持てない。
    """

    dataset: str
    series: str
    method: str
    replicate: int
    seed_reservoir: int
    seed_task: int
    seed_split: int
    seed_control: int
    normalize: str
    preprocessor_id: str
    selected_alpha: float
    auprc: float
    auprc_random: float
    anomaly_rate: float
    threshold: float
    f1_calibrated: float
    precision_calibrated: float
    recall_calibrated: float
    far_test: float
    n_evaluated: int
    n_train: int
    n_calibration: int
    n_test: int
    t0: int
    split_offset: int
    wall_time_s: float
    f1_test_optimal: float = math.nan
    point_adjust: tuple[PointAdjustReport, ...] = ()


F1_TEST_OPTIMAL_COLUMN = "f1_test_optimal"
"""テスト側最適化の**別列** (D-56)。``f1_calibrated`` を置き換えることはない。"""

_OPTIONAL_FIELDS = frozenset({F1_TEST_OPTIMAL_COLUMN, "point_adjust"})

ANOMALY_SCALAR_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(AnomalyRow) if item.name not in _OPTIONAL_FIELDS
)
"""常に出る列の順 (``AnomalyRow`` の宣言順が単一の真実)。"""

PA_F1_PREFIX = "pa_f1_k"
PA_F1_RANDOM_PREFIX = "pa_f1_random_k"


def pa_columns(k: float) -> tuple[str, str]:
    """PA%K の列名を**必ず対で**返す (D-55)。

    片方だけを返す関数をこのモジュールに置かない。``pa_f1`` 単独の列を作る
    には、この関数を書き換えるか使わないかしかない —— どちらも
    ``test_point_adjust_is_never_reported_without_the_random_control`` が
    列の対応を数えて落とす。
    """
    return (f"{PA_F1_PREFIX}{k:g}", f"{PA_F1_RANDOM_PREFIX}{k:g}")


def anomaly_csv_columns(config: Anomaly05Config) -> tuple[str, ...]:
    """``anomaly.csv`` の列順 (設定で増減する列を含む)。"""
    columns = list(ANOMALY_SCALAR_COLUMNS)
    if config.threshold.report_test_optimal:
        columns.append(F1_TEST_OPTIMAL_COLUMN)
    if config.evaluation.report_point_adjust:
        for k in config.evaluation.pa_k_grid:
            columns.extend(pa_columns(k))
    return tuple(columns)


def anomaly_row_as_dict(row: AnomalyRow) -> dict[str, object]:
    """1行を「列名 -> 値」にする (列順は ``anomaly_csv_columns`` と同じ規則)。

    ``f1_test_optimal`` は「報告しない = ``nan``」で表す。``best_test_f1`` は
    ``[0, 1]`` の値しか返さないので、``nan`` と有効値が混ざることはない。
    """
    values: dict[str, object] = {
        name: getattr(row, name) for name in ANOMALY_SCALAR_COLUMNS
    }
    if not math.isnan(row.f1_test_optimal):
        values[F1_TEST_OPTIMAL_COLUMN] = row.f1_test_optimal
    for report in row.point_adjust:
        pa_column, pa_random_column = pa_columns(report.k)
        values[pa_column] = report.pa_f1
        values[pa_random_column] = report.pa_f1_random
    return values


@dataclass(frozen=True, slots=True)
class ThresholdSweepRow:
    """``anomaly_threshold.csv`` の1行 (5-B)。

    Attributes:
        dataset: ``dataset.source``。
        series: 系列名。
        method: 系統名。
        replicate: レプリケート番号。
        target_false_alarm_rate: 掃引した警報予算。
        threshold: その予算に対応する閾値。
        precision: 適合率。
        recall: 再現率。
        f1: F1。
        false_alarm_rate: 実測誤報率。
        n_alarms: 警報数。
        calibrated_threshold: 較正区間から決めた運用閾値 (図に運用点を
            重ねるため、掃引の各行が基準を持ち歩く)。
    """

    dataset: str
    series: str
    method: str
    replicate: int
    target_false_alarm_rate: float
    threshold: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    n_alarms: int
    calibrated_threshold: float


ANOMALY_THRESHOLD_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(ThresholdSweepRow)
)
"""``anomaly_threshold.csv`` の列順 (``ThresholdSweepRow`` の宣言順)。"""


def rows_as_dicts(rows: Sequence[AnomalyRow]) -> list[dict[str, object]]:
    """CSV 書き出し用に行を dict へ落とす (列順は ``anomaly_csv_columns``)。"""
    return [anomaly_row_as_dict(row) for row in rows]


__all__ = [
    "ANOMALY_CSV",
    "ANOMALY_SCALAR_COLUMNS",
    "ANOMALY_THRESHOLD_CSV",
    "ANOMALY_THRESHOLD_CSV_COLUMNS",
    "F1_TEST_OPTIMAL_COLUMN",
    "PA_F1_PREFIX",
    "PA_F1_RANDOM_PREFIX",
    "AnomalyRow",
    "ThresholdSweepRow",
    "anomaly_csv_columns",
    "anomaly_row_as_dict",
    "pa_columns",
    "rows_as_dicts",
]
