"""05 の成果物の行 dataclass と CSV 列 (4枚ぶん).

``anomaly.csv`` / ``anomaly_threshold.csv`` (5-A / 5-B) に加えて、T4 が
``anomaly_protocol.csv`` (5-C) と ``anomaly_size.csv`` (5-D) の行を足した。
掃引そのもの (格子の組み立てと集計) は ``experiment/anomaly_sweep.py`` にあり、
ここは**成果物の形だけ**を持つ —— T3 が ``anomaly.py`` から行 dataclass を
切り出したのと同じ分け方で、掃引の行を ``anomaly_sweep.py`` に置くと
あちらが 600 行 (D-63) に届く。

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


@dataclass(frozen=True, slots=True)
class ProtocolSweepRow:
    """``anomaly_protocol.csv`` の1行 (1格子点 x 1系統、5-C)。

    5-C が答えるのは「前処理・整形のプロトコルを変えると手法の順位が
    入れ替わるか」である。**全系統の順位を記録したうえで、各系統に
    「一様乱数対照と区別できるか」の印を付ける** (D-78) ——
    対照と区別できない系統どうしの順位が入れ替わっても、それは
    プロトコル感度ではなく雑音である。

    条件そのものの量 (``kendall_tau`` / 不一致対の数) は、その格子点の
    **全行が持ち歩く** (``ThresholdSweepRow.calibrated_threshold`` と同じ
    流儀)。図は CSV の行だけを読むので、条件レベルの量を別ファイルに置くと
    図が2枚の CSV を突き合わせることになる。

    Attributes:
        dataset: ``dataset.source``。
        normalize: この格子点の ``preprocess.normalize``。
        input_window: この格子点の ``preprocess.input_window``。
        score_smoothing: この格子点の ``preprocess.score_smoothing``。
        is_headline: この格子点が ``config.preprocess`` と一致する
            (= 5-A と同じ条件) か。順位の基準もこの点である (D-79)。
        method: ``ANOMALY_METHODS`` のいずれか。
        auprc_mean: この格子点での ``auprc`` の平均 (系列 x レプリケート)。
        auprc_sd: 同じ集合の標準偏差 (``ddof=1``。1点しかなければ 0)。
        auprc_random_mean: 同じ行が持つ ``auprc_random`` の平均 (D-61)。
        n_pairs: 集計に使った (系列, レプリケート) の対の数。
        n_better_than_control: そのうち ``auprc > auprc_random`` だった対の数。
        control_sign_p: 片側符号検定の p 値 (帰無仮説「対照より高い確率は
            1/2」)。**印の根拠を行の中に置く**ための列。
        distinguishable: ``control_sign_p <= CONTROL_SIGN_TEST_ALPHA`` か
            (= 一様乱数対照と区別できるか、D-78)。
        rank: この格子点での順位 (``auprc_mean`` の降順、1 が最良。同値は
            同順位)。**対照を除外せず6系統すべてに付ける**。
        reference_rank: 基準の格子点 (``is_headline``) での順位。
        rank_changed: ``rank != reference_rank``。
        reference_distinguishable: 基準の格子点での ``distinguishable``。
            「区別できる系統どうしの入れ替わりか」を行だけで判定するために要る。
        kendall_tau: 基準の格子点の順位との Kendall tau-b (条件の量)。
        n_discordant_pairs: 基準と順序が逆転した系統対の数 (条件の量)。
        n_discordant_pairs_distinguishable: そのうち**両方の系統が両条件で
            対照と区別できる**対の数 (条件の量)。この数が記事の結論を分ける。
    """

    dataset: str
    normalize: str
    input_window: int
    score_smoothing: int
    is_headline: bool
    method: str
    auprc_mean: float
    auprc_sd: float
    auprc_random_mean: float
    n_pairs: int
    n_better_than_control: int
    control_sign_p: float
    distinguishable: bool
    rank: int
    reference_rank: int
    rank_changed: bool
    reference_distinguishable: bool
    kendall_tau: float
    n_discordant_pairs: int
    n_discordant_pairs_distinguishable: int


ANOMALY_PROTOCOL_CSV = "anomaly_protocol.csv"
ANOMALY_PROTOCOL_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(ProtocolSweepRow)
)
"""``anomaly_protocol.csv`` の列順 (``ProtocolSweepRow`` の宣言順)。"""


@dataclass(frozen=True, slots=True)
class SizeSweepRow:
    """``anomaly_size.csv`` の1行 (1つの N x 1系統、5-D)。

    5-D が答えるのは「N を削ると性能がどこで落ちるか」である。落ちる場所を
    学習量不足と混ぜないため、**全系列が同じ ``n_train`` で回っていること**を
    掃引の入口が検査する (D-78)。その値を行が持ち歩くので、成果物だけで
    「学習量は揃っていた」を確認できる。

    Attributes:
        dataset: ``dataset.source``。
        n_units: この行のリザバー規模 N。
        method: ``ANOMALY_METHODS`` のいずれか。**N に依存しない系統
            (対照を含む) も落とさない** (D-61)。図の基準線になる。
        auprc_mean: ``auprc`` の平均 (系列 x レプリケート)。
        auprc_sd: 同じ集合の標準偏差 (``ddof=1``)。
        auprc_random_mean: ``auprc_random`` の平均 (D-61)。
        n_pairs: 集計に使った対の数。
        n_better_than_control: ``auprc > auprc_random`` だった対の数。
        control_sign_p: 片側符号検定の p 値。
        distinguishable: 一様乱数対照と区別できるか (D-78)。
        reference_n_units: 基準 N (``reservoir.n_units``。5-A と同じ条件)。
        auprc_reference: 基準 N でのその系統の ``auprc_mean``。
        auprc_ratio: ``auprc_mean / auprc_reference`` (基準が 0 なら ``nan``)。
        below_reference_fraction: ``auprc_ratio`` が
            ``DEGRADATION_FRACTION`` (0.9) を下回るか。``n_units_at_90pct``
            はこの列だけから決まる。
        n_train: 学習行数。**全行で同一**であることを掃引の入口が要求する。
    """

    dataset: str
    n_units: int
    method: str
    auprc_mean: float
    auprc_sd: float
    auprc_random_mean: float
    n_pairs: int
    n_better_than_control: int
    control_sign_p: float
    distinguishable: bool
    reference_n_units: int
    auprc_reference: float
    auprc_ratio: float
    below_reference_fraction: bool
    n_train: int


ANOMALY_SIZE_CSV = "anomaly_size.csv"
ANOMALY_SIZE_CSV_COLUMNS: tuple[str, ...] = tuple(
    item.name for item in fields(SizeSweepRow)
)
"""``anomaly_size.csv`` の列順 (``SizeSweepRow`` の宣言順)。"""


__all__ = [
    "ANOMALY_CSV",
    "ANOMALY_PROTOCOL_CSV",
    "ANOMALY_PROTOCOL_CSV_COLUMNS",
    "ANOMALY_SCALAR_COLUMNS",
    "ANOMALY_SIZE_CSV",
    "ANOMALY_SIZE_CSV_COLUMNS",
    "ANOMALY_THRESHOLD_CSV",
    "ANOMALY_THRESHOLD_CSV_COLUMNS",
    "F1_TEST_OPTIMAL_COLUMN",
    "PA_F1_PREFIX",
    "PA_F1_RANDOM_PREFIX",
    "AnomalyRow",
    "ProtocolSweepRow",
    "SizeSweepRow",
    "ThresholdSweepRow",
    "anomaly_csv_columns",
    "anomaly_row_as_dict",
    "pa_columns",
    "rows_as_dicts",
]
