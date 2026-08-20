"""異常スコアの構成器 —— 6系統を分ける唯一の場所 (D-61 / D-05).

``readout/design.py`` の ``FeatureSpec`` が3ベースラインを分ける唯一の場所で
あるのと同じ形で、05 の6系統は **``ScoreSpec`` の差だけ**で表され、
``build_score`` という単一の呼び出し口を通る。手法ごとに別関数を実験層から
直接呼ぶと「実装差ではなくスコア定義の差」という主張が崩れる。

6系統 (``ANOMALY_METHODS``):

===========================  ==================================================
``esn_residual``             リザバー読み出しの1ステップ先予測残差
``delay_line_residual``      遅延線読み出しの1ステップ先予測残差
``persistence_residual``     直前値をそのまま予測とした残差 (学習なし)
``moving_statistics``        後方窓の平均・標準偏差からの乖離 (学習なし)
``random_control``           一様乱数 (**常置の対照**, D-61)
``input_norm_control``       前処理後の入力の絶対値 (**常置の対照**, D-61)
===========================  ==================================================

**後ろ2つを設定から外せない**のが D-61 の実体である。``ANOMALY_METHODS`` は
モジュール定数で、``Anomaly05Config`` には手法を選ぶ葉が1つも無い ——
対照を設定で外せるようにすると、予算が厳しい日に真っ先に外され、外した図が
記事に載る。

spec の型は5つで系統は6つある。ESN と遅延線が同じ ``RidgeResidualSpec`` を
共有し、違いを ``FeatureSpec`` 1個に閉じているためで、これは
「新しい手法分岐を作らない」(``readout/design.py`` の規律) の帰結である。

スコアは**時刻 index の空間**で揃える。``AnomalyScore.first_valid`` より手前は
NaN で埋め (``build_design_matrix`` と同じ流儀)、実験層が
``compute_t0`` で全系統の ``first_valid`` と washout から単一の基準行を決める
(D-05)。0 埋めにすると ``t0`` の取り違えが「少しずれたスコア」として静かに
通ってしまう。

このモジュールは**乱数源を持たない**。一様乱数対照のスコアは
``ScoreInputs.control_scores`` として値で受け取る (``metrics_detection.
point_adjust_report`` が対照を引数で受けるのと同じ理由 —— 内部で引くと
呼び出しごとに値が動き CSV の再現性が落ちる)。同じ配列を PA の対照にも使う
ことで、「乱数対照の行」と「PA の乱数対照」が同じ乱数列であることが構造で
保証される。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.config import Anomaly05Config
from rc_basics_lab.readout.design import (
    DelayLineSpec,
    FeatureSpec,
    ReservoirSpec,
    build_design_matrix,
    first_valid_for,
)
from rc_basics_lab.readout.ridge import fit_ridge, predict, select_alpha
from rc_basics_lab.types import FloatArray

ESN_RESIDUAL = "esn_residual"
DELAY_LINE_RESIDUAL = "delay_line_residual"
PERSISTENCE_RESIDUAL = "persistence_residual"
MOVING_STATISTICS = "moving_statistics"
RANDOM_CONTROL = "random_control"
INPUT_NORM_CONTROL = "input_norm_control"

CONTROL_METHODS: tuple[str, ...] = (RANDOM_CONTROL, INPUT_NORM_CONTROL)
"""**常置の対照** (D-61)。設定から外す経路を作らない。

一様乱数の AUPRC は異常率付近に張り付くため「主指標が point-adjust を通って
いないこと」の証拠として機能し、入力ノルムは「学習していないものがどこまで
届くか」の下限になる。
"""

ANOMALY_METHODS: tuple[str, ...] = (
    ESN_RESIDUAL,
    DELAY_LINE_RESIDUAL,
    PERSISTENCE_RESIDUAL,
    MOVING_STATISTICS,
    *CONTROL_METHODS,
)
"""6系統を列挙する**唯一の場所**。``build_score_specs`` の鍵と厳密に一致する。"""

_STD_FLOOR = 1e-12
"""移動統計の標準偏差の下限。

窓が定数のときに 0 除算で ``inf`` / ``nan`` が出ると
``metrics_detection.precision_recall_curve`` が「非有限スコア」で例外にする。
検知指標側で落とすより、スコア構成器の側で有限に閉じるほうが原因が読める。
"""


@dataclass(frozen=True, slots=True)
class RidgeResidualSpec:
    """教師強制の1ステップ先予測の残差 (ESN / 遅延線に共通)。

    ``x[t]`` を ``phi[t-1]`` から線形に予測し、``|x[t] - x_hat[t]|`` を
    スコアにする。自走は要らないので ``readout/autoregressive.py`` には
    触れない (``StateUpdater`` の「reservoir 非依存」の書き方だけを踏襲する)。

    Attributes:
        feature: ``ReservoirSpec`` (ESN) か ``DelayLineSpec`` (遅延線)。
            **系統の違いはこの1フィールドだけ**である。
    """

    feature: FeatureSpec


@dataclass(frozen=True, slots=True)
class PersistenceSpec:
    """直前値を予測とした残差 ``|x[t] - x[t-1]|`` (学習なしの下限)。"""


@dataclass(frozen=True, slots=True)
class MovingStatisticsSpec:
    """後方窓の平均・標準偏差からの乖離 ``|x[t] - mu| / sigma`` (学習なし)。

    Attributes:
        window: ``t`` の**手前** ``window`` 点で統計を取る (``t`` 自身を
            含めない —— 含めると異常点自身が自分の基準を押し上げる)。
    """

    window: int


@dataclass(frozen=True, slots=True)
class RandomControlSpec:
    """一様乱数の対照 (D-61)。値は ``ScoreInputs.control_scores`` から来る。"""


@dataclass(frozen=True, slots=True)
class InputNormControlSpec:
    """前処理後の入力の絶対値 ``|x[t]|`` (D-61)。"""


type ScoreSpec = (
    RidgeResidualSpec
    | PersistenceSpec
    | MovingStatisticsSpec
    | RandomControlSpec
    | InputNormControlSpec
)
"""6系統を切り替える唯一の軸 (``FeatureSpec`` と同じ役割)。"""


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """全系統が共有する材料 (1レプリケート = 1インスタンス、D-57)。

    Attributes:
        values: **前処理後**の系列 ``(T, 1)``。係数は
            ``AnomalyPreprocessor.from_training_prefix`` が作った1組だけで、
            ここに来る時点で全系統に同じものが配られている。
        states: リザバー状態 ``(T, N)``。
        control_scores: 一様乱数 ``(T,)``。対照の行と PA の対照が**同じ乱数列**
            であることを構造で保証するため、値で1本だけ持ち回る。
        train: 学習行 (係数推定に使う連続区間)。
        calibration: 較正行。alpha の選択にも使う —— **ラベルを使わない**ので
            D-56 に触れない (閾値もここから決まる)。
        alphas: ridge の探索格子。全系統が同一格子を読む (D-04)。
    """

    values: FloatArray
    states: FloatArray
    control_scores: FloatArray
    train: range
    calibration: range
    alphas: tuple[float, ...]

    @property
    def n_steps(self) -> int:
        """系列長 T。"""
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """1系統ぶんの異常スコア。

    Attributes:
        values: ``(T,)`` のスコア (大きいほど異常)。``first_valid`` より手前は
            NaN。
        first_valid: スコアが定義される最初の行 index。
        selected_alpha: 検証で選ばれた alpha。学習しない系統では ``nan``
            (「格子を読んでいない」ことを列で見分けられるようにする)。
    """

    values: FloatArray
    first_valid: int
    selected_alpha: float


def build_score_specs(config: Anomaly05Config) -> Mapping[str, ScoreSpec]:
    """6系統の spec を返す。**系統を列挙する唯一の場所** (D-61)。

    鍵は必ず ``ANOMALY_METHODS`` と一致する。設定から系統を足したり外したり
    する経路は無い —— ``Anomaly05Config`` に手法を選ぶ葉が1つも無いことが、
    「対照を外せない」の実体である。
    """
    window = config.preprocess.input_window
    specs: dict[str, ScoreSpec] = {
        ESN_RESIDUAL: RidgeResidualSpec(feature=ReservoirSpec()),
        DELAY_LINE_RESIDUAL: RidgeResidualSpec(feature=DelayLineSpec(n_lags=window)),
        PERSISTENCE_RESIDUAL: PersistenceSpec(),
        MOVING_STATISTICS: MovingStatisticsSpec(window=window),
        RANDOM_CONTROL: RandomControlSpec(),
        INPUT_NORM_CONTROL: InputNormControlSpec(),
    }
    if not config.evaluation.report_point_adjust:
        del specs[INPUT_NORM_CONTROL]
    return specs


def score_first_valid(spec: ScoreSpec) -> int:
    """スコアを作らずに ``first_valid`` を求める (``first_valid_for`` と同型)。

    実験層は系列を流す**前に** ``t0 = compute_t0(全系統の first_valid,
    washout)`` を知る必要がある (D-05。学習区間の位置が t0 で決まり、学習は
    スコア構成の中で行われる)。予測を実験層へ書き写すと、系統を足したときに
    予測と実際の ``first_valid`` が黙って食い違う。

    Raises:
        ValueError: ``MovingStatisticsSpec.window`` が 1 未満の場合。
    """
    match spec:
        case RidgeResidualSpec():
            # phi[t-1] を使うので、設計行列の first_valid より1行遅れる。
            return first_valid_for(spec.feature) + 1
        case PersistenceSpec():
            return 1
        case MovingStatisticsSpec():
            if spec.window < 1:
                raise ValueError(f"window は 1 以上が必要です: {spec.window}")
            return spec.window
        case RandomControlSpec() | InputNormControlSpec():
            return 0


def smoothing_shift(score_smoothing: int) -> int:
    """後方移動平均で失う行数 (``score_smoothing - 1``)。

    Raises:
        ValueError: ``score_smoothing`` が 1 未満の場合。
    """
    if score_smoothing < 1:
        raise ValueError(f"score_smoothing は 1 以上が必要です: {score_smoothing}")
    return score_smoothing - 1


def _rows(array: FloatArray, selection: range) -> FloatArray:
    """連続区間の行を切り出す (分割は常に連続区間)。"""
    block: FloatArray = array[selection.start : selection.stop]
    return block


def _lagged(matrix: FloatArray) -> FloatArray:
    """``phi[t-1]`` を ``t`` 行に置いた行列 (先頭行は NaN)。

    教師強制の1ステップ先予測を「時刻 index の空間」で書くための唯一の道具。
    目標側 (``values``) をずらす書き方にすると、系統ごとに「どちらをずらしたか」
    が割れる。
    """
    shifted: FloatArray = np.full(matrix.shape, np.nan, dtype=np.float64)
    shifted[1:] = matrix[:-1]
    return shifted


def _empty_scores(n_steps: int) -> FloatArray:
    """全行 NaN のスコア配列 (有効行だけを後から埋める)。"""
    values: FloatArray = np.full(n_steps, np.nan, dtype=np.float64)
    return values


def _ridge_residual(spec: RidgeResidualSpec, inputs: ScoreInputs) -> AnomalyScore:
    """1ステップ先予測の残差 (ESN / 遅延線)。

    alpha は較正区間で選ぶ (D-04)。**ラベルを1ビットも見ない**ので D-56 とは
    独立で、較正区間に異常が混ざっていても「教師なしで手に入る材料だけを
    使っている」という性格は変わらない。
    """
    design = build_design_matrix(spec.feature, inputs.values, inputs.states)
    lagged = _lagged(design.phi)
    first_valid = design.first_valid + 1
    target = inputs.values
    bias_column = design.bias_column
    selection = select_alpha(
        _rows(lagged, inputs.train),
        _rows(target, inputs.train),
        _rows(lagged, inputs.calibration),
        _rows(target, inputs.calibration),
        inputs.alphas,
        bias_column=bias_column,
    )
    coefficients = fit_ridge(
        _rows(lagged, inputs.train),
        _rows(target, inputs.train),
        selection.alpha,
        bias_column=bias_column,
    )
    prediction = predict(lagged[first_valid:], coefficients)
    values = _empty_scores(inputs.n_steps)
    values[first_valid:] = np.abs(target[first_valid:, 0] - prediction[:, 0])
    return AnomalyScore(
        values=values, first_valid=first_valid, selected_alpha=selection.alpha
    )


def _persistence_residual(inputs: ScoreInputs) -> AnomalyScore:
    """``|x[t] - x[t-1]|``。学習しないので alpha は ``nan``。"""
    series = inputs.values[:, 0]
    values = _empty_scores(inputs.n_steps)
    values[1:] = np.abs(series[1:] - series[:-1])
    return AnomalyScore(values=values, first_valid=1, selected_alpha=math.nan)


def _trailing_sums(series: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """``t`` の手前 ``window`` 点の和と2乗和 (``t >= window`` の行だけ有効)。

    累積和の差分で O(T) にする。``sliding_window_view`` で書くと計算量が
    ``T * window`` になり、``input_window`` を大きくしたときだけ静かに重くなる
    (確保軸を1本増やさずに済ませるための選択)。
    """
    cumulative: FloatArray = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(series))
    )
    cumulative_squared: FloatArray = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(series * series))
    )
    total = cumulative[window:-1] - cumulative[: -window - 1]
    total_squared = cumulative_squared[window:-1] - cumulative_squared[: -window - 1]
    return total, total_squared


def _moving_statistics(spec: MovingStatisticsSpec, inputs: ScoreInputs) -> AnomalyScore:
    """後方窓の平均・標準偏差からの乖離 ``|x[t] - mu| / sigma``。"""
    first_valid = score_first_valid(spec)
    series = inputs.values[:, 0]
    window = float(spec.window)
    total, total_squared = _trailing_sums(series, spec.window)
    mean = total / window
    variance = np.maximum(total_squared / window - mean * mean, 0.0)
    scale = np.maximum(np.sqrt(variance), _STD_FLOOR)
    values = _empty_scores(inputs.n_steps)
    values[first_valid:] = np.abs(series[first_valid:] - mean) / scale
    return AnomalyScore(values=values, first_valid=first_valid, selected_alpha=math.nan)


def _random_control(inputs: ScoreInputs) -> AnomalyScore:
    """一様乱数 (D-61)。値は呼び出し側が引いたものをそのまま使う。"""
    control: FloatArray = np.array(inputs.control_scores, dtype=np.float64, copy=True)
    if control.shape != (inputs.n_steps,):
        raise ValueError(
            "control_scores の形状が系列と一致しません: "
            f"{control.shape} != {(inputs.n_steps,)}"
        )
    return AnomalyScore(values=control, first_valid=0, selected_alpha=math.nan)


def _input_norm_control(inputs: ScoreInputs) -> AnomalyScore:
    """前処理後の入力の絶対値 (D-61)。"""
    values: FloatArray = np.abs(inputs.values[:, 0]).astype(np.float64, copy=True)
    return AnomalyScore(values=values, first_valid=0, selected_alpha=math.nan)


def build_score(spec: ScoreSpec, inputs: ScoreInputs) -> AnomalyScore:
    """spec から異常スコアを作る (**6系統共通の唯一の入口**)。

    Args:
        spec: ``build_score_specs`` が返した spec のいずれか。
        inputs: 1レプリケートで全系統が共有する材料。

    Returns:
        ``AnomalyScore``。``first_valid`` は ``score_first_valid(spec)`` と
        必ず一致する (予測経路と実経路が同じ値を返すことを構造で保証する)。
    """
    match spec:
        case RidgeResidualSpec():
            return _ridge_residual(spec, inputs)
        case PersistenceSpec():
            return _persistence_residual(inputs)
        case MovingStatisticsSpec():
            return _moving_statistics(spec, inputs)
        case RandomControlSpec():
            return _random_control(inputs)
        case InputNormControlSpec():
            return _input_norm_control(inputs)


def smooth_score(score: AnomalyScore, score_smoothing: int) -> AnomalyScore:
    """スコアに後方移動平均を掛ける (**全系統に同じ窓**)。

    平滑化は「異常が数点にしか出ない残差」を区間として見えるようにする常套
    手段だが、系統ごとに窓を変えられると「平滑化した手法が強く見える」比較が
    作れてしまう。窓は ``preprocess.score_smoothing`` の1本だけで、
    ``first_valid`` は全系統そろって ``window - 1`` 行ぶん後ろへ動く。

    Args:
        score: 平滑化前のスコア。
        score_smoothing: 窓 [点]。``1`` なら何もしない。

    Raises:
        ValueError: 窓が 1 未満、または有効行が窓より短い場合。
    """
    shift = smoothing_shift(score_smoothing)
    if shift == 0:
        return score
    valid = score.values[score.first_valid :]
    if valid.size < score_smoothing:
        raise ValueError(
            f"有効なスコア行が平滑化窓より短いです: {valid.size} < {score_smoothing}"
        )
    cumulative: FloatArray = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(valid))
    )
    averaged = (cumulative[score_smoothing:] - cumulative[:-score_smoothing]) / float(
        score_smoothing
    )
    values = _empty_scores(score.values.shape[0])
    first_valid = score.first_valid + shift
    values[first_valid:] = averaged
    return AnomalyScore(
        values=values, first_valid=first_valid, selected_alpha=score.selected_alpha
    )


__all__ = [
    "ANOMALY_METHODS",
    "CONTROL_METHODS",
    "DELAY_LINE_RESIDUAL",
    "ESN_RESIDUAL",
    "INPUT_NORM_CONTROL",
    "MOVING_STATISTICS",
    "PERSISTENCE_RESIDUAL",
    "RANDOM_CONTROL",
    "AnomalyScore",
    "InputNormControlSpec",
    "MovingStatisticsSpec",
    "PersistenceSpec",
    "RandomControlSpec",
    "RidgeResidualSpec",
    "ScoreInputs",
    "ScoreSpec",
    "build_score",
    "build_score_specs",
    "score_first_valid",
    "smooth_score",
    "smoothing_shift",
]
