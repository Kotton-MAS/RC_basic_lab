"""異常検知の評価指標のテスト (D-54 / D-55 / D-62).

主眼は3つ:

1. **AUPRC の自前実装が sklearn と厳密に一致すること** (D-62)。AUPRC は同順位の
   扱いで容易に間違え、間違いは「少し違う値」としてしか現れないため、独立
   オラクルとのランダム入力照合を最強の guard に据える。
2. **階段和であって台形則ではないこと** (D-54)。台形則は値が 0〜1 に収まり
   曲線も滑らかなので、図でも CSV でも壊れて見えない。
3. **PA-F1 が一様乱数対照と同時にしか取れないこと** (D-55)。
"""

from __future__ import annotations

import ast
import dataclasses
import tomllib
from pathlib import Path

import numpy as np
import pytest
from ast_imports import imported_roots
from sklearn.metrics import average_precision_score

from rc_basics_lab import metrics_detection
from rc_basics_lab.metrics_detection import (
    BoolArray,
    MaskedEvaluation,
    PointAdjustReport,
    PointScores,
    PrecisionRecallCurve,
    apply_ignore_mask,
    average_precision,
    point_adjust_report,
    point_scores,
    precision_recall_curve,
    threshold_at_false_alarm_rate,
)
from rc_basics_lab.types import FloatArray

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
MODULE_PATH = REPO_ROOT / "src" / "rc_basics_lab" / "metrics_detection.py"

N_SKLEARN_CASES = 1000
"""sklearn と照合するランダム入力の件数 (仕様 §4 T1 受け入れ基準1)。"""


def _random_case(rng: np.random.Generator) -> tuple[BoolArray, FloatArray]:
    """``n`` 50〜500・異常率 1〜20%・**同順位を必ず含む**1ケースを作る。

    スコアは ``max(2, n // 10)`` 段の整数格子から引く。抽選数 ``n`` が段数を
    必ず上回るので、鳩の巣原理から同順位が必ず生じる。
    """
    n = int(rng.integers(50, 501))
    rate = float(rng.uniform(0.01, 0.20))
    n_positive = max(1, round(rate * n))
    labels = np.zeros(n, dtype=np.bool_)
    labels[rng.choice(n, size=n_positive, replace=False)] = True
    n_levels = max(2, n // 10)
    scores = rng.integers(0, n_levels, size=n).astype(np.float64)
    return labels, scores


def test_matches_scikit_learn_average_precision_on_random_inputs() -> None:
    """ランダム入力 1000 ケースで sklearn と ``rtol=1e-12`` 一致する (D-62 の実体)。

    同順位 (同スコア) を1つの閾値に畳む規則が sklearn とずれると、値は
    「もっともらしいが少し違う」形にしかならないので、この照合が唯一の検出手段。
    """
    rng = np.random.default_rng(20260820)
    worst_relative_difference = 0.0
    for _ in range(N_SKLEARN_CASES):
        labels, scores = _random_case(rng)
        assert np.unique(scores).size < scores.size, "同順位を含まないケースが出た"
        ours = average_precision(labels, scores)
        reference = float(average_precision_score(labels, scores))
        assert ours == pytest.approx(reference, rel=1e-12)
        worst_relative_difference = max(
            worst_relative_difference, abs(ours - reference) / abs(reference)
        )
    # 実測 (2026-08-20): 最悪相対差 5.50e-16 (= 数 ulp)。許容 1e-12 に対し4桁の余裕。
    assert worst_relative_difference < 1e-13


def _trapezoid_average_precision(curve: PrecisionRecallCurve) -> float:
    """PR 曲線を台形則で積分する (**採用しない方の計算**。D-54 の対照)。

    ``(recall=0, precision=1)`` を端点として補い、隣接点の適合率を線形補間する。
    `average_precision` をこの式に差し替えると、本ファイルの guard が落ちる。
    """
    recall: FloatArray = np.concatenate((np.zeros(1, dtype=np.float64), curve.recall))
    precision: FloatArray = np.concatenate(
        (np.ones(1, dtype=np.float64), curve.precision)
    )
    return float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2.0))


def test_average_precision_is_the_step_sum_not_the_trapezoid() -> None:
    """台形則の方が大きくなる具体例を実測で固定する (D-54 の guard test)。

    最上位スコアが同順位 (陽性1点 + 陰性1点) の場合、最初の recall 上昇区間で
    台形則は端点 ``precision = 1`` との平均を取るため、階段和より必ず楽観側へ
    ずれる。``average_precision`` を台形則に差し替えると、下の厳密不等号と
    リテラル固定の両方が落ちる。
    """
    labels = np.array([True, False, True, False])
    scores = np.array([1.0, 1.0, 0.5, 0.4])
    curve = precision_recall_curve(labels, scores)

    step_sum = average_precision(labels, scores)
    trapezoid = _trapezoid_average_precision(curve)

    # 実測 (2026-08-20): 階段和 7/12 = 0.5833..., 台形則 2/3 = 0.6666...
    assert step_sum == pytest.approx(7.0 / 12.0, rel=1e-12)
    assert trapezoid == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert trapezoid > step_sum
    assert trapezoid - step_sum == pytest.approx(1.0 / 12.0, rel=1e-12)
    # 正本は sklearn 側 (階段和) であることも同時に固定する。
    assert step_sum == pytest.approx(
        float(average_precision_score(labels, scores)), rel=1e-12
    )


def test_uniform_random_scores_have_average_precision_at_the_anomaly_rate() -> None:
    """一様乱数スコアの AP は異常率に収束する (仕様 §4 T1 受け入れ基準3)。

    「AUPRC が point-adjust を通っていない」ことの証拠として 05 全体で使う性質
    (D-61)。PA を通すと乱数対照でも 0.9 を超えるため、この収束は成立しなくなる。
    """
    rng = np.random.default_rng(0)
    n = 200_000
    anomaly_rate = 0.05
    labels = rng.random(n) < anomaly_rate
    scores = rng.random(n)

    value = average_precision(labels, scores)
    # 実測 (2026-08-20, seed=0): AP = 0.050114、実現異常率 0.049475、差 6.39e-04。
    assert abs(value - float(np.mean(labels))) < 0.01
    assert abs(value - anomaly_rate) < 0.01


def _segmented_labels(n_segments: int, segment_length: int, rate: float) -> BoolArray:
    """等間隔に ``n_segments`` 本の異常区間を置いたラベル列を作る。"""
    total = round(n_segments * segment_length / rate)
    labels = np.zeros(total, dtype=np.bool_)
    stride = total // n_segments
    for index in range(n_segments):
        start = index * stride + (stride - segment_length) // 2
        labels[start : start + segment_length] = True
    return labels


SWAT_SEGMENT_LENGTH = 1500
"""SWaT 相当の異常区間長。

実測: SWaT のテスト区間は 449,919 点・異常率 11.98%・攻撃 35 本なので
平均区間長は約 1,540 点。仕様 §4 T1 は「区間長 100 / 異常率 5%」と書いていたが、
その条件では PA-F1 の乱数対照は**原理的に 0.9 に届かない**
(`test_point_adjust_random_control_is_weaker_on_short_segments` が実測で固定)。
SWaT 相当を名乗る以上、SWaT の実際の形状を使う (§4 T1 の実装メモに追記済み)。
"""

SWAT_ANOMALY_RATE = 0.12
"""SWaT 相当の異常率 (実測 11.98% を丸めた値)。"""


def test_point_adjust_is_never_reported_without_the_random_control() -> None:
    """PA-F1 は一様乱数対照と同時にしか取得できない (D-55 の型側の強制)。

    型の検査と、対照が実際に高く出ることの実測を1本にまとめる。
    「対照を並べれば読者が気づける」という主張は、対照が本当に高いときにしか
    成り立たないため、両方を同じテストで固定する。
    """
    # (a) 型: point-adjust に関わる公開名は2つだけで、関数は報告型しか返さない。
    adjust_names = sorted(
        name
        for name in metrics_detection.__all__
        if "adjust" in name.lower() or "pa_f1" in name.lower()
    )
    assert adjust_names == ["PointAdjustReport", "point_adjust_report"]
    field_names = {f.name for f in dataclasses.fields(PointAdjustReport)}
    assert field_names == {"pa_f1", "pa_f1_random", "k"}
    assert point_adjust_report.__annotations__["return"] == "PointAdjustReport"

    # (b) 実測: SWaT 相当の合成条件で乱数対照の PA-F1 が 0.9 を超える。
    labels = _segmented_labels(16, SWAT_SEGMENT_LENGTH, SWAT_ANOMALY_RATE)
    n = labels.size
    n_alarms = int(0.003 * n)
    measured: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        control_scores = rng.random(n)
        detector_scores = rng.random(n)
        predictions = np.zeros(n, dtype=np.bool_)
        predictions[np.argsort(detector_scores)[::-1][:n_alarms]] = True
        report = point_adjust_report(labels, predictions, control_scores, k=0.0)
        assert report.k == 0.0
        measured.append(report.pa_f1_random)

    # 実測 (2026-08-20, n=200,000 / 区間長 1,500 x 16 本 / 警報率 0.3%):
    # seed 0..4 で 0.9570 / 0.9893 / 0.9891 / 0.9567 / 0.9567。
    # Kim et al. AAAI 2022 が SWaT で報告した乱数の F1_PA = 0.969 と同水準。
    assert min(measured) > 0.9
    assert min(measured) == pytest.approx(0.9567, abs=5e-3)


def test_point_adjust_random_control_is_weaker_on_short_segments() -> None:
    """区間長 100 / 異常率 5% では乱数対照の PA-F1 が 0.9 に届かない (実測で固定)。

    仕様 §4 T1 受け入れ基準4 が「異常区間長 100・異常率 5% で
    ``pa_f1_random > 0.9``」と書いていた条件そのものを測る。PA の乱数対照の
    強さは**区間長**で決まり、区間が短いほど 1 点でも当てるための警報予算が
    大きくなって適合率が落ちる。SWaT の区間長が約 1,540 点であることが
    Kim et al. の 0.969 の正体なので、区間長 100 を「SWaT 相当」と呼べない。
    """
    labels = _segmented_labels(100, 100, 0.05)
    n = labels.size
    rng = np.random.default_rng(0)
    control_scores = rng.random(n)

    best = 0.0
    for alarm_rate in np.linspace(0.001, 0.10, 40):
        predictions = np.zeros(n, dtype=np.bool_)
        predictions[: int(alarm_rate * n)] = True
        report = point_adjust_report(labels, predictions, control_scores, k=0.0)
        best = max(best, report.pa_f1_random)

    # 実測 (2026-08-20, seed=0): 警報予算 0.1%〜10% を掃引した最大が 0.7646。
    assert best == pytest.approx(0.7646, abs=5e-3)
    assert best < 0.9


def test_runtime_dependencies_do_not_include_scikit_learn() -> None:
    """scikit-learn は dev グループにのみ置く (D-62 の guard test)。

    実行時依存は matplotlib / numpy / pyyaml / scipy の4つに保つ (D-10 と同じ
    規律)。sklearn はテストのオラクルであって製品の一部ではない。
    """
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    runtime = [
        requirement.split(";")[0]
        .split("[")[0]
        .split("==")[0]
        .split(">=")[0]
        .split("<")[0]
        .split("~=")[0]
        .strip()
        .lower()
        .replace("_", "-")
        for requirement in pyproject["project"]["dependencies"]
    ]
    assert "scikit-learn" not in runtime
    assert "sklearn" not in runtime
    assert sorted(runtime) == ["matplotlib", "numpy", "pyyaml", "scipy"]

    dev_group = pyproject["dependency-groups"]["dev"]
    assert any(entry.lower().startswith("scikit-learn") for entry in dev_group), (
        "オラクルとして使う以上、dev グループには入っていなければならない"
    )


ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "dataclasses", "numpy", "rc_basics_lab"}
)
"""`metrics_detection.py` が import してよいトップレベル名 (D-59)。"""


def test_metrics_detection_performs_no_io() -> None:
    """純関数層である: I/O も乱数源も持たない (D-59)。

    T2 が `tests/test_layer_boundaries.py` に `tasks/` 込みの AST 走査を足すが、
    このモジュール単体の性質は指標層のテストでも固定しておく。
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        root.split(".")[0]
        for root in imported_roots(MODULE_PATH, include_function_bodies=True)
    }
    assert imported <= ALLOWED_IMPORT_ROOTS, f"想定外の import: {imported}"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called
    assert "eval" not in called
    assert "exec" not in called


def test_public_api_snapshot() -> None:
    """公開名の一覧を固定する (増減を無言で通さない)。"""
    assert sorted(metrics_detection.__all__) == [
        "BoolArray",
        "MaskedEvaluation",
        "PointAdjustReport",
        "PointScores",
        "PrecisionRecallCurve",
        "apply_ignore_mask",
        "average_precision",
        "point_adjust_report",
        "point_scores",
        "precision_recall_curve",
        "threshold_at_false_alarm_rate",
    ]


# --- PR 曲線 ---------------------------------------------------------------


def test_precision_recall_curve_is_the_single_source_of_average_precision() -> None:
    """図・CSV・AP が同じ1本の点列から出る (2実装に割れる経路を作らない)。"""
    rng = np.random.default_rng(7)
    labels, scores = _random_case(rng)
    curve = precision_recall_curve(labels, scores)
    assert curve.average_precision() == pytest.approx(
        average_precision(labels, scores), rel=1e-15
    )
    assert curve.n_points == curve.threshold.size


def test_tied_scores_collapse_into_one_threshold() -> None:
    """同順位は1つの閾値に畳む (点数 = 相異なるスコアの個数)。"""
    labels = np.array([True, False, True, False, True])
    scores = np.array([2.0, 2.0, 1.0, 1.0, 1.0])
    curve = precision_recall_curve(labels, scores)
    assert curve.n_points == 2
    assert curve.threshold.tolist() == [2.0, 1.0]
    assert curve.precision.tolist() == pytest.approx([0.5, 0.6])
    assert curve.recall.tolist() == pytest.approx([1.0 / 3.0, 1.0])


def test_precision_recall_curve_is_ordered_by_decreasing_threshold() -> None:
    """閾値は降順・recall は非減少 (作図がそのまま左から右へ描ける)。"""
    rng = np.random.default_rng(11)
    labels, scores = _random_case(rng)
    curve = precision_recall_curve(labels, scores)
    assert np.all(np.diff(curve.threshold) < 0.0)
    assert np.all(np.diff(curve.recall) >= 0.0)
    assert curve.recall[-1] == pytest.approx(1.0)


def test_average_precision_is_invariant_under_monotone_score_transforms() -> None:
    """単調増加変換でスコアを潰しても AP は変わらない (順位だけの指標)。"""
    rng = np.random.default_rng(3)
    labels, scores = _random_case(rng)
    transformed = np.expm1(scores / 10.0)
    assert average_precision(labels, transformed) == pytest.approx(
        average_precision(labels, scores), rel=1e-12
    )


def test_average_precision_of_a_perfect_ranking_is_one() -> None:
    labels = np.array([True, True, False, False, False])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert average_precision(labels, scores) == pytest.approx(1.0, rel=1e-15)


def test_average_precision_matches_a_hand_computed_case() -> None:
    """手計算 ``1/2 + (2/3)/2 = 5/6`` と一致する。"""
    labels = np.array([True, False, True])
    scores = np.array([3.0, 2.0, 1.0])
    assert average_precision(labels, scores) == pytest.approx(5.0 / 6.0, rel=1e-12)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        (np.zeros(3, dtype=np.bool_), np.zeros(4), "形状"),
        (np.zeros(0, dtype=np.bool_), np.zeros(0), "空"),
        (np.array([True, False]), np.array([np.nan, 1.0]), "非有限"),
        (np.zeros(3, dtype=np.bool_), np.zeros(3), "陽性"),
    ],
)
def test_average_precision_rejects_ill_posed_inputs(
    labels: BoolArray, scores: FloatArray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        average_precision(labels, scores)


def test_precision_recall_curve_rejects_two_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="1次元"):
        precision_recall_curve(np.ones((2, 2), dtype=np.bool_), np.ones((2, 2)))


# --- 点単位の指標 -----------------------------------------------------------


def test_point_scores_match_a_hand_computed_case() -> None:
    labels = np.array([True, True, False, False])
    predictions = np.array([True, False, True, False])
    result = point_scores(labels, predictions)
    assert result == PointScores(precision=0.5, recall=0.5, f1=0.5)


def test_point_scores_are_zero_when_nothing_is_predicted() -> None:
    """閾値掃引の端では陽性ゼロが必ず出る。例外にせず 0.0 を返す。"""
    labels = np.array([True, False, True])
    result = point_scores(labels, np.zeros(3, dtype=np.bool_))
    assert result == PointScores(precision=0.0, recall=0.0, f1=0.0)


def test_point_scores_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="形状"):
        point_scores(np.zeros(3, dtype=np.bool_), np.zeros(4, dtype=np.bool_))


def test_point_scores_rejects_a_two_dimensional_input() -> None:
    """``_as_mask_pair`` の1次元検査 (reviewer-test 指摘)。"""
    with pytest.raises(ValueError, match="1次元"):
        point_scores(np.zeros((2, 2), dtype=np.bool_), np.zeros((2, 2), dtype=np.bool_))


def test_point_scores_rejects_an_empty_series() -> None:
    """``_as_mask_pair`` の空配列検査 (reviewer-test 指摘)。

    ``average_precision`` 側の同種チェック (``_as_labeled_scores``) はテスト
    済みだったが、``point_scores``/``point_adjust_report`` が経由する
    ``_as_mask_pair`` の空配列分岐は測っていなかった (非対称)。
    """
    with pytest.raises(ValueError, match="空の系列"):
        point_scores(np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_))


def test_point_adjust_report_rejects_an_empty_series() -> None:
    """``point_adjust_report`` も同じ ``_as_mask_pair`` を経由する。"""
    with pytest.raises(ValueError, match="空の系列"):
        point_adjust_report(
            np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_), np.zeros(0)
        )


# --- point-adjust / PA%K ----------------------------------------------------


def _pa_fixture() -> tuple[BoolArray, BoolArray, FloatArray]:
    """異常区間 4 点 x 2 本、うち 1 本を 1 点だけ当てた予測。"""
    labels = np.zeros(20, dtype=np.bool_)
    labels[4:8] = True
    labels[14:18] = True
    predictions = np.zeros(20, dtype=np.bool_)
    predictions[5] = True  # 1本目を1点だけ検知
    predictions[14:17] = True  # 2本目を3点検知
    predictions[0] = True  # 誤報1点
    control = np.linspace(0.0, 1.0, 20)
    return labels, predictions, control


def test_point_adjust_at_k_zero_credits_the_whole_segment() -> None:
    """K=0 は従来の point-adjust (1点でも当たれば区間全部が当たり扱い)。"""
    labels, predictions, control = _pa_fixture()
    report = point_adjust_report(labels, predictions, control, k=0.0)
    # 調整後: TP=8, FP=1, FN=0 -> P=8/9, R=1, F1=16/17
    assert report.pa_f1 == pytest.approx(16.0 / 17.0, rel=1e-12)


def test_point_adjust_at_k_hundred_is_the_plain_f1() -> None:
    """K=100 は素の点単位 F1 と一致する (検知率が 1 を超えないため無調整)。"""
    labels, predictions, control = _pa_fixture()
    report = point_adjust_report(labels, predictions, control, k=100.0)
    assert report.pa_f1 == pytest.approx(
        point_scores(labels, predictions).f1, rel=1e-15
    )
    assert report.pa_f1 < point_adjust_report(labels, predictions, control, k=0.0).pa_f1


def test_point_adjust_at_intermediate_k_adjusts_only_well_detected_segments() -> None:
    """K=50 では検知率 3/4 の区間だけが調整され、1/4 の区間は素のまま。"""
    labels, predictions, control = _pa_fixture()
    report = point_adjust_report(labels, predictions, control, k=50.0)
    # 調整後: 2本目のみ全点 True -> TP=1+4=5, FP=1, FN=3 -> P=5/6, R=5/8
    assert report.pa_f1 == pytest.approx(2 * 5 / (2 * 5 + 1 + 3), rel=1e-12)
    assert report.k == 50.0


def test_point_adjust_f1_is_non_increasing_in_k() -> None:
    """K を上げるほど甘さが減る (PA%K の意味そのもの)。"""
    labels, predictions, control = _pa_fixture()
    values = [
        point_adjust_report(labels, predictions, control, k=k).pa_f1
        for k in (0.0, 25.0, 50.0, 75.0, 100.0)
    ]
    assert values == sorted(values, reverse=True)


def test_point_adjust_random_control_uses_the_same_alarm_budget() -> None:
    """対照は評価対象と**同じ警報数**で切る (予算差による逃げ道を塞ぐ)。"""
    labels = _segmented_labels(4, 50, 0.10)
    n = labels.size
    rng = np.random.default_rng(5)
    control = rng.random(n)
    predictions = np.zeros(n, dtype=np.bool_)
    predictions[:17] = True
    report = point_adjust_report(labels, predictions, control, k=0.0)
    n_control_alarms = int(np.count_nonzero(control >= np.sort(control)[-17]))
    assert n_control_alarms == 17
    assert 0.0 <= report.pa_f1_random <= 1.0


@pytest.mark.parametrize("k", [-0.1, 100.1])
def test_point_adjust_rejects_out_of_range_k(k: float) -> None:
    labels, predictions, control = _pa_fixture()
    with pytest.raises(ValueError, match="百分率"):
        point_adjust_report(labels, predictions, control, k=k)


def test_point_adjust_rejects_control_of_a_different_length() -> None:
    labels, predictions, _ = _pa_fixture()
    with pytest.raises(ValueError, match="control_scores"):
        point_adjust_report(labels, predictions, np.zeros(3))


def test_point_adjust_report_with_no_alarms_gives_zero_f1() -> None:
    """``_top_alarm_mask`` の ``n_alarms <= 0`` 分岐 (reviewer-test 指摘)。

    ``predictions`` が全 False (警報ゼロ) のとき、対照も ``n_alarms=0`` で
    警報ゼロになる (249行目の早期 return)。両方とも TP=0 なので pa_f1 /
    pa_f1_random はともに 0.0 になる。
    """
    labels, _, control = _pa_fixture()
    predictions = np.zeros(labels.size, dtype=np.bool_)
    report = point_adjust_report(labels, predictions, control, k=0.0)
    assert report.pa_f1 == 0.0
    assert report.pa_f1_random == 0.0


# --- 固定誤報率からの閾値 ---------------------------------------------------


def test_threshold_at_false_alarm_rate_hits_the_requested_alarm_count() -> None:
    """較正区間で ``floor(rate * n)`` 点だけが警報になる。"""
    rng = np.random.default_rng(2)
    calibration = rng.random(1000)
    threshold = threshold_at_false_alarm_rate(calibration, 0.05)
    assert int(np.count_nonzero(calibration >= threshold)) == 50


def test_threshold_at_false_alarm_rate_is_monotone_in_the_rate() -> None:
    rng = np.random.default_rng(4)
    calibration = rng.random(1000)
    thresholds = [
        threshold_at_false_alarm_rate(calibration, rate)
        for rate in (0.01, 0.05, 0.10, 0.50)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_threshold_at_false_alarm_rate_never_returns_an_unreachable_value() -> None:
    """``rate * n < 1`` でも警報 1 点ぶんの閾値 (= 最大値) を返す。"""
    calibration = np.array([0.1, 0.5, 0.9])
    assert threshold_at_false_alarm_rate(calibration, 0.001) == pytest.approx(0.9)


def test_threshold_at_false_alarm_rate_does_not_take_labels() -> None:
    """署名にラベルが無い (D-56 を型側で強制する下地)。"""
    assert set(threshold_at_false_alarm_rate.__annotations__) == {
        "calibration_scores",
        "target_false_alarm_rate",
        "return",
    }


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_threshold_at_false_alarm_rate_rejects_out_of_range_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="target_false_alarm_rate"):
        threshold_at_false_alarm_rate(np.linspace(0.0, 1.0, 10), rate)


def test_threshold_at_false_alarm_rate_rejects_an_empty_calibration_window() -> None:
    with pytest.raises(ValueError, match="空"):
        threshold_at_false_alarm_rate(np.zeros(0), 0.05)


def test_threshold_at_false_alarm_rate_rejects_a_two_dimensional_input() -> None:
    """``calibration_scores`` の2次元入力検査 (reviewer-test 指摘)。"""
    with pytest.raises(ValueError, match="1次元"):
        threshold_at_false_alarm_rate(np.zeros((10, 2)), 0.05)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_threshold_at_false_alarm_rate_rejects_non_finite_values(value: float) -> None:
    """docstring が明記する非有限値の検査 (reviewer-test 指摘)。

    docstring の Raises には非有限値で ``ValueError`` になると明記されているが
    検証するテストが無かった。
    """
    calibration = np.array([0.1, 0.5, value, 0.9])
    with pytest.raises(ValueError, match="非有限"):
        threshold_at_false_alarm_rate(calibration, 0.05)


# --- is_ignored マスク ------------------------------------------------------


def test_apply_ignore_mask_drops_the_ignored_points() -> None:
    labels = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    ignored = np.array([False, True, False, True])
    masked = apply_ignore_mask(labels, scores, ignored)
    assert isinstance(masked, MaskedEvaluation)
    assert masked.labels.tolist() == [True, True]
    assert masked.scores.tolist() == [0.9, 0.7]


def test_apply_ignore_mask_changes_the_average_precision() -> None:
    """遷移点を落とすと評価点数が変わり AP も変わる (マスクが効いている証拠)。"""
    rng = np.random.default_rng(9)
    labels, scores = _random_case(rng)
    ignored = np.zeros(labels.size, dtype=np.bool_)
    ignored[: labels.size // 4] = True
    masked = apply_ignore_mask(labels, scores, ignored)
    assert masked.labels.size == labels.size - labels.size // 4
    assert average_precision(masked.labels, masked.scores) != average_precision(
        labels, scores
    )


def test_apply_ignore_mask_rejects_a_fully_ignored_series() -> None:
    with pytest.raises(ValueError, match="全点"):
        apply_ignore_mask(
            np.array([True, False]), np.array([1.0, 0.0]), np.array([True, True])
        )


def test_apply_ignore_mask_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="is_ignored"):
        apply_ignore_mask(
            np.array([True, False]), np.array([1.0, 0.0]), np.zeros(3, dtype=np.bool_)
        )
