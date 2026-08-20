"""異常検知の課題層 (``tasks/anomaly.py``) の検査 (D-57 / D-59 / D-60).

このファイルは**ネットワークにもファイルにも触れない** —— 唯一の例外が
``tasks/anomaly.py`` のソースを AST で読む構造検査で、これは「係数を作れる
場所が1箇所しかない」ことを実装の形から確かめるためのものである。
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from rc_basics_lab.config import MackeyGlassConfig, SyntheticAnomalyConfig
from rc_basics_lab.tasks.anomaly import (
    NORMALIZE_METHODS,
    AnomalyPreprocessor,
    AnomalySeries,
    generate_synthetic_anomalies,
)
from rc_basics_lab.types import FloatArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rc_basics_lab"
MODULE_PATH = SRC / "tasks" / "anomaly.py"

SMALL = SyntheticAnomalyConfig(
    length=4000,
    n_anomalies=3,
    segment_length=60,
    ignore_margin=20,
    mackey_glass=MackeyGlassConfig(),
)
"""テスト用の小さい設定 (既定より 5 倍速い。構造は同じ)。"""


def _series(length: int = 200) -> AnomalySeries:
    """検証用の最小の ``AnomalySeries``。"""
    values = np.linspace(0.0, 1.0, length).reshape(-1, 1)
    labels = np.zeros(length, dtype=np.bool_)
    labels[150:160] = True
    ignore = np.zeros(length, dtype=np.bool_)
    ignore[145:150] = True
    return AnomalySeries(
        values=values, labels=labels, ignore=ignore, train_end=100, name="probe"
    )


# --- AnomalySeries の不変条件 -----------------------------------------------


def test_anomaly_series_exposes_length_count_and_rate() -> None:
    """派生量 (系列長 / 異常区間の個数 / 異常点の割合) が器から取れる。"""
    series = _series()
    assert series.n_steps == 200
    assert series.n_anomalies == 1
    assert series.anomaly_rate == pytest.approx(10 / 200)


def test_anomaly_series_counts_segments_not_points() -> None:
    """``n_anomalies`` は**区間**の数 (点数と取り違えない)。"""
    labels = np.zeros(50, dtype=np.bool_)
    labels[10:13] = True
    labels[20:25] = True
    series = AnomalySeries(
        values=np.zeros((50, 1)),
        labels=labels,
        ignore=np.zeros(50, dtype=np.bool_),
        train_end=5,
        name="two",
    )
    assert series.n_anomalies == 2
    assert int(np.count_nonzero(series.labels)) == 8


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.zeros(10), "2次元"),
        (np.zeros((10, 2)), "単変量"),
        (np.full((10, 1), np.nan), "有限"),
    ],
)
def test_anomaly_series_rejects_malformed_values(
    values: FloatArray, message: str
) -> None:
    """形状・次元・有限性を器の側で落とす (``TaskData`` と同じ流儀)。"""
    labels = np.zeros(10, dtype=np.bool_)
    labels[8] = True
    with pytest.raises(ValueError, match=message):
        AnomalySeries(
            values=values,
            labels=labels,
            ignore=np.zeros(10, dtype=np.bool_),
            train_end=5,
            name="bad",
        )


def test_anomaly_series_rejects_non_boolean_masks() -> None:
    """``labels`` / ``ignore`` は bool 配列でなければならない。

    0/1 の int を黙って受けると「``labels`` に確率値を入れた」が通ってしまう。
    """
    with pytest.raises(ValueError, match="bool"):
        AnomalySeries(
            values=np.zeros((10, 1)),
            labels=np.ones(10, dtype=np.int64),
            ignore=np.zeros(10, dtype=np.bool_),
            train_end=5,
            name="int-labels",
        )


def test_anomaly_series_rejects_an_anomaly_inside_the_training_prefix() -> None:
    """``train_end`` より手前に異常があれば器が落ちる (受け入れ基準5 の不変条件)。

    源ごとにテストで確かめる形にすると、次に足した源が静かに破る。
    """
    labels = np.zeros(20, dtype=np.bool_)
    labels[3] = True
    with pytest.raises(ValueError, match="train_end"):
        AnomalySeries(
            values=np.zeros((20, 1)),
            labels=labels,
            ignore=np.zeros(20, dtype=np.bool_),
            train_end=10,
            name="leaky",
        )


def test_anomaly_series_rejects_a_series_without_any_anomaly() -> None:
    """陽性が1点も無い系列は ``ValueError`` (AUPRC が定義されない)。

    ``metrics_detection.average_precision`` が同じ規律で落ちる (T1 の決定5)。
    """
    with pytest.raises(ValueError, match="異常が1点も"):
        AnomalySeries(
            values=np.zeros((20, 1)),
            labels=np.zeros(20, dtype=np.bool_),
            ignore=np.zeros(20, dtype=np.bool_),
            train_end=10,
            name="clean",
        )


# --- 前処理の共通化 (D-57) ---------------------------------------------------


def _two_regime_series(n_train: int = 500, n_test: int = 500) -> FloatArray:
    """前半と後半で平均も分散も全く違う系列 (再推定を見破るための材料)。"""
    rng = np.random.default_rng(0)
    head = rng.normal(0.0, 1.0, n_train)
    tail = rng.normal(50.0, 10.0, n_test)
    return np.concatenate([head, tail]).reshape(-1, 1)


@pytest.mark.parametrize("normalize", NORMALIZE_METHODS)
def test_all_methods_share_one_preprocessor_fitted_on_training_prefix(
    normalize: str,
) -> None:
    """係数は訓練区間の先頭から作った1組だけで、全手法・全区間に配られる (D-57)。

    2つを同時に測る:

    1. **``from_training_prefix`` 以外に係数を作る経路が存在しない** ——
       ``src/`` 全体を AST で走査し、``AnomalyPreprocessor`` を構築する呼び出しが
       ``tasks/anomaly.py`` の ``from_training_prefix`` の中にしか無いこと、
       および ``AnomalyPreprocessor`` を返す関数が他に無いことを確かめる
    2. **係数がテスト区間から再推定されていない** —— 前半と後半で平均も分散も
       違う系列を作り、後半に ``apply`` した結果が「先頭から推定した係数」で
       説明できること (再推定していれば後半は中心 0・尺度 1 付近に化ける)

    異常検知でテスト区間から推定すると、その区間の**異常が尺度に入る**ため
    異常が「正常なばらつき」として吸収される。これが D-57 の根拠である。
    """
    series = _two_regime_series()
    n_train = 500
    preprocessor = AnomalyPreprocessor.from_training_prefix(series, n_train, normalize)

    # 1. 構造: 係数を作る呼び出しは1箇所しかない
    constructors = _constructor_call_sites()
    assert constructors == {("tasks/anomaly.py", "from_training_prefix")}, (
        "AnomalyPreprocessor を構築している場所が from_training_prefix 以外に "
        f"あります (D-57): {sorted(constructors)}"
    )
    assert _functions_returning_preprocessor() == {"from_training_prefix"}, (
        "AnomalyPreprocessor を返す関数が from_training_prefix 以外にあります"
    )

    # 2. 値: テスト区間の統計から作り直していない
    transformed = preprocessor.apply(series)
    expected = (series - preprocessor.center) / preprocessor.scale
    assert np.allclose(transformed, expected, rtol=0.0, atol=0.0)

    refitted = AnomalyPreprocessor.from_training_prefix(
        series[n_train:], series.shape[0] - n_train, normalize
    )
    tail = transformed[n_train:]
    if normalize == "none":
        assert np.allclose(preprocessor.center, 0.0)
        assert np.allclose(preprocessor.scale, 1.0)
    else:
        assert not np.allclose(preprocessor.center, refitted.center)
        assert not np.allclose(preprocessor.scale, refitted.scale)
        # 再推定していれば後半は中心 0 付近に来る。実際は遠く離れている。
        assert abs(float(np.mean(tail))) > 1.0

    # 3. 手法が何本あっても値として同じものを配る (インスタンスが1つ)
    segments = (series[:n_train], series[n_train:], series)
    for segment in segments:
        first = preprocessor.apply(segment)
        second = preprocessor.apply(segment)
        assert np.array_equal(first, second)


def _constructor_call_sites() -> set[tuple[str, str]]:
    """``src/`` 全体で ``AnomalyPreprocessor`` を構築している (ファイル, 関数)。

    ``from_training_prefix`` は ``cls(...)`` で構築するので、
    ``AnomalyPreprocessor`` クラス本体の中の ``cls(...)`` も構築とみなす。
    """
    sites: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(SRC).as_posix()
        for function, call_names in _calls_by_function(tree):
            constructs = "AnomalyPreprocessor" in call_names or (
                "cls" in call_names and _is_preprocessor_method(tree, function)
            )
            if constructs:
                sites.add((relative, function))
    return sites


def _calls_by_function(tree: ast.Module) -> list[tuple[str, set[str]]]:
    """関数ごとに「``Name`` で呼ばれた名前の集合」を返す。"""
    result: list[tuple[str, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        names = {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        result.append((node.name, names))
    return result


def _is_preprocessor_method(tree: ast.Module, function_name: str) -> bool:
    """``function_name`` が ``AnomalyPreprocessor`` のメソッドか。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AnomalyPreprocessor":
            return any(
                isinstance(child, ast.FunctionDef) and child.name == function_name
                for child in node.body
            )
    return False


def _functions_returning_preprocessor() -> set[str]:
    """戻り値注釈が ``AnomalyPreprocessor`` の関数名。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        annotation = node.returns
        named = isinstance(annotation, ast.Name) and annotation.id == (
            "AnomalyPreprocessor"
        )
        quoted = (
            isinstance(annotation, ast.Constant)
            and annotation.value == "AnomalyPreprocessor"
        )
        if named or quoted:
            found.add(node.name)
    return found


@pytest.mark.parametrize("normalize", NORMALIZE_METHODS)
def test_preprocessor_round_trips_through_invert(normalize: str) -> None:
    """``invert(apply(x)) == x`` (残差を物理量へ戻す経路が壊れていない)。"""
    series = _two_regime_series()
    preprocessor = AnomalyPreprocessor.from_training_prefix(series, 500, normalize)
    restored = preprocessor.invert(preprocessor.apply(series))
    assert np.allclose(restored, series)


def test_preprocessor_coefficients_match_the_named_statistics() -> None:
    """4方式の係数が名前どおりの統計量である (中身の取り違えを固定する)。"""
    series = np.arange(100, dtype=np.float64).reshape(-1, 1)
    zscore = AnomalyPreprocessor.from_training_prefix(series, 100, "zscore")
    assert zscore.center == pytest.approx(np.mean(series))
    assert zscore.scale == pytest.approx(np.std(series))

    minmax = AnomalyPreprocessor.from_training_prefix(series, 100, "minmax")
    assert minmax.center == pytest.approx(np.min(series))
    assert minmax.scale == pytest.approx(np.max(series) - np.min(series))

    robust = AnomalyPreprocessor.from_training_prefix(series, 100, "robust")
    assert robust.center == pytest.approx(np.median(series))
    assert robust.scale == pytest.approx(
        np.percentile(series, 75.0) - np.percentile(series, 25.0)
    )


def test_preprocessor_rejects_an_unsupported_normalize() -> None:
    """未対応の ``normalize`` は ``ValueError`` (黙って恒等変換にしない)。"""
    series = np.arange(20, dtype=np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match="normalize"):
        AnomalyPreprocessor.from_training_prefix(series, 10, "standardize")


def test_preprocessor_rejects_a_constant_training_prefix() -> None:
    """尺度が 0 になる前処理は ``ValueError`` (``Standardizer`` と同じ規律)。"""
    series = np.ones((50, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="尺度"):
        AnomalyPreprocessor.from_training_prefix(series, 20, "zscore")


def test_preprocessor_rejects_an_out_of_range_prefix_length() -> None:
    """``n_steps`` が範囲外なら ``ValueError``。"""
    series = np.arange(20, dtype=np.float64).reshape(-1, 1)
    with pytest.raises(ValueError, match="n_steps"):
        AnomalyPreprocessor.from_training_prefix(series, 21, "zscore")


# --- 合成源 (MGAB と同じ構造) ------------------------------------------------


def test_synthetic_source_has_the_same_structure_as_mgab() -> None:
    """異常が指定個数・``ignore`` が前後に付き・値は有限・訓練前半は正常。

    仕様 §4 T2 受け入れ基準5 そのもの。実測の MGAB (``1.csv``) は 100,000 点 /
    異常区間 10 本 / 各 401 点 / 異常率 0.0401 で、ここはその縮小版である。
    """
    series = generate_synthetic_anomalies(SMALL, np.random.default_rng(0))
    assert series.n_steps == SMALL.length
    assert series.n_anomalies == SMALL.n_anomalies
    assert np.all(np.isfinite(series.values))
    assert not bool(np.any(series.labels[: series.train_end]))
    assert series.anomaly_rate == pytest.approx(
        SMALL.n_anomalies * SMALL.segment_length / SMALL.length
    )

    labels = np.asarray(series.labels)
    ignore = np.asarray(series.ignore)
    starts = np.flatnonzero(
        np.diff(np.concatenate(([False], labels)).astype(np.int8)) == 1
    )
    ends = np.flatnonzero(
        np.diff(np.concatenate((labels, [False])).astype(np.int8)) == -1
    )
    assert starts.size == SMALL.n_anomalies
    for start, end in zip(starts, ends, strict=True):
        assert bool(ignore[start - 1]), "異常区間の直前が ignore になっていません"
        assert bool(ignore[end + 1]), "異常区間の直後が ignore になっていません"
    assert not bool(np.any(ignore & labels)), "ignore と labels が重なっています"
    assert int(np.count_nonzero(ignore)) == 2 * SMALL.ignore_margin * SMALL.n_anomalies


def test_synthetic_source_splices_without_a_visible_jump() -> None:
    """縫合点で値が飛ばない (MGAB の「目で見つからない異常」の実体)。

    値と微分が一致する2点で切っているので、1次差分の最大値は**元の系列の
    1次差分の範囲を超えない**。ここが崩れると異常がただの段差になり、
    どの手法でも当たってしまう (実験そのものが無意味になる)。
    """
    series = generate_synthetic_anomalies(SMALL, np.random.default_rng(1))
    values = np.asarray(series.values)[:, 0]
    steps = np.abs(np.diff(values))
    labels = np.asarray(series.labels)
    starts = np.flatnonzero(
        np.diff(np.concatenate(([False], labels)).astype(np.int8)) == 1
    )
    quiet = float(np.quantile(steps, 0.999))
    for start in starts:
        splice = int(start) + SMALL.segment_length // 2
        window = steps[max(0, splice - 2) : splice + 2]
        assert float(np.max(window)) <= 4.0 * quiet, (
            f"縫合点 {splice} に段差があります (最大差分 {float(np.max(window))})"
        )


def test_synthetic_source_is_reproducible_and_seed_dependent() -> None:
    """同じ seed で同じ系列、違う seed で違う系列 (D-06 の task ストリーム)。"""
    first = generate_synthetic_anomalies(SMALL, np.random.default_rng(3))
    again = generate_synthetic_anomalies(SMALL, np.random.default_rng(3))
    other = generate_synthetic_anomalies(SMALL, np.random.default_rng(4))
    assert np.array_equal(first.values, again.values)
    assert np.array_equal(first.labels, again.labels)
    assert not np.array_equal(first.values, other.values)


def test_synthetic_source_values_match_a_known_seed_golden_case() -> None:
    """既知 seed に対する出力値そのものを固定する (golden test、reviewer-test 指摘)。

    ``value_scale``/``slope_scale`` を ``_find_cut`` の外へ1回だけ出す変更
    (性能改善) は、``raw``/``derivative`` がループ中に再代入されないため
    数学的には無演算のはずだが、それを固定する回帰テストが無かった。
    ``test_synthetic_source_is_reproducible_and_seed_dependent`` は「同じ
    seed なら同じ出力」という自己無矛盾性しか見ておらず、``value_scale`` /
    ``slope_scale`` の計算対象や計算回数がこの先変わって出力の値そのものが
    変化しても、同じコードで同じ seed を渡す限り依然として green のまま通る
    (自己参照的な検査であり、この回帰クラスには無力)。ここでは既知 seed
    (``12345``) に対する ``values[:5]`` / 全量の sha256 / ``train_end`` を
    実測でハードコードして固定する。
    """
    series = generate_synthetic_anomalies(SMALL, np.random.default_rng(12345))
    values = np.asarray(series.values)[:, 0]

    first_five = values[:5]
    expected_first_five = np.array(
        [
            1.1878190605057366,
            1.195324334947058,
            1.204402731064357,
            1.2171136359115597,
            1.234185802976733,
        ]
    )
    np.testing.assert_allclose(first_five, expected_first_five, rtol=0.0, atol=0.0)

    digest = hashlib.sha256(values.tobytes()).hexdigest()
    assert digest == (
        "297264042ec3d20487e20afc7d269a0af29b27cf2c4f5ef9c7314bbf7f4bbbc4"
    )
    assert series.train_end == 1747


def test_synthetic_source_delegates_to_the_existing_mackey_glass_generator() -> None:
    """Mackey-Glass を再実装していない (``generate_mackey_glass`` へ委譲)。

    積分を2実装に割ると、04 の較正 (D-41) と 05 の合成源が別々の系になる。
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "generate_mackey_glass" in imported
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "rk4" not in source.lower(), "積分器を課題層で再実装しています"


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (SyntheticAnomalyConfig(length=4000, n_anomalies=0), "n_anomalies"),
        (SyntheticAnomalyConfig(length=4000, segment_length=1), "segment_length"),
        (SyntheticAnomalyConfig(length=4000, ignore_margin=-1), "ignore_margin"),
        (SyntheticAnomalyConfig(length=100), "length"),
    ],
)
def test_synthetic_source_rejects_impossible_settings(
    cfg: SyntheticAnomalyConfig, message: str
) -> None:
    """確保する前に設定を落とす (``tasks/mackey_glass.py`` の ``_validate`` と同型)。"""
    with pytest.raises(ValueError, match=message):
        generate_synthetic_anomalies(cfg, np.random.default_rng(0))


def test_find_cut_search_cells_matches_the_measured_formula() -> None:
    """``_find_cut`` の実際の確保サイズ (reviewer-performance の実測式) を固定する。

    既定設定 (``segment_length=200``) では 80,601 要素 (無害) —— reviewer が
    実測した値と一致することを固定する。
    """
    from rc_basics_lab.tasks.anomaly import _find_cut_search_cells

    cfg = SyntheticAnomalyConfig(length=20000, n_anomalies=1, segment_length=200)
    assert _find_cut_search_cells(cfg) == 80_601


@pytest.mark.parametrize("segment_length", [5_000, 20_000, 100_000])
def test_synthetic_source_rejects_a_segment_length_that_would_allocate_too_much(
    segment_length: int,
) -> None:
    """``_find_cut`` の探索行列が大きすぎる設定を確保前に落とす (reviewer-performance)。

    過去に『確保軸の積を検査しないまま巨大配列を確保 -> peak RSS 8.6GB /
    13時間』を起こしたのと同型のガード漏れ —— ``raw_samples`` への線形の上限
    (``_MAX_RAW_SAMPLES``) だけでは、``segment_length`` の**2乗**で増える
    ``starts.size * spans.size`` を捕まえられない。オーケストレータの実測
    (``segment_length=20,000`` で 6.4GB・``100,000`` で 160.0GB 相当) と同じ
    設定を、確保前に ``ValueError`` で落とすことを確認する。
    """
    cfg = SyntheticAnomalyConfig(
        length=segment_length * 10, n_anomalies=1, segment_length=segment_length
    )
    with pytest.raises(ValueError, match="探索行列"):
        generate_synthetic_anomalies(cfg, np.random.default_rng(0))


def test_synthetic_source_needs_no_config_from_the_datasets_layer() -> None:
    """課題層が ``datasets`` を知らない (依存の向きは ``datasets -> tasks``、D-59)。"""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "rc_basics_lab.datasets" not in source
