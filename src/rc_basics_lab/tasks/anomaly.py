"""異常検知の課題層 —— 系列の器・前処理・合成源 (D-57 / D-59 / D-60).

この層は**純関数だけ**である。ネットワークもファイル I/O も持たない
(D-59。``tests/test_layer_boundaries.py::test_tasks_and_metrics_never_perform_io``
が AST で機械検査する)。実データの取得と読み取りは ``datasets/`` にあり、
依存の向きは ``datasets -> tasks`` の一方向。

置くものは3つ:

- ``AnomalySeries``: ラベル付き単変量系列の器。``TaskData`` を継承・拡張せず
  別の dataclass にしてある —— ``TaskData`` は ``(u, y)`` の対で、異常検知には
  ``y`` が無くラベルとマスクが要る。継承すると 01〜04 の全実験が読む器に
  異常検知専用のフィールドが生える
- ``AnomalyPreprocessor``: 手法間で共通の前処理 (D-57)。係数を作れる場所を
  ``from_training_prefix`` **1本**に閉じる (``tasks/chaotic.py`` の
  ``Standardizer`` = D-41 と同じ形)
- ``generate_synthetic_anomalies``: MGAB と同じ手続きの合成源。Mackey-Glass の
  生成は ``tasks/mackey_glass.py`` へ**委譲**する (再実装しない)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from rc_basics_lab.config import SyntheticAnomalyConfig
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.types import BoolArray, FloatArray

TASK_NAME = "synthetic_mackey_glass_anomalies"

NORMALIZE_METHODS: tuple[str, ...] = ("zscore", "minmax", "robust", "none")
"""``AnomalyPreprocessor.from_training_prefix`` が受理する ``normalize`` の4値。

未対応の値は ``ValueError``。文字列を受ける以上、タイプミスを黙って
「標準化しない」に倒すと**前処理が効いていない実験**が緑のまま通る。
"""

FIRST_ANOMALY_FRACTION = 0.3
"""最初の異常を置き始める位置 (系列長に対する比)。

実測の MGAB (``1.csv``) は 100,000 点の系列で最初の異常が 32,518 点目に始まる
(= 0.325)。訓練区間を「異常が1点も無いことが保証された前半」として取れる
だけの長さが要るので、合成源でも同じ程度に取る。
"""

_REMOVED_SPAN_FACTORS = (1, 3)
"""除去するセグメント長の探索範囲を ``segment_length`` の何倍にするか。

独立した設定軸にしない —— 「標識する幅は変えたが除去する幅は変えていない」
条件を作れると、異常の強さと評価の甘さが別々に動いてしまう。
"""

_MAX_RAW_SAMPLES = 4_000_000
"""合成源が Mackey-Glass に要求してよいサンプル数の絶対上限 (確保前に検査)。"""

_MAX_FIND_CUT_CELLS = 5_000_000
"""``_find_cut`` が確保する ``starts x spans`` (``ends``/``cost``) の要素数上限。

``_MAX_RAW_SAMPLES`` は ``raw_samples`` (Mackey-Glass 側の系列長、``length`` に
対して線形の項) だけを検査しており、``_find_cut`` が実際に確保する行列の
要素数 (``starts.size * spans.size``、``segment_length`` の**2乗**のオーダー)
には一切上限が無かった (reviewer-performance 指摘)。既定の ``segment_length=200`` では
``starts.size≈201`` / ``spans.size=401`` で 80,601 要素 (無害) だが、
``segment_length`` だけを大きくすると ``_MAX_RAW_SAMPLES`` の制約下でも
``raw_samples`` は線形にしか増えないため通過してしまい、``cost`` 行列は
容易に数億〜数百億要素 (float64 で数GB〜数百GB) に達する —— 過去に起きた
「確保軸の積を検査しないまま巨大配列を確保 → peak RSS 8.6GB / 13時間」と
同型のガード漏れ。閾値 5,000,000 要素は既定設定に対して60倍以上の余地を
残しつつ、cost/ends 行列を合わせて 100MB 未満に収める。
"""


def _as_series_matrix(series: FloatArray, label: str) -> FloatArray:
    """``(T, D)`` の float64 行列に揃える (1次元は受理しない)。"""
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{label} は (T, D) の2次元配列が必要です: {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} が空です: {array.shape}")
    return array


@dataclass(frozen=True, slots=True)
class AnomalySeries:
    """ラベル付き単変量系列1本ぶん。

    Attributes:
        values: 値 ``(T, 1)``。有限値のみ。
        labels: 異常ラベル ``(T,)`` bool。**主指標 AUPRC の正解**。
        ignore: 評価から除外する点 ``(T,)`` bool。縫合の近傍 (合成源) や
            過渡区間 (MGAB の ``is_ignored``) のように、正常とも異常とも
            言えない点を落とすためのマスク。
        train_end: 学習に使ってよい区間の終端 (**排他的** index)。
            ``values[:train_end]`` が「異常が1点も無いことが保証された前半」で、
            ``labels[:train_end]`` が全て ``False`` であることを検証する。
        name: 系列名 (CSV の ``series`` 列になる)。
        params: 由来を記録するメタ情報 (文字列)。

    Note:
        検証を器の側に置く (``TaskData.__post_init__`` と同じ流儀)。
        「``train_end`` より手前に異常が無い」は**どの源でも成り立つべき不変条件**
        であり、源ごとにテストで確かめる形にすると、次に足した源が静かに破る。
    """

    values: FloatArray
    labels: BoolArray
    ignore: BoolArray
    train_end: int
    name: str
    params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _as_series_matrix(self.values, "values")
        if values.shape[1] != 1:
            raise ValueError(f"values は単変量 (T, 1) が必要です: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"系列 {self.name} の values に有限でない値があります")
        n_steps = values.shape[0]
        for mask, label in ((self.labels, "labels"), (self.ignore, "ignore")):
            array = np.asarray(mask)
            if array.dtype != np.bool_:
                raise ValueError(f"{label} は bool 配列が必要です: {array.dtype}")
            if array.shape != (n_steps,):
                raise ValueError(
                    f"{label} の形状が values と一致しません: "
                    f"{array.shape} != {(n_steps,)}"
                )
        if not 1 <= self.train_end < n_steps:
            raise ValueError(
                "train_end は 1 以上・系列長未満である必要があります: "
                f"train_end={self.train_end}, T={n_steps}"
            )
        if bool(np.any(self.labels[: self.train_end])):
            raise ValueError(
                f"系列 {self.name} は train_end={self.train_end} より手前に異常を"
                "含みます (訓練区間は正常であることが保証されていなければ"
                "なりません)"
            )
        if not bool(np.any(self.labels)):
            raise ValueError(
                f"系列 {self.name} に異常が1点もありません "
                "(AUPRC は陽性が無いと定義されません)"
            )

    @property
    def n_steps(self) -> int:
        """系列長 T。"""
        return int(self.values.shape[0])

    @property
    def n_anomalies(self) -> int:
        """異常**区間**の個数 (点数ではない)。"""
        padded = np.concatenate(([False], np.asarray(self.labels, dtype=np.bool_)))
        return int(np.count_nonzero(np.diff(padded.astype(np.int8)) == 1))

    @property
    def anomaly_rate(self) -> float:
        """異常**点**の割合 (AUPRC の乱数対照が張り付く値)。"""
        return float(np.count_nonzero(self.labels)) / float(self.n_steps)


@dataclass(frozen=True, slots=True)
class AnomalyPreprocessor:
    """手法間で共通の成分ごとアフィン変換 (D-57)。

    **係数を作れる場所を ``from_training_prefix`` 1本に閉じる**。値として
    持ち回れば、手法ごと・区間ごとに「その区間から推定し直した」係数が紛れ込む
    余地が構造上なくなる。異常検知では区間ごとの再推定が特に悪い —— テスト区間
    から推定した尺度にはその区間の**異常が入っている**ため、異常が「正常な
    ばらつき」として吸収される (D-57 の根拠)。

    Attributes:
        center: 成分ごとの中心 ``(D,)``。
        scale: 成分ごとの尺度 ``(D,)``。0 は許さない。
        normalize: 使った方式 (``NORMALIZE_METHODS`` のいずれか)。来歴として
            持ち回り、CSV の ``normalize`` 列と 5-C の格子点の照合に使う。
        n_steps: 係数の推定に使った先頭行数 (来歴)。
    """

    center: FloatArray
    scale: FloatArray
    normalize: str
    n_steps: int

    @classmethod
    def from_training_prefix(
        cls, series: FloatArray, n_steps: int, normalize: str = "zscore"
    ) -> AnomalyPreprocessor:
        """先頭 ``n_steps`` 行から係数を1組だけ推定する (D-57)。

        Args:
            series: ``(T, D)`` の系列。
            n_steps: 係数の推定に使う先頭行数。**訓練区間の内側**であること。
            normalize: ``"zscore"`` (平均・標準偏差) / ``"minmax"`` (最小・幅) /
                ``"robust"`` (中央値・IQR) / ``"none"`` (恒等変換)。

        Raises:
            ValueError: 形状不正、``n_steps`` が範囲外、``normalize`` が未対応、
                または推定した尺度に 0 が含まれる場合。
        """
        array = _as_series_matrix(series, "series")
        if normalize not in NORMALIZE_METHODS:
            raise ValueError(
                f"normalize は {NORMALIZE_METHODS} のいずれかです: {normalize!r}"
            )
        if not 2 <= n_steps <= array.shape[0]:
            raise ValueError(
                "n_steps は 2 以上・系列長以下である必要があります: "
                f"n_steps={n_steps}, T={array.shape[0]}"
            )
        prefix = array[:n_steps]
        center, scale = _coefficients(prefix, normalize)
        if not np.all(np.isfinite(scale)) or not np.all(scale > 0.0):
            raise ValueError(
                f"normalize={normalize!r} の尺度が 0 または非有限です "
                f"(定数成分は正規化できません): scale={scale!r}"
            )
        return cls(center=center, scale=scale, normalize=normalize, n_steps=n_steps)

    def apply(self, series: FloatArray) -> FloatArray:
        """``(series - center) / scale``。**係数を再推定しない**。"""
        transformed: FloatArray = (
            _as_series_matrix(series, "series") - self.center
        ) / self.scale
        return transformed

    def invert(self, series: FloatArray) -> FloatArray:
        """``apply`` の逆変換 (残差を物理量へ戻すときに使う)。"""
        original: FloatArray = (
            _as_series_matrix(series, "series") * self.scale + self.center
        )
        return original


def _coefficients(prefix: FloatArray, normalize: str) -> tuple[FloatArray, FloatArray]:
    """方式ごとの (中心, 尺度)。``from_training_prefix`` だけが呼ぶ内部関数。"""
    if normalize == "zscore":
        return np.mean(prefix, axis=0), np.std(prefix, axis=0)
    if normalize == "minmax":
        low = np.min(prefix, axis=0)
        return low, np.max(prefix, axis=0) - low
    if normalize == "robust":
        quartiles = np.percentile(prefix, (25.0, 50.0, 75.0), axis=0)
        return quartiles[1], quartiles[2] - quartiles[0]
    return (
        np.zeros(prefix.shape[1], dtype=np.float64),
        np.ones(prefix.shape[1], dtype=np.float64),
    )


def _find_cut_search_cells(cfg: SyntheticAnomalyConfig) -> int:
    """``_find_cut`` が確保する ``starts x spans`` の要素数 (``_validate`` 用)。

    ``_find_cut`` 本体 (297-304行) の窓サイズ計算と同じ式を使う。``_validate``
    は生成前に呼ばれ実際の ``raw.size`` を持たないが、通常経路の窓幅は
    ``segment_length`` だけで決まるため、確保前にこの式だけで検査できる。
    """
    min_span = _REMOVED_SPAN_FACTORS[0] * cfg.segment_length
    max_span = _REMOVED_SPAN_FACTORS[1] * cfg.segment_length
    half_width = max(1, cfg.segment_length // 2)
    starts_size = 2 * half_width + 1
    spans_size = max_span - min_span + 1
    return starts_size * spans_size


def _validate(cfg: SyntheticAnomalyConfig) -> None:
    """合成源の設定を検査する (確保より前に落とす)。"""
    if cfg.n_anomalies < 1:
        raise ValueError(f"n_anomalies は 1 以上が必要です: {cfg.n_anomalies}")
    if cfg.segment_length < 2:
        raise ValueError(f"segment_length は 2 以上が必要です: {cfg.segment_length}")
    if cfg.ignore_margin < 0:
        raise ValueError(f"ignore_margin は 0 以上が必要です: {cfg.ignore_margin}")
    block = cfg.segment_length + 2 * cfg.ignore_margin
    needed = int(np.ceil(cfg.n_anomalies * 2 * block / (1.0 - FIRST_ANOMALY_FRACTION)))
    if cfg.length < needed:
        raise ValueError(
            "length が短すぎます (異常区間と ignore の余白が入りません): "
            f"length={cfg.length}, 必要={needed} "
            f"(n_anomalies={cfg.n_anomalies}, segment_length={cfg.segment_length}, "
            f"ignore_margin={cfg.ignore_margin})"
        )
    max_span = _REMOVED_SPAN_FACTORS[1] * cfg.segment_length
    raw_samples = cfg.length + cfg.n_anomalies * (max_span + cfg.segment_length) + 2
    if raw_samples > _MAX_RAW_SAMPLES:
        raise ValueError(
            "合成源が要求する Mackey-Glass のサンプル数が上限を超えます: "
            f"{raw_samples} > {_MAX_RAW_SAMPLES}"
        )
    cells = _find_cut_search_cells(cfg)
    if cells > _MAX_FIND_CUT_CELLS:
        raise ValueError(
            "_find_cut が確保する探索行列 (starts x spans) が大きすぎます: "
            f"{cells} > {_MAX_FIND_CUT_CELLS} "
            f"(segment_length={cfg.segment_length})"
        )


def _splice_positions(
    cfg: SyntheticAnomalyConfig, rng: np.random.Generator
) -> list[int]:
    """出力座標での縫合点の目標位置 (先頭は ``FIRST_ANOMALY_FRACTION`` より後ろ)。"""
    half = cfg.segment_length // 2
    guard = half + cfg.ignore_margin + 1
    low = int(FIRST_ANOMALY_FRACTION * cfg.length) + guard
    high = cfg.length - guard
    edges = np.linspace(low, high, cfg.n_anomalies + 1)
    positions: list[int] = []
    for index in range(cfg.n_anomalies):
        block_low = int(edges[index]) + guard
        block_high = int(edges[index + 1]) - guard
        if block_high <= block_low:
            positions.append(int(0.5 * (edges[index] + edges[index + 1])))
            continue
        positions.append(int(rng.integers(block_low, block_high)))
    return positions


def _find_cut(
    raw: FloatArray,
    derivative: FloatArray,
    center: int,
    cfg: SyntheticAnomalyConfig,
    value_scale: float,
    slope_scale: float,
) -> tuple[int, int]:
    """値と微分が最も一致する ``(i, span)`` を探す (MGAB と同じ着想)。

    ``raw[i]`` と ``raw[i + span]`` の値・微分が一致する2点を選び、その間
    (``raw[i+1 : i+span+1]``) を取り除いて縫合する。値も微分も連続なので、
    縫合そのものは目視でも1次差分でも見えない —— 見えるのは「そこで系の位相が
    飛んだ」という**力学の異常**だけである。

    (reviewer-performance 指摘) ``value_scale`` / ``slope_scale`` は ``raw`` /
    ``derivative`` 全体に対する標準偏差であり、呼び出しをまたいで不変。以前は
    ここで呼び出しごとに ``np.std`` を再計算しており、
    ``generate_synthetic_anomalies`` が本関数を ``n_anomalies`` 回呼ぶことと
    合わせて全体のコストが実質 O(n_anomalies^2) になっていた。呼び出し側
    (``generate_synthetic_anomalies``) が1回だけ計算して引数で渡すことで、本関数
    のコストは ``_MAX_FIND_CUT_CELLS`` で上限管理されている軸 (cells) だけに
    戻り、``n_anomalies`` に対して線形になる。
    """
    min_span = _REMOVED_SPAN_FACTORS[0] * cfg.segment_length
    max_span = _REMOVED_SPAN_FACTORS[1] * cfg.segment_length
    half_width = max(1, cfg.segment_length // 2)
    highest_possible = raw.size - max_span - 2
    lowest = max(1, center - half_width)
    highest = min(highest_possible, center + half_width)
    if highest <= lowest:
        lowest, highest = 1, max(2, highest_possible)
    starts = np.arange(lowest, highest + 1)
    spans = np.arange(min_span, max_span + 1)
    ends = starts[:, None] + spans[None, :]
    cost = (
        np.abs(raw[ends] - raw[starts][:, None]) / value_scale
        + np.abs(derivative[ends] - derivative[starts][:, None]) / slope_scale
    )
    row, column = divmod(int(np.argmin(cost)), spans.size)
    return int(starts[row]), int(spans[column])


def generate_synthetic_anomalies(
    cfg: SyntheticAnomalyConfig, rng: np.random.Generator
) -> AnomalySeries:
    """MGAB と同じ手続きで異常を挿入した Mackey-Glass 系列を作る。

    Mackey-Glass の積分は ``tasks/mackey_glass.py`` の
    ``generate_mackey_glass`` へ委譲する (**再実装しない**)。手続きは MGAB
    (Thill et al., CC0-1.0) と同じで、

    1. 目標より長い系列を作る
    2. 値と微分が一致する2点を見つけ、その間のセグメントを取り除いて縫合する
    3. 縫合点を中心に ``segment_length`` 点を異常として標識し、その**前後**
       ``ignore_margin`` 点を評価から除外する

    Args:
        cfg: 合成源の設定。
        rng: **task ストリーム**の Generator (D-06)。

    Returns:
        長さ ``cfg.length`` の ``AnomalySeries``。
    """
    _validate(cfg)
    max_span = _REMOVED_SPAN_FACTORS[1] * cfg.segment_length
    raw_samples = cfg.length + cfg.n_anomalies * (max_span + cfg.segment_length) + 2
    from rc_basics_lab.config import MackeyGlassConfig

    raw_config = MackeyGlassConfig(
        tau=cfg.mackey_glass.tau, length=raw_samples, horizon=1
    )
    raw: FloatArray = generate_mackey_glass(raw_config, rng).u[:, 0]
    derivative: FloatArray = np.gradient(raw)
    value_scale = float(np.std(raw)) or 1.0
    slope_scale = float(np.std(derivative)) or 1.0

    pieces: list[FloatArray] = []
    splices: list[int] = []
    cursor = 0
    removed = 0
    for target in _splice_positions(cfg, rng):
        start, span = _find_cut(
            raw, derivative, target + removed, cfg, value_scale, slope_scale
        )
        if start < cursor:
            raise ValueError(
                "縫合点の探索が直前の切断と重なりました "
                f"(start={start}, cursor={cursor})。segment_length に対して "
                "length が短すぎます"
            )
        pieces.append(raw[cursor : start + 1])
        splices.append(start - removed)
        cursor = start + span + 1
        removed += span
    pieces.append(raw[cursor:])
    joined: FloatArray = np.concatenate(pieces)[: cfg.length]
    if joined.size != cfg.length:
        raise ValueError(
            f"合成後の系列長が足りません: {joined.size} != {cfg.length} "
            "(除去したセグメントが想定より長い)"
        )

    labels = np.zeros(cfg.length, dtype=np.bool_)
    ignore = np.zeros(cfg.length, dtype=np.bool_)
    half = cfg.segment_length // 2
    for splice in splices:
        start = max(0, splice - half)
        end = min(cfg.length, start + cfg.segment_length)
        labels[start:end] = True
        ignore[max(0, start - cfg.ignore_margin) : start] = True
        ignore[end : min(cfg.length, end + cfg.ignore_margin)] = True
    ignore &= ~labels
    first_anomaly = int(np.argmax(labels))
    train_end = max(1, first_anomaly - cfg.ignore_margin)
    params = {
        "source": "synthetic",
        "n_anomalies": str(len(splices)),
        "segment_length": str(cfg.segment_length),
        "ignore_margin": str(cfg.ignore_margin),
        "tau": str(cfg.mackey_glass.tau),
    }
    return AnomalySeries(
        values=joined.reshape(-1, 1),
        labels=labels,
        ignore=ignore,
        train_end=train_end,
        name=TASK_NAME,
        params=params,
    )


__all__ = [
    "FIRST_ANOMALY_FRACTION",
    "NORMALIZE_METHODS",
    "TASK_NAME",
    "AnomalyPreprocessor",
    "AnomalySeries",
    "generate_synthetic_anomalies",
]
