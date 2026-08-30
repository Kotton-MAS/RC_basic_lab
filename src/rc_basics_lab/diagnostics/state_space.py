"""状態空間の広がりを見る診断 (PCA).

実験1-B「リザバー状態は入力空間より高次元に広がる」の数値的裏付けと、
``fig_state_space.png`` の材料の両方をここが供給する。
"""

from __future__ import annotations

import numpy as np

from rc_basics_lab.diagnostics.base import (
    DiagnosticContext,
    DiagnosticResult,
    resolve_context,
    validate_diagnostic_input,
)
from rc_basics_lab.types import FloatArray

NAME = "state_pca"

NAME_UNIT_ACTIVITY = "unit_activity"

DORMANT_VARIANCE_RATIO = 1.0e-2
"""「ほとんど動かない」と見なす分散の基準 (中央値に対する比)。

絶対値ではなく**中央値との比**で決める。状態の分散のスケールは
``input_scale`` / ``sigma_u`` / ``leak_rate`` で桁ごと動くので、絶対閾値では
条件を変えるたびに意味が変わる。比なら「他のユニットに比べて動いていない」
という主張がそのまま測れる。

1/100 は分散の比なので、標準偏差では 1/10 にあたる。
"""

_ACTIVITY_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
"""報告する分位点。分布の形を数値で残すため中央値だけにしない。"""

_CUMULATIVE_THRESHOLD = 0.95
"""``n_components_95`` の閾値。累積寄与率がこの値に到達する最小の主成分数を数える。"""

_N_PC_SCORES = 2
"""散布図用に返す主成分スコアの本数。"""


def _participation_ratio(eigenvalues: FloatArray) -> float:
    """``(Σλ)² / Σλ²``。等方な N 次元分布でおおよそ N になる有効次元数。"""
    total = float(np.sum(eigenvalues))
    squared = float(np.sum(eigenvalues**2))
    if squared == 0.0:
        raise ValueError("分散が全て 0 のため participation_ratio を定義できません")
    return total**2 / squared


def state_pca(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
) -> DiagnosticResult:
    """状態系列を中心化して特異値分解し、実効次元の指標を返す。

    Args:
        X: 状態系列 ``(T, N)``。
        u: 未使用 (プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` のみ参照する。

    Returns:
        ``scalars``: ``n_components_95`` / ``participation_ratio`` /
        ``total_variance`` / ``n_samples`` / ``n_features``。
        ``arrays``: ``eigenvalues`` / ``explained_variance_ratio`` /
        ``cumulative_ratio`` / ``pc_scores`` (先頭2成分)。
    """
    validate_diagnostic_input(X, u, y, ctx)
    context = resolve_context(ctx)
    states = np.asarray(X, dtype=np.float64)[context.washout :]
    n_samples, n_features = states.shape
    if n_samples < 2:
        raise ValueError(f"washout 後のサンプル数が不足しています: {n_samples}")

    centered = states - states.mean(axis=0, keepdims=True)
    # 共分散行列を作らず SVD で解く (条件数の悪化を避ける)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues: FloatArray = singular_values**2 / (n_samples - 1)

    total_variance = float(np.sum(eigenvalues))
    if total_variance == 0.0:
        raise ValueError("状態系列が定数のため PCA を定義できません")
    explained_variance_ratio: FloatArray = eigenvalues / total_variance
    cumulative_ratio: FloatArray = np.cumsum(explained_variance_ratio)
    # 浮動小数の丸めで最終要素が 0.9999... になっても最大値で打ち切れるようにする
    n_components_95 = int(
        np.searchsorted(cumulative_ratio, _CUMULATIVE_THRESHOLD, side="left") + 1
    )
    n_components_95 = min(n_components_95, len(cumulative_ratio))

    n_scores = min(_N_PC_SCORES, len(singular_values))
    pc_scores: FloatArray = left[:, :n_scores] * singular_values[:n_scores]

    return DiagnosticResult(
        name=NAME,
        scalars={
            "n_components_95": float(n_components_95),
            "participation_ratio": _participation_ratio(eigenvalues),
            "total_variance": total_variance,
            "n_samples": float(n_samples),
            "n_features": float(n_features),
        },
        arrays={
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_ratio": cumulative_ratio,
            "pc_scores": pc_scores,
        },
        params={
            "washout": str(context.washout),
            "cumulative_threshold": str(_CUMULATIVE_THRESHOLD),
        },
    )


def unit_activity(
    X: FloatArray,
    u: FloatArray | None = None,
    y: FloatArray | None = None,
    *,
    ctx: DiagnosticContext | None = None,
) -> DiagnosticResult:
    """ユニットごとの分散の分布と、ほとんど動かないユニットの割合を返す。

    記事01 §3.3 の「ほとんど動かないユニットが混ざっている」を**印象ではなく
    行の値**で言えるようにするための診断 (T4)。``state_pca`` が空間全体の
    広がりを見るのに対し、こちらは**個々のユニットの動きの偏り**を見る。
    有効次元が低いことと、特定のユニットが死んでいることは別の現象である
    (全ユニットが等しく弱く動いても有効次元は下がる)。

    ``X`` だけを入力に取り ESN を参照しない (D-01 の共通署名)。

    Args:
        X: 状態系列 ``(T, N)``。
        u: 未使用 (プロトコル適合のために受け取る)。
        y: 未使用 (同上)。
        ctx: ``washout`` のみ参照する。

    Returns:
        ``scalars``: ``n_units`` / ``n_dormant`` / ``dormant_fraction`` /
        ``variance_median`` / ``variance_min`` / ``variance_max`` /
        ``variance_q05`` 〜 ``variance_q95`` / ``variance_min_to_median``。
        ``arrays``: ``unit_variance`` (ユニットごとの分散、列の順)。

    Raises:
        ValueError: washout 後のサンプル数が 2 未満、または全ユニットの分散が
            0 で中央値との比を定義できない場合。
    """
    validate_diagnostic_input(X, u, y, ctx)
    context = resolve_context(ctx)
    states = np.asarray(X, dtype=np.float64)[context.washout :]
    n_samples, n_units = states.shape
    if n_samples < 2:
        raise ValueError(f"washout 後のサンプル数が不足しています: {n_samples}")

    variance: FloatArray = np.var(states, axis=0, ddof=1)
    median = float(np.median(variance))
    if median <= 0.0:
        raise ValueError(
            "分散の中央値が 0 のため、ほとんど動かないユニットを定義できません"
        )
    dormant = variance < median * DORMANT_VARIANCE_RATIO
    n_dormant = int(np.count_nonzero(dormant))
    quantiles = np.quantile(variance, _ACTIVITY_QUANTILES)

    scalars: dict[str, float] = {
        "n_units": float(n_units),
        "n_dormant": float(n_dormant),
        "dormant_fraction": n_dormant / n_units,
        "variance_median": median,
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
        "variance_min_to_median": float(np.min(variance)) / median,
    }
    for level, value in zip(_ACTIVITY_QUANTILES, quantiles, strict=True):
        scalars[f"variance_q{int(level * 100):02d}"] = float(value)

    return DiagnosticResult(
        name=NAME_UNIT_ACTIVITY,
        scalars=scalars,
        arrays={"unit_variance": variance},
        params={
            "washout": str(context.washout),
            "dormant_variance_ratio": str(DORMANT_VARIANCE_RATIO),
        },
    )


__all__ = [
    "DORMANT_VARIANCE_RATIO",
    "NAME",
    "NAME_UNIT_ACTIVITY",
    "state_pca",
    "unit_activity",
]
