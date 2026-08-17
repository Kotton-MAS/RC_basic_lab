"""実験 2-D の配線層 —— washout 長への性能感度 (D-19).

**手法比較の公平性はここで作らない**。実体は ``dataclasses.replace`` で
``split.washout`` を差し替えて 01 の ``run_experiment`` を呼ぶだけのループであり、
D-04 (alpha 格子は単一キー) / D-05 (全手法が同じ行 index) / D-08 (検証分割で
構造ハイパーパラメータを選ばない) は既存経路がそのまま担保する。自前で比較
ロジックを書くと、そこだけ公平性の保証が二重化して割れる。

**交絡の除去 (D-19) がこのモジュールの本体**である。``make_split`` は

    n_usable = n_steps - max_start_offset - t0

で訓練/検証/テストの行数を決め、``t0 = max(washout, 各手法の first_valid)`` な
ので、washout を素直に振ると**訓練データ量が同時に減る**。この交絡は滑らかな
単調曲線として出るため図を見ても気づけず、受け入れ条件5 (「washout 長の性能
変動を定量化」) が別の量の測定に化ける。``pad_series=True`` (既定) では
``t0`` の増分ぶんだけ系列長を伸ばし、行数を格子全体で一定に保つ。
``pad_series=False`` は交絡ありの設計を再現する対比用モードとして残す。

対象は Mackey-Glass と遅延パリティの**両方**である (``run_experiment`` が両方を
回すので追加コストはほぼゼロ)。図の主役は MG で、パリティは「washout に反応
しない対照」として重ねる: パリティ用 ESN は ``leak_rate = 1.0`` (前ステップの
状態を持ち越さない) かつ目標が直近2入力にしか依存しないため、過渡を捨てる量を
変えても性能が動かないはずである。この**予測が外れた場合はそれ自体が発見**なので、
閾値を動かして予測に合わせない。
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields

from rc_basics_lab.config import Esp02Config, ExperimentConfig, WashoutSweepConfig
from rc_basics_lab.experiment.runner import (
    ESN_METHOD,
    ResultRow,
    build_methods,
    run_experiment,
)
from rc_basics_lab.experiment.split import compute_t0
from rc_basics_lab.readout.design import first_valid_for
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as MACKEY_GLASS

logger = logging.getLogger(__name__)

HEADLINE_TASK = MACKEY_GLASS
HEADLINE_METHOD = ESN_METHOD
"""図と ``meta.json`` の主役 (受け入れ条件5 の数値はこの組で報告する)。

パリティは「washout に反応しない対照」なので変動幅の主張には使わない。
"""


@dataclass(frozen=True, slots=True)
class WashoutRow:
    """``washout_sensitivity.csv`` の1行。**宣言順が CSV の列順の単一の真実**。

    長形式 (1行 = 1 (課題, 手法, washout, レプリケート)) である。
    ``nrmse_std`` だけは行の粒度より1段粗く、**同じ (課題, 手法, washout) の
    レプリケート間の標準偏差**を各行に載せている (レプリケート1本なら 0)。
    図の誤差棒と「変動幅がレプリケート間のばらつきより大きいか」の判断が、
    この CSV1枚だけで完結するようにするため。

    Attributes:
        task: 課題名 (``mackey_glass`` / ``delay_parity``)。
        method: 手法名 (``linear`` / ``delay_line`` / ``esn``)。
        washout: 掃引した washout [ステップ]。
        replicate: レプリケート番号 (0 始まり)。
        alpha: 検証分割が選んだ ridge の alpha。
        n_lags: 選ばれた候補の ``first_valid`` (遅延線以外は 0)。
        nrmse: テスト分割の NRMSE。
        nrmse_std: 同じ (課題, 手法, washout) のレプリケート間標準偏差。
        n_train: 学習行数。``pad_series=True`` なら格子全体で一定 (D-19)。
        n_val: 検証行数。
        n_test: テスト行数。
        t0: その washout での全手法共通の基準行。
        pad_series: 系列長で行数を補償したか (D-19)。
        wall_time_s: この (課題, 手法, レプリケート) の実測時間。
    """

    task: str
    method: str
    washout: int
    replicate: int
    alpha: float
    n_lags: int
    nrmse: float
    nrmse_std: float
    n_train: int
    n_val: int
    n_test: int
    t0: int
    pad_series: bool
    wall_time_s: float


WASHOUT_CSV_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(WashoutRow))
"""``washout_sensitivity.csv`` の列順 (``WashoutRow`` の宣言順が単一の真実)。"""


@dataclass(frozen=True, slots=True)
class MethodSensitivity:
    """1 (課題, 手法) の washout 感度 (レプリケート平均 NRMSE の変動幅)。

    Attributes:
        task: 課題名。
        method: 手法名。
        nrmse_min: 格子上での最小のレプリケート平均 NRMSE。
        nrmse_max: 同じく最大。
        washout_at_min: ``nrmse_min`` を与えた washout。
        washout_at_max: ``nrmse_max`` を与えた washout。
        ratio: ``nrmse_max / nrmse_min`` (受け入れ条件5 の「変動」)。
        nrmse_at_reference: 01 の本番値 (``base.split.washout``) での平均 NRMSE。
    """

    task: str
    method: str
    nrmse_min: float
    nrmse_max: float
    washout_at_min: int
    washout_at_max: int
    ratio: float
    nrmse_at_reference: float

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "task": self.task,
            "method": self.method,
            "nrmse_min": self.nrmse_min,
            "nrmse_max": self.nrmse_max,
            "washout_at_min": self.washout_at_min,
            "washout_at_max": self.washout_at_max,
            "ratio": self.ratio,
            "nrmse_at_reference": self.nrmse_at_reference,
        }


@dataclass(frozen=True, slots=True)
class WashoutSensitivity:
    """2-D の要約 (``meta.json`` の ``washout_sensitivity``)。

    受け入れ条件5 は「washout 長の性能変動が定量化されている」ことなので、
    ``headline.ratio`` (MG x ESN の最大/最小) を一次資料として残す。併せて
    ``training_size_is_constant`` を残すのは、D-19 の補償が実際に効いた実行
    だったかを成果物だけから確かめられるようにするため (補償が外れた実行の
    数値は「washout の効果」ではない)。

    Attributes:
        grid: 掃引した washout。
        pad_series: 行数を補償したか。
        reference_washout: 01 の本番値 (図の垂直線の位置)。
        n_rows: ``washout_sensitivity.csv`` の行数。
        training_size_is_constant: 全格子点で ``(n_train, n_val, n_test)`` が一致。
        sizes_by_washout: washout ごとの ``(n_train, n_val, n_test)``。
        t0_by_washout: washout ごとの ``t0``。
        by_method: (課題, 手法) ごとの変動幅。
        headline: ``HEADLINE_TASK`` x ``HEADLINE_METHOD`` の変動幅。
    """

    grid: tuple[int, ...]
    pad_series: bool
    reference_washout: int
    n_rows: int
    training_size_is_constant: bool
    sizes_by_washout: tuple[tuple[int, tuple[int, int, int]], ...]
    t0_by_washout: tuple[tuple[int, int], ...]
    by_method: tuple[MethodSensitivity, ...]
    headline: MethodSensitivity

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "grid": list(self.grid),
            "pad_series": self.pad_series,
            "reference_washout": self.reference_washout,
            "n_rows": self.n_rows,
            "training_size_is_constant": self.training_size_is_constant,
            "sizes_by_washout": [
                {
                    "washout": washout,
                    "n_train": sizes[0],
                    "n_val": sizes[1],
                    "n_test": sizes[2],
                }
                for washout, sizes in self.sizes_by_washout
            ],
            "t0_by_washout": [
                {"washout": washout, "t0": t0} for washout, t0 in self.t0_by_washout
            ],
            "by_method": [item.to_summary() for item in self.by_method],
            "headline": self.headline.to_summary(),
        }

    def find(self, task: str, method: str) -> MethodSensitivity:
        """(課題, 手法) の変動幅を取り出す。

        Raises:
            KeyError: その組が掃引に含まれない場合。
        """
        for item in self.by_method:
            if item.task == task and item.method == method:
                return item
        raise KeyError(f"掃引に含まれない組です: task={task!r}, method={method!r}")


def _first_valids(base: ExperimentConfig) -> tuple[int, ...]:
    """全手法・全候補の ``first_valid`` (系列を作らずに求める)。

    ``plan_replicate`` が ``compute_t0`` へ渡すのと同じ列。設計行列を実際に
    組んでから読むのではなく ``first_valid_for`` から引くので、系列長を決める
    前に ``t0`` が分かる (D-19 の補償に必要)。
    """
    return tuple(
        first_valid_for(spec)
        for method in build_methods(base)
        for spec in method.candidates
    )


def predicted_t0(base: ExperimentConfig, washout: int) -> int:
    """``washout`` で実験を回したときの ``t0`` を、実験を回さずに求める。

    ``plan_replicate`` の ``compute_t0(全 first_valid, config.split.washout)``
    と同じ計算を、系列を生成せずに行う。``tests/test_experiment_washout.py``
    が実際の ``t0`` と一致することを実測で固定する。
    """
    return compute_t0(_first_valids(base), washout)


def variant_for(section: WashoutSweepConfig, washout: int) -> ExperimentConfig:
    """1格子点ぶんの 01 用設定を作る (**差し替えるのは washout と系列長だけ**)。

    ``pad_series`` が真なら、``t0`` が増えたぶんだけ両課題の ``length`` を
    伸ばして ``n_usable = n_steps - max_start_offset - t0`` を一定に保つ
    (D-19)。基準は**格子の最小値**での ``t0`` なので、補償は常に「伸ばす」側に
    働き、01 の本番設定より短い系列で測ることはない。

    仕様 §4 T4 の式は ``length = base_length + (max(grid) - washout)`` と
    書かれていたが、この式は washout が大きいほど系列を**短く**するので
    ``n_usable`` の差が広がる (符号が逆)。ここでは仕様が述べている目的
    (「行数を格子全体で一定に保つ」) を満たす式を採る。加えて ``t0`` は
    ``max(washout, first_valid)`` なので、washout が遅延線の最大ラグより
    小さい領域では ``washout`` の差分ではなく ``t0`` の差分で補償しないと
    行数がそろわない (実測: 本番格子の washout=0 と 50 はどちらも
    ``t0 = 64`` になる)。

    Raises:
        ValueError: ``washout`` が格子の最小値より小さい場合 (補償が負になる)。
    """
    base = section.base
    baseline_t0 = predicted_t0(base, min(section.grid))
    t0 = predicted_t0(base, washout)
    if t0 < baseline_t0:
        raise ValueError(
            f"washout={washout} の t0 が格子最小値の t0 を下回ります: "
            f"{t0} < {baseline_t0}"
        )
    padding = t0 - baseline_t0 if section.pad_series else 0
    return dataclasses.replace(
        base,
        split=dataclasses.replace(base.split, washout=washout),
        mackey_glass=dataclasses.replace(
            base.mackey_glass, length=base.mackey_glass.length + padding
        ),
        delay_parity=dataclasses.replace(
            base.delay_parity, length=base.delay_parity.length + padding
        ),
    )


def _validate_grid(grid: Sequence[int]) -> None:
    if not grid:
        raise ValueError("washout.grid が空です")
    if any(washout < 0 for washout in grid):
        raise ValueError(f"washout は 0 以上である必要があります: {tuple(grid)}")


def _std_by_group(
    pairs: Sequence[tuple[int, ResultRow]],
) -> Mapping[tuple[str, str, int], float]:
    """(課題, 手法, washout) ごとの NRMSE のレプリケート間標準偏差 (ddof=1)。"""
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for washout, row in pairs:
        grouped.setdefault((row.task, row.method, washout), []).append(row.nrmse)
    return {
        key: statistics.stdev(values) if len(values) > 1 else 0.0
        for key, values in grouped.items()
    }


def run_washout_sweep(config: Esp02Config) -> tuple[WashoutRow, ...]:
    """実験 2-D: washout を振って 01 の3手法 x 2課題を回す (受け入れ条件5)。

    ``config.washout.grid`` の各点について ``variant_for`` で 01 用設定を作り、
    既存の ``run_experiment`` をそのまま呼ぶ。手法比較の公平性はその経路が
    担保しており、ここは washout と系列長以外を一切触らない。

    Args:
        config: 02 の設定 (読むのは ``config.washout`` だけ)。

    Returns:
        長形式の行 (格子の宣言順 x ``run_experiment`` の返す順)。

    Raises:
        ValueError: 格子が空、または負の washout を含む場合。
    """
    section = config.washout
    _validate_grid(section.grid)
    pairs: list[tuple[int, ResultRow]] = []
    for washout in section.grid:
        started = time.perf_counter()
        variant = variant_for(section, washout)
        result_rows = run_experiment(variant)
        pairs.extend((washout, row) for row in result_rows)
        logger.info(
            "2-D washout=%d pad_series=%s t0=%d length=%d 行数=%d (%.2fs)",
            washout,
            section.pad_series,
            result_rows[0].t0 if result_rows else -1,
            variant.mackey_glass.length,
            len(result_rows),
            time.perf_counter() - started,
        )
    stds = _std_by_group(pairs)
    return tuple(
        WashoutRow(
            task=row.task,
            method=row.method,
            washout=washout,
            replicate=row.replicate,
            alpha=row.alpha,
            n_lags=row.n_lags,
            nrmse=row.nrmse,
            nrmse_std=stds[row.task, row.method, washout],
            n_train=row.n_train,
            n_val=row.n_val,
            n_test=row.n_test,
            t0=row.t0,
            pad_series=section.pad_series,
            wall_time_s=row.wall_time_s,
        )
        for washout, row in pairs
    )


def mean_nrmse_by_washout(
    rows: Iterable[WashoutRow], task: str, method: str
) -> dict[int, float]:
    """(課題, 手法) の washout -> レプリケート平均 NRMSE。図と要約が共有する。"""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.task == task and row.method == method:
            grouped.setdefault(row.washout, []).append(row.nrmse)
    return {
        washout: statistics.fmean(values) for washout, values in sorted(grouped.items())
    }


def _sensitivity_for(
    rows: Sequence[WashoutRow], task: str, method: str, reference_washout: int
) -> MethodSensitivity:
    means = mean_nrmse_by_washout(rows, task, method)
    if not means:
        raise ValueError(f"行がありません: task={task!r}, method={method!r}")
    washout_at_min = min(means, key=lambda key: means[key])
    washout_at_max = max(means, key=lambda key: means[key])
    smallest = means[washout_at_min]
    largest = means[washout_at_max]
    return MethodSensitivity(
        task=task,
        method=method,
        nrmse_min=smallest,
        nrmse_max=largest,
        washout_at_min=washout_at_min,
        washout_at_max=washout_at_max,
        # NRMSE が厳密に 0 になるのは予測が完全一致した場合だけで、実験では
        # 起きない。起きたら比は定義できないので nan を返す (1.0 で埋めると
        # 「変動が無かった」と読めてしまう)。
        ratio=largest / smallest if smallest > 0.0 else math.nan,
        nrmse_at_reference=means.get(reference_washout, math.nan),
    )


def _task_method_pairs(rows: Sequence[WashoutRow]) -> tuple[tuple[str, str], ...]:
    """出現順を保った (課題, 手法) の重複なし列。"""
    seen: dict[tuple[str, str], None] = {}
    for row in rows:
        seen.setdefault((row.task, row.method), None)
    return tuple(seen)


def summarize_washout_sensitivity(
    config: Esp02Config, rows: Sequence[WashoutRow]
) -> WashoutSensitivity:
    """2-D の結果を ``meta.json`` 用に要約する (受け入れ条件5 の一次資料)。

    Raises:
        ValueError: 行が空、または主役の組 (MG x ESN) が掃引に無い場合。
    """
    if not rows:
        raise ValueError("2-D の行がありません")
    section = config.washout
    reference_washout = section.base.split.washout
    sizes: dict[int, tuple[int, int, int]] = {}
    t0s: dict[int, int] = {}
    for row in rows:
        sizes[row.washout] = (row.n_train, row.n_val, row.n_test)
        t0s[row.washout] = row.t0
    by_method = tuple(
        _sensitivity_for(rows, task, method, reference_washout)
        for task, method in _task_method_pairs(rows)
    )
    headline = next(
        (
            item
            for item in by_method
            if item.task == HEADLINE_TASK and item.method == HEADLINE_METHOD
        ),
        None,
    )
    if headline is None:
        raise ValueError(
            f"主役の組が掃引にありません: {HEADLINE_TASK} x {HEADLINE_METHOD}"
        )
    return WashoutSensitivity(
        grid=tuple(section.grid),
        pad_series=section.pad_series,
        reference_washout=reference_washout,
        n_rows=len(rows),
        training_size_is_constant=len(set(sizes.values())) == 1,
        sizes_by_washout=tuple(sorted(sizes.items())),
        t0_by_washout=tuple(sorted(t0s.items())),
        by_method=by_method,
        headline=headline,
    )


__all__ = [
    "EXPERIMENT_WASHOUT",
    "HEADLINE_METHOD",
    "HEADLINE_TASK",
    "WASHOUT_CSV_COLUMNS",
    "MethodSensitivity",
    "WashoutRow",
    "WashoutSensitivity",
    "mean_nrmse_by_washout",
    "predicted_t0",
    "run_washout_sweep",
    "summarize_washout_sensitivity",
    "variant_for",
]
