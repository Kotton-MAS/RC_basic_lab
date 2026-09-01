"""1コマンドで 04 の成果物をそろえる経路 (受け入れ条件6).

``onestep.csv`` / ``freerun.csv`` / ``freerun_profile.csv`` / ``stability.csv`` /
``capacity.csv`` / 図5枚 / ``meta.json`` をここで一括生成する。CLI
(``main.py --experiment 04`` と ``main.py --experiment 04``) は
この関数を呼ぶだけの薄い層にして、「どのコマンドから走らせても同じ成果物が出る」
を構造で保証する (01 の ``pipeline.py`` / 02 の ``esp_pipeline.py`` / 03 の
``capacity_pipeline.py`` と同じ規律)。

**真の軌道と lambda_max は1回だけ推定する** (仕様 §5 禁止する構造3)。
``estimate_lorenz_lyapunov`` の結果を 4-B と 4-C の両方へ引数で配るので、
条件ごとに積分し直す経路が存在しない。

**作図層の import は関数本体の中で行う** (D-53)。``experiment`` 配下が
``plotting`` を module-level で import すると循環が復活し、
``tests/test_layer_boundaries.py`` の AST guard と subprocess guard の両方が
落ちる。

``meta.json`` には性能予算 (仕様 §5) を成果物だけで判定できるよう、区間ごとの
実測時間 (``wall_time_breakdown``) を載せる。加えて受け入れ条件の一次資料
(``valid_time_sensitivity`` / ``attractor_verdict`` / ``regime_counts``) も
ここに残す —— ``docs/design.md`` §12 の表はこれらから機械照合される。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.config import Chaos04Config
from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.capacity_pipeline import write_capacity_csv
from rc_basics_lab.experiment.capacity_rows import (
    CapacityRow,
)
from rc_basics_lab.experiment.diagnostics_rows import (
    DIAGNOSTICS_CSV,
    scalar_rows,
    write_diagnostics_csv,
)
from rc_basics_lab.experiment.freerun import (
    FREERUN_CSV,
    FREERUN_PROFILE_CSV,
    ONESTEP_CSV,
    FreeRunResults,
    estimate_lorenz_lyapunov,
    run_freerun_experiment,
    run_onestep,
    write_freerun_csv,
    write_freerun_profile_csv,
    write_onestep_csv,
)
from rc_basics_lab.experiment.freerun_rows import (
    FreeRunEvaluation,
    FreeRunRow,
)
from rc_basics_lab.experiment.report import (
    META_JSON,
    DataclassSummaryMixin,
    write_meta_for,
)
from rc_basics_lab.experiment.runner import ESN_METHOD, ResultRow
from rc_basics_lab.experiment.stability import (
    CAPACITY_CSV,
    STABILITY_CSV,
    StabilityResults,
    regime_counts,
    run_stability_experiment,
    valid_time_by_regime,
    write_stability_csv,
)
from rc_basics_lab.experiment.valid_time import VALID_TIME_THRESHOLD_GRID
from rc_basics_lab.tasks.chaotic import sampling_interval
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

EXPERIMENT_LORENZ_LYAPUNOV = "4A_lorenz_lyapunov"
"""``diagnostics.csv`` の ``experiment`` 列 (真の Lorenz 系の lambda_max)。"""

FIG_FREERUN_ATTRACTOR = "fig_freerun_attractor.png"
FIG_FREERUN_TIMELINE = "fig_freerun_timeline.png"
FIG_VALID_TIME = "fig_valid_time.png"
FIG_STABILITY_MAP = "fig_stability_map.png"
FIG_FREERUN_STATS = "fig_freerun_stats.png"

FREERUN_ARTIFACTS: tuple[str, ...] = (
    ONESTEP_CSV,
    FREERUN_CSV,
    FREERUN_PROFILE_CSV,
    STABILITY_CSV,
    CAPACITY_CSV,
    DIAGNOSTICS_CSV,
    FIG_FREERUN_ATTRACTOR,
    FIG_FREERUN_TIMELINE,
    FIG_VALID_TIME,
    FIG_STABILITY_MAP,
    FIG_FREERUN_STATS,
    META_JSON,
)
"""1コマンド (``make figures-04``) で必ず出る 04 の成果物。

並びは 02・03 と同じく「CSV -> 図 -> meta.json」で、
``run_and_report_freerun`` が返す ``paths`` の順序と一致する。宣言と実体が
食い違ったら落ちるテストがこの並びを見る。

CSV が5枚あるのは仕様 §4 T5-4 の3枚より多い。増えた2枚には理由がある:

- ``freerun_profile.csv``: 位相図・リターンマップ・スペクトルは「行」ではなく
  「配列」だが、**図は成果物 CSV の行だけを読む** (仕様 §5 禁止する構造7) ので
  書き出す先が要る (03 の ``capacity_profile.csv`` と同じ役割)
- ``capacity.csv``: 4-D は 03 の接ぎ目 (``capacity_row_from``) をそのまま使う
  ので行の形は ``CapacityRow`` (約35列) である。03 の
  ``results/03_capacity/capacity.csv`` はバイト不変でなければならないから、
  同じ列のまま 04 のディレクトリへ出す (D-51)
"""


@dataclass(frozen=True, slots=True)
class SectionTiming(DataclassSummaryMixin):
    """区間ごとの実測時間 (``meta.json`` の ``wall_time_breakdown``)。

    仕様 §5 が区間ごとに予算を切っているので、成果物だけで「どの区間が予算を
    割ったか」を判定できる形にする。

    Attributes:
        lyapunov_s: 真の軌道の生成 + lambda_max の推定 (予算 < 60 秒)。
        onestep_s: 4-A (予算 < 120 秒)。
        freerun_s: 4-B (予算 < 240 秒)。
        stability_s: 4-C (予算 < 300 秒。**4-D を含まない**)。
        capacity_s: 4-D の MC + IPC の合計 (予算 < 150 秒)。
        figures_s: 図5枚 (予算 < 20 秒)。
    """

    lyapunov_s: float
    onestep_s: float
    freerun_s: float
    stability_s: float
    capacity_s: float
    figures_s: float


@dataclass(frozen=True, slots=True)
class FreeRunOutputs:
    """``run_and_report_freerun`` の成果物。

    Attributes:
        onestep_rows: 4-A の行 (``onestep.csv``)。
        freerun: 4-B の結果 (``freerun.csv`` / ``freerun_profile.csv``)。
        stability: 4-C + 4-D の結果 (``stability.csv`` / ``capacity.csv``)。
        timing: 区間ごとの実測時間。
        paths: 生成したファイル (``FREERUN_ARTIFACTS`` と同じ並び)。
        wall_time_s: 計算部分の実測 wall time (図と書き出しを含む)。
    """

    onestep_rows: tuple[ResultRow, ...]
    freerun: FreeRunResults
    stability: StabilityResults
    timing: SectionTiming
    paths: tuple[Path, ...]
    wall_time_s: float

    @property
    def freerun_rows(self) -> tuple[FreeRunRow, ...]:
        """``freerun.csv`` と同じ行。"""
        return self.freerun.rows

    @property
    def capacity_rows(self) -> tuple[CapacityRow, ...]:
        """``capacity.csv`` (04) と同じ行。"""
        return self.stability.capacity_rows


def _log_timing(timing: SectionTiming, wall_time_s: float) -> None:
    """区間ごとの実測時間を**数値として**ログに残す (仕様 §5 の性能予算)。"""
    logger.info(
        "04 の区間別 wall time: lambda_max %.1fs / 4-A %.1fs / 4-B %.1fs / "
        "4-C %.1fs (うち 4-D %.1fs) / 図 %.1fs / 合計 %.1fs",
        timing.lyapunov_s,
        timing.onestep_s,
        timing.freerun_s,
        timing.stability_s,
        timing.capacity_s,
        timing.figures_s,
        wall_time_s,
    )


def _timeline_source(results: FreeRunResults) -> FreeRunEvaluation | None:
    """時系列図に使う 1 本を選ぶ (D-107)。

    **選び方を固定する**: Lorenz の ESN、レプリケートは
    ``WAVEFORM_REPLICATE``。うまくいった 1 本を選べる図にしない。

    ``plotting`` は関数内 import にする (D-53)。
    """
    from rc_basics_lab.plotting.waveforms import WAVEFORM_REPLICATE

    for evaluation in results.evaluations:
        row = evaluation.row
        if (
            row.task == "lorenz"
            and row.method == ESN_METHOD
            and row.replicate == WAVEFORM_REPLICATE
        ):
            return evaluation
    return None


def _timeline_inputs(
    results: FreeRunResults,
) -> tuple[FloatArray, FloatArray, str, float, float]:
    """時系列図に渡す (真値, 自走, 手法, 有効ステップ, Lyapunov 時間) を組む。

    ``plotting`` の型に触れないのは D-53 のためである。図を描くのは
    呼び出し側で、ここは**選び方と単位だけ**を決める。

    Args:
        results: 4-B の自走の結果。

    Returns:
        ``plot_freerun_timeline`` にそのまま渡せる 5 つ組。

    Raises:
        ValueError: 固定した 1 本 (lorenz / esn) が見つからない場合。
    """
    evaluation = _timeline_source(results)
    if evaluation is None:
        raise ValueError("時系列図に使う自走が見つかりません (lorenz / esn)")
    # **truth_aligned** を使う (truth_series は系列全体で長さが合わない)。
    # 自走は free_run_steps ぶん回るが真値は評価区間ぶんしか無いので、
    # 短いほうに合わせる (実測: truth 2000 / predicted 20000)。
    trajectory = evaluation.trajectory[:, 0]
    truth = evaluation.truth_aligned[:, 0]
    length = int(min(trajectory.size, truth.size))
    row = evaluation.row
    # **valid_time はステップではなく時間単位である。** 縦線はステップ軸に
    # 引くので valid_time_steps を使う (実測: 混同して 6 ステップ目に線が出た
    # —— 図の上では ~700 ステップで外れており、そこで気づいた)。
    steps_per_lyapunov = (
        row.valid_time_steps / row.valid_time_lyapunov
        if row.valid_time_lyapunov
        else float("nan")
    )
    return (
        truth[:length],
        trajectory[:length],
        row.method,
        float(row.valid_time_steps),
        steps_per_lyapunov,
    )


def run_and_report_freerun(config: Chaos04Config, out_dir: Path) -> FreeRunOutputs:
    """実験 4-A / 4-B / 4-C / 4-D を実行し、CSV5枚・図5枚・meta.json を書く。

    Args:
        config: 04 の実験設定。
        out_dir: 出力ディレクトリ (無ければ作る)。

    Returns:
        生成した結果・区間ごとの実測時間・ファイルパス・実測 wall time。

    Raises:
        ValueError: 確保軸を超える設定、または課題・診断側の値域違反。
    """
    # 作図層の import を関数本体に置くのは D-53 (循環 import の解消)。
    # 先頭へ戻すと tests/test_layer_boundaries.py の AST guard と
    # subprocess guard の両方が落ちる。
    from rc_basics_lab.meta import git_commit
    from rc_basics_lab.plotting.figures_freerun import (
        plot_freerun_attractor,
        plot_freerun_stats,
        plot_valid_time,
    )
    from rc_basics_lab.plotting.figures_freerun_time import plot_freerun_timeline
    from rc_basics_lab.plotting.figures_stability import plot_stability_map
    from rc_basics_lab.plotting.style import setup_style

    started = time.perf_counter()
    lyapunov_started = time.perf_counter()
    lyapunov = estimate_lorenz_lyapunov(config)
    lyapunov_s = time.perf_counter() - lyapunov_started

    onestep_started = time.perf_counter()
    onestep_rows = tuple(run_onestep(config))
    onestep_s = time.perf_counter() - onestep_started

    freerun = run_freerun_experiment(config, lyapunov)
    stability = run_stability_experiment(config, lyapunov)

    figures_started = time.perf_counter()
    # commit は meta.json と図の footnote (FIG-6 / D-87) で同じ値を使う。
    style = setup_style(commit=git_commit())
    truth, trajectory, method, valid_steps, lyapunov_time = _timeline_inputs(freerun)
    paths = (
        write_onestep_csv(onestep_rows, out_dir / ONESTEP_CSV),
        write_freerun_csv(freerun.rows, out_dir / FREERUN_CSV),
        write_freerun_profile_csv(freerun.profile_rows, out_dir / FREERUN_PROFILE_CSV),
        write_stability_csv(stability.rows, out_dir / STABILITY_CSV),
        # 列順は 03 の CAPACITY_CSV_COLUMNS (= CapacityRow の宣言順) をそのまま
        # 使う。04 専用の書き出しを作ると列順の単一の真実が2つになる。
        write_capacity_csv(stability.capacity_rows, out_dir / CAPACITY_CSV),
        # 診断のスカラは長形式へ (D-118)。lambda_max は条件を振らない全体の量
        # なので condition_id は空にする (軸を振っていない = 軸を書かない)。
        write_diagnostics_csv(
            scalar_rows(
                (lyapunov,),
                experiment=EXPERIMENT_LORENZ_LYAPUNOV,
                condition_id="",
                replicate=0,
            )
            + stability.diagnostics,
            out_dir,
        ),
        # FIG-12: 4-A の 6 点は単独図をやめ、位相図と同じ figure のパネルへ。
        plot_freerun_attractor(
            freerun.profile_rows,
            out_dir / FIG_FREERUN_ATTRACTOR,
            onestep_rows=onestep_rows,
            style=style,
        ),
        plot_valid_time(freerun.rows, out_dir / FIG_VALID_TIME, style=style),
        # FIG-11 追加図4 (D-107)。位相図と valid time はあるのに
        # **時間軸の図が無かった** —— 「いつ外れるか」は時間軸でしか見えない。
        plot_freerun_timeline(
            truth,
            trajectory,
            out_dir / FIG_FREERUN_TIMELINE,
            method=method,
            task_label=("Lorenz", "Lorenz"),
            valid_steps=valid_steps,
            lyapunov_time=lyapunov_time,
            style=style,
        ),
        plot_stability_map(
            stability.rows,
            stability.capacity_rows,
            out_dir / FIG_STABILITY_MAP,
            style=style,
        ),
        plot_freerun_stats(
            freerun.profile_rows,
            freerun.rows,
            out_dir / FIG_FREERUN_STATS,
            style=style,
        ),
    )
    figures_s = time.perf_counter() - figures_started

    timing = SectionTiming(
        lyapunov_s=lyapunov_s,
        onestep_s=onestep_s,
        freerun_s=freerun.wall_time_s,
        stability_s=stability.wall_time_s - stability.wall_time_capacity_s,
        capacity_s=stability.wall_time_capacity_s,
        figures_s=figures_s,
    )
    wall_time_s = time.perf_counter() - started
    _log_timing(timing, wall_time_s)
    meta_path = write_meta_for(
        config,
        config.base.seeds,
        wall_time_s,
        len(onestep_rows),
        out_dir / META_JSON,
        extra=_meta_extra(config, freerun, stability, lyapunov, timing, style.cjk_font),
    )
    logger.info(
        "04 の成果物を書きました: %s (4-A %d 行 / 4-B %d 行 / 4-C %d 行 / "
        "4-D %d 行 / profile %d 行, wall_time=%.1fs)",
        [path.name for path in (*paths, meta_path)],
        len(onestep_rows),
        len(freerun.rows),
        len(stability.rows),
        len(stability.capacity_rows),
        len(freerun.profile_rows),
        wall_time_s,
    )
    return FreeRunOutputs(
        onestep_rows=onestep_rows,
        freerun=freerun,
        stability=stability,
        timing=timing,
        paths=(*paths, meta_path),
        wall_time_s=wall_time_s,
    )


CAPACITY_NOTE = (
    "注: 4-D の MC / IPC は自走と同じ状態行列 (Lorenz で駆動した ESN) に対して"
    "測っている。駆動が i.i.d. ではなく決定的な時系列なので遅延目標が互いに"
    "予測可能であり、mc_total / ipc_total は保存則 (<= N) を超える。"
    "絶対値は 03 の掃引と比較できず、読めるのは同じ駆動の下での条件間の相対"
    "比較だけである。"
)
"""``meta.json`` と ``docs/design.md`` §12 に載せる 4-D の但し書き。

数字だけを孤立して残すと、後から「保存則が破れている」とだけ読まれる。
"""


def _meta_extra(
    config: Chaos04Config,
    freerun: FreeRunResults,
    stability: StabilityResults,
    lyapunov: DiagnosticResult,
    timing: SectionTiming,
    cjk_font: str | None,
) -> dict[str, object]:
    """``meta.json`` の追加項目 (**受け入れ条件の一次資料をここに集める**)。"""
    return {
        "lorenz_dt": sampling_interval(config.lorenz),
        "lyapunov": dict(lyapunov.scalars),
        "lyapunov_params": dict(lyapunov.params),
        "wall_time_breakdown": timing.to_summary(),
        "n_freerun_rows": len(freerun.rows),
        "n_profile_rows": len(freerun.profile_rows),
        "n_stability_rows": len(stability.rows),
        "n_capacity_rows": len(stability.capacity_rows),
        # 受け入れ条件2 / D-43: 閾値を変えても結論が変わらないことの一次資料。
        "valid_time_threshold_grid": list(VALID_TIME_THRESHOLD_GRID),
        "valid_time_sensitivity": [item.to_summary() for item in freerun.sensitivity],
        # 受け入れ条件1 / 5 / D-46: シャッフル代替との比較 (図ではなく数値)。
        "attractor_verdict": [item.to_summary() for item in freerun.attractor],
        # 受け入れ条件4 / D-45: 3態の集計と、状態ノイズ量ごとの内訳。
        "regime_counts": regime_counts(stability.rows),
        "regime_counts_by_noise": {
            f"{noise:g}": regime_counts(
                [row for row in stability.rows if row.state_noise == noise]
            )
            for noise in sorted({row.state_noise for row in stability.rows})
        },
        "valid_time_by_regime": valid_time_by_regime(stability.rows),
        # 4-D の測定上の限界 (数字だけを孤立して残さない)。
        "capacity_note": CAPACITY_NOTE,
        # 図のラベル言語を決めた要因 (02・03 の meta.json と同じ形)。
        "cjk_font": cjk_font,
    }


__all__ = [
    "CAPACITY_NOTE",
    "FIG_FREERUN_ATTRACTOR",
    "FIG_FREERUN_STATS",
    "FIG_STABILITY_MAP",
    "FIG_VALID_TIME",
    "FREERUN_ARTIFACTS",
    "FreeRunOutputs",
    "SectionTiming",
    "run_and_report_freerun",
]
