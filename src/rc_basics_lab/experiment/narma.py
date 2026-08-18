"""実験 3-C —— 公平な対照での NARMA10 (D-31).

**01 の ``run_task`` をそのまま通す**。手法の列挙 (``build_methods``)・alpha 格子
の共有 (D-04)・全手法が同一の行 index で学習評価すること (D-05)・ESN の構造
ハイパーパラメータを検証分割で選ばないこと (D-08) は、すべて 01 の経路が
担保している。3-C 側で書き写すと、公平性の3決定が2箇所に分かれて片方だけ
劣化しうる。ここが組み立てるのは ``TaskEntry`` (課題の生成関数 + ESN 設定) 1つ
だけで、``build_tasks`` にも ``ExperimentConfig`` にも NARMA10 を足さない
(足すと 01 の ``comparison.csv`` に行が増えて 01 の成果物が変わる)。

**レプリケート0 の ``ReplicatePlan`` はここで1回だけ作り、``run_task`` へ
``plan0=`` で渡す** (01 の ``pipeline.py`` が ``collect_state_space`` に対して
やっているのと同じ形、F-1-009)。こうすると「NARMA10 を解いた ESN」と
「容量 (MC / IPC) を測った ESN」が**同一のリザバー・同一の状態行列**である
ことが構造で保証される。``plan_replicate`` を2回呼ぶ実装でも同じシードから
同じ値にはなるが、それは「たまたま一致している」だけで、片方の経路の
シードや ``n_steps`` を変えたときに黙って別物になる。

容量の行は ``capacity.csv`` に ``experiment="3C_narma10"`` として合流させる
(``measure_capacity`` -> ``capacity_row_from`` -> ``capacity_outcome_from``、
F-3b1-1-004)。**行の組み立てを複製しない** —— ``CapacityRow`` は約35列あり、
複製すると列を1本足したときに片方が置き去りになる。成績 (``narma10.csv``、
01 の ``ResultRow``) と容量 (``capacity.csv``) は条件キーで join でき、
「NARMA10 の成績が容量のどの成分と相関するか」(要件書 実験3-C) を見られる。
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rc_basics_lab.config import Capacity03Config, ESNConfig, ExperimentConfig
from rc_basics_lab.diagnostics.base import DiagnosticContext
from rc_basics_lab.experiment.capacity import (
    EXPERIMENT_NARMA10,
    CapacityOutcome,
    capacity_outcome_from,
    capacity_row_from,
    ipc_config_for,
    measure_capacity,
)
from rc_basics_lab.experiment.runner import (
    DELAY_LINE,
    ESN_METHOD,
    ReplicatePlan,
    ResultRow,
    TaskEntry,
    plan_replicate,
    run_task,
)
from rc_basics_lab.tasks.narma import (
    NARMA10_INPUT_STD,
    TASK_NAME,
    generate_narma10,
)

logger = logging.getLogger(__name__)

NARMA10_ESN_SECTION = "esn_mackey_glass"
"""3-C が読む ESN セクション名 (D-39 の N=50 を適用するのはこの1本)。

``narma.base`` は 01 の ``ExperimentConfig`` をまるごと内包しているので ESN
セクションが2本 (``esn_mackey_glass`` / ``esn_delay_parity``) 在るが、3-C の
課題は NARMA10 の1本だけなので読むのはどちらか一方である。**Mackey-Glass 側**
を選ぶ理由は、NARMA10 が連続値の入力 (``u ~ U[0, 0.5]``) を受けて連続値を
出す回帰課題であり、漏れ積分 (``leak_rate=0.3``) が効く点で MG と同型だから
である (``esn_delay_parity`` は ±1 の2値入力・``leak_rate=1.0`` を前提とした
設定で、10 ステップの記憶を要する NARMA10 とは動作点が違う)。

D-08 により ESN の構造ハイパーパラメータは検証分割で選ばれないので、
**宣言した1点をそのまま報告する**。この定数は宣言であって切り替えスイッチ
ではない (実際に読む属性は ``narma_esn_config``、両者の一致は
``test_narma10_esn_size_matches_the_declared_choice`` が固定する)。
"""

NARMA10_REFERENCE_NMSE: Mapping[str, float] = {
    "linear_ceiling": 0.16,
    "nonlinear_rc": 0.107,
}
"""記事が引く参照点 (要件書 実験3-C)。**原典は未特定**。

複数の物理 RC 論文が「非線形性が全く無い場合の NARMA10 最良 NMSE ≈ 0.16」を
引用し、良好な非線形 RC (50ノード級) で ≈ 0.107 とされている。要件書 未確定1
が「原典特定」を残件にしているため、図の注と ``meta.json`` の両方に
**原典未特定である**ことを明記して引く (数字だけを孤立して引くと、後から
出典が違っていたときに成果物の側から辿れない)。
"""

NARMA10_REFERENCE_NOTE = (
    "注: 参照値 NMSE = 0.16 (非線形性なしの天井) / 0.107 (良好な非線形 RC、"
    "50ノード級) は複数の物理 RC 論文が引用する値で、原典は未特定である "
    "(要件_rc-basics-03 未確定1)。"
)
"""``meta.json`` と図の注に載せる、参照線の出所についての但し書き (日本語)。"""

NARMA10_REFERENCE_NOTE_EN = (
    "Reference NMSE = 0.16 (ceiling without nonlinearity) / 0.107 (a good "
    "nonlinear RC with ~50 nodes) are values quoted by several physical-RC "
    "papers; the primary source is NOT identified (requirements, open item 1)."
)
"""同じ但し書きの英語版 (CJK フォントが無い環境の図に出る、D-10)。"""


def narma_esn_config(base: ExperimentConfig) -> ESNConfig:
    """3-C が使う ESN 設定を返す (``NARMA10_ESN_SECTION`` の1本)。

    「どのセクションを読むか」をここ1か所に閉じる。呼び出し側が属性を直接
    書くと、D-39 (N=50) を適用したセクションと実際に読むセクションが
    食い違っても何も落ちない。
    """
    return base.esn_mackey_glass


def narma_task_entry(config: Capacity03Config) -> TaskEntry:
    """3-C の ``TaskEntry`` を組む (**``build_tasks`` には足さない**、D-31)。

    ``run_task`` は ``task_entry`` を引数に取るので、01 の公平性機構
    (D-04 / D-05 / D-08) を1行も書き写さずに 3-C へ効かせられる。
    """
    narma = config.narma
    return TaskEntry(
        name=TASK_NAME,
        esn=narma_esn_config(narma.base),
        generate=lambda rng: generate_narma10(narma, rng),
    )


@dataclass(frozen=True, slots=True)
class Narma10Verdict:
    """3手法の成績の要約 (``meta.json`` の ``narma10_verdict``)。

    仕様 §4 T4 は「**結果の向きは問わない**。遅延線が上回った場合は
    ``meta.json`` の ``narma10_verdict`` に記録」と書いている。向きを問わない
    以上、どちらに転んでも同じ形で残る器が要る —— 「ESN が勝った回だけ書く」
    形にすると、負けた回に成果物から主張が消える。

    Attributes:
        best_method: テスト NMSE のレプリケート平均が最小の手法。
        nmse_mean: 手法ごとの NMSE のレプリケート平均。
        delay_line_beats_esn: 遅延線の平均 NMSE が ESN より小さいか。
        selected_n_lags: 遅延線が検証分割で選んだタップ数 (昇順・重複なし)。
    """

    best_method: str
    nmse_mean: Mapping[str, float]
    delay_line_beats_esn: bool
    selected_n_lags: tuple[int, ...]

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "best_method": self.best_method,
            "nmse_mean": dict(self.nmse_mean),
            "delay_line_beats_esn": self.delay_line_beats_esn,
            "selected_n_lags": list(self.selected_n_lags),
            "reference_nmse": dict(NARMA10_REFERENCE_NMSE),
            "reference_note": NARMA10_REFERENCE_NOTE,
        }


def summarize_narma10(rows: Sequence[ResultRow]) -> Narma10Verdict:
    """手法ごとの NMSE 平均から勝敗を要約する (向きは問わない)。

    Raises:
        ValueError: ``rows`` が空の場合。
    """
    if not rows:
        raise ValueError("rows が空です")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.method, []).append(row.nmse)
    nmse_mean = {method: statistics.fmean(values) for method, values in grouped.items()}
    best = min(nmse_mean, key=lambda method: (nmse_mean[method], method))
    delay_line = nmse_mean.get(DELAY_LINE)
    esn = nmse_mean.get(ESN_METHOD)
    return Narma10Verdict(
        best_method=best,
        nmse_mean=nmse_mean,
        delay_line_beats_esn=(
            delay_line is not None and esn is not None and delay_line < esn
        ),
        selected_n_lags=tuple(
            sorted({row.n_lags for row in rows if row.method == DELAY_LINE})
        ),
    )


@dataclass(frozen=True, slots=True)
class Narma10Results:
    """実験 3-C の結果 (成績の行 + レプリケート0 の容量)。

    Attributes:
        rows: ``narma10.csv`` の行 (01 の ``ResultRow`` をそのまま使う)。
        capacity: レプリケート0 の状態行列に対する MC / IPC
            (``capacity.csv`` / ``capacity_profile.csv`` へ合流する)。
        verdict: 3手法の成績の要約 (``meta.json``)。
        plan0: ``run_task`` と容量測定が共有した ``ReplicatePlan``。
        wall_time_s: 3-C 全体の実測 wall time [秒]。
    """

    rows: tuple[ResultRow, ...]
    capacity: CapacityOutcome
    verdict: Narma10Verdict
    plan0: ReplicatePlan
    wall_time_s: float


def run_narma10(config: Capacity03Config) -> Narma10Results:
    """実験 3-C を1回だけ回す (成績 + 容量)。

    手順は4つで、**順序そのものが設計判断**である。

    1. ``plan_replicate`` でレプリケート0 の課題・状態・設計行列・分割を作る。
    2. ``measure_capacity`` に ``plan0.states`` を**そのまま**渡す。ここで
       状態行列が読み取り専用になる (D-35) ので、以降の手順が状態を書き換え
       ようとすれば ``ValueError`` になる (静かな desync が起きない)。
    3. ``run_task(base, entry, plan0=plan0)`` で3手法 x 全レプリケートを回す。
       レプリケート0 は 1. の ``plan0`` を再利用するので、状態行列は
       **1本しか存在しない**。
    4. ``capacity_row_from`` で ``capacity.csv`` の行を組む (複製しない)。

    ``ctx`` は掃引と同じ ``DiagnosticContext`` の作り方 (washout は 03 の
    ``drive.washout``、seed は ``seeds.surrogate``) にそろえる —— サロゲートの
    シードを全条件で共有する (D-37) 対象には 3-C も含まれる。含めないと
    3-C の容量だけ独立なしきい値推定ノイズが乗り、3-B との突き合わせが濁る。

    Args:
        config: 03 の設定 (``narma`` セクションと診断設定を読む)。

    Returns:
        成績の行・容量・要約・共有した ``ReplicatePlan``・実測 wall time。

    Raises:
        ValueError: NARMA10 が発散した (D-30) / 系列が短すぎる / 設定が範囲外の
            場合 (課題層・診断層が投げる)。
    """
    started = time.perf_counter()
    base = config.narma.base
    entry = narma_task_entry(config)

    plan0 = plan_replicate(base, entry, 0)
    wall_time_state_s = time.perf_counter() - started

    ctx = DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)
    measurement = measure_capacity(
        plan0.states,
        plan0.task.u,
        ctx=ctx,
        mc_cfg=config.mc,
        ipc_cfg=ipc_config_for(config, EXPERIMENT_NARMA10),
    )
    wall_time_capacity_s = time.perf_counter() - started

    rows = tuple(run_task(base, entry, plan0=plan0))

    esn = entry.esn
    row = capacity_row_from(
        measurement,
        experiment=EXPERIMENT_NARMA10,
        replicate=0,
        seed_reservoir=base.seeds.reservoir,
        # 3-C のリザバーを駆動するのは課題の入力そのものなので、駆動側の
        # 基底シードは task ストリーム (D-06) である。
        seed_drive=base.seeds.task,
        seed_surrogate=config.seeds.surrogate,
        rho=esn.spectral_radius,
        leak_rate=esn.leak_rate,
        input_scale=esn.input_scale,
        # 3-C に「駆動強度の設定値」は無いので、宣言した入力分布
        # U[0, 0.5] の標準偏差の閉形式を書く (実測値は input_drive_std)。
        sigma_u=NARMA10_INPUT_STD,
        n_units=esn.n_units,
        density=esn.density,
        state_noise=esn.state_noise,
        n_steps=int(plan0.states.shape[0]),
        washout=config.drive.washout,
        wall_time_state_s=wall_time_state_s,
        wall_time_s=wall_time_capacity_s,
    )
    verdict = summarize_narma10(rows)
    wall_time_s = time.perf_counter() - started
    logger.info(
        "experiment=%s 行数=%d N=%d T=%d 最良=%s NMSE=%s "
        "(遅延線が ESN を上回った=%s / タップ数=%s) "
        "mc_total=%.3f ipc_total=%.3f wall_time=%.2fs",
        EXPERIMENT_NARMA10,
        len(rows),
        esn.n_units,
        row.n_steps,
        verdict.best_method,
        {key: round(value, 5) for key, value in verdict.nmse_mean.items()},
        verdict.delay_line_beats_esn,
        verdict.selected_n_lags,
        row.mc_total,
        row.ipc_total,
        wall_time_s,
    )
    return Narma10Results(
        rows=rows,
        capacity=capacity_outcome_from(measurement, row),
        verdict=verdict,
        plan0=plan0,
        wall_time_s=wall_time_s,
    )


__all__ = [
    "NARMA10_ESN_SECTION",
    "NARMA10_REFERENCE_NMSE",
    "NARMA10_REFERENCE_NOTE",
    "NARMA10_REFERENCE_NOTE_EN",
    "Narma10Results",
    "Narma10Verdict",
    "narma_esn_config",
    "narma_task_entry",
    "run_narma10",
    "summarize_narma10",
]
