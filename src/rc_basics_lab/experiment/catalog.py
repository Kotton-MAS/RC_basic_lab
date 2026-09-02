"""実験の宣言を1箇所に置く (D-125).

``CATALOG`` が「どの実験を、どの設定で、どう走らせ、どこへ書くか」の唯一の真実で、
``main.py`` も ``Makefile`` の ``figures-%`` もここを読む。実験を1本足すのは
ここへ ``ExperimentSpec`` を1エントリ (手順は ``docs/guide/実験を足す.md``)。

走らせ方の切り替えは ``variants`` の辞書で行う (``--variant length``)。
真偽フラグにしない理由は D-125 にある。
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rc_basics_lab.cli import default_out_for
from rc_basics_lab.config import (
    Anomaly05Config,
    Capacity03Config,
    Chaos04Config,
    Esp02Config,
    load_config,
    load_config_as,
)
from rc_basics_lab.experiment.anomaly_pipeline import (
    ANOMALY_ARTIFACTS,
    run_and_report_anomaly,
)
from rc_basics_lab.experiment.capacity_pipeline import (
    CAPACITY_ARTIFACTS,
    run_and_report_capacity,
    run_and_report_length_sweep,
    run_and_report_narma10_operating,
    run_and_report_symmetry_sweep,
)
from rc_basics_lab.experiment.esp_pipeline import (
    ESP_ARTIFACTS,
    run_and_report_esp,
    run_and_report_threshold_sweep,
    write_washout_csv,
)
from rc_basics_lab.experiment.freerun_pipeline import (
    FREERUN_ARTIFACTS,
    run_and_report_freerun,
)
from rc_basics_lab.experiment.ladder_pipeline import (
    run_and_report_ladder_threshold,
    run_and_report_topology_ladder,
)
from rc_basics_lab.experiment.pipeline import ARTIFACTS, run_and_report
from rc_basics_lab.experiment.washout import run_washout_sweep

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = REPO_ROOT / "results"

MAIN = "main"
"""既定の variant 名。``--variant`` を省いたときに走るもの。"""


@dataclass(frozen=True, slots=True)
class RunRequest:
    """1回の実行に必要なものだけ。

    Attributes:
        config: 読む設定 YAML。
        out: 成果物の書き出し先。
        preset: かぶせるプリセット (``--preset``)。
        overrides: ``--set`` の並び。
    """

    config: Path
    out: Path
    preset: Path | None = None
    overrides: Sequence[str] = ()


type Variant = Callable[[RunRequest], None]
"""1つの走らせ方。**設定の読み込みまで含めて閉じる**。

実験ごとに設定の型が違う (``ExperimentConfig`` / ``Esp02Config`` / …) ので、
ローダとパイプラインを組にして持つと単一の型で書けない。読み込みを関数の
内側に入れると、カタログは1つの型で並べられる。
"""


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """実験1本ぶんの宣言 (**ここが唯一の真実**)。

    Attributes:
        number: ``--experiment`` に渡す番号 (``"03"``)。
        name: ディレクトリ名 (``"03_capacity"``)。
        results_dir: ``make figures-0N`` が書く先。01 だけ ``results/`` 直下
            なので、名前から導かずに宣言する。
        artifacts: 1コマンドで必ず出る成果物のファイル名。
        budget_s: 予算 [秒]。超えたら設計の見直しを要求する目安。
        variants: 走らせ方。``MAIN`` は必須。
    """

    number: str
    name: str
    results_dir: Path
    artifacts: tuple[str, ...]
    budget_s: float
    variants: Mapping[str, Variant] = field(default_factory=dict)

    @property
    def config_path(self) -> Path:
        """既定の設定 YAML (``experiments/<name>/config.yaml``)。"""
        return EXPERIMENTS_DIR / self.name / "config.yaml"

    @property
    def scratch_dir(self) -> Path:
        """``--out`` も ``--results`` も無いときの書き出し先 (手元用)。"""
        return default_out_for(self.config_path)

    def variant(self, name: str) -> Variant:
        """走らせ方を1つ取り出す。**無ければ候補を並べて落とす**。

        Raises:
            ValueError: その variant が無い場合。
        """
        try:
            return self.variants[name]
        except KeyError:
            known = ", ".join(sorted(self.variants))
            raise ValueError(
                f"実験 {self.number} に variant {name!r} はありません (既知: {known})"
            ) from None


def _run_01(request: RunRequest) -> None:
    """実験01 (3ベースラインの比較 + 状態空間 PCA)。"""
    config = load_config(
        request.config, preset=request.preset, overrides=request.overrides
    )
    logger.info(
        "実験01 を実行します: %s (n_replicates=%d)", request.config, config.n_replicates
    )
    run_and_report(config, request.out)


def _load_02(request: RunRequest) -> Esp02Config:
    return load_config_as(
        request.config, Esp02Config, preset=request.preset, overrides=request.overrides
    )


def _run_02(request: RunRequest) -> None:
    """実験02 (ESP・スペクトル半径・リーク率)。"""
    config = _load_02(request)
    logger.info(
        "実験02 を実行します: %s (n_units=%d, n_steps=%d, n_replicates=%d)",
        request.config,
        config.reservoir.n_units,
        config.drive.n_steps,
        config.reservoir.n_replicates,
    )
    run_and_report_esp(config, request.out)


def _run_02_threshold(request: RunRequest) -> None:
    """2-C の判定閾値の感度だけを回す。"""
    run_and_report_threshold_sweep(_load_02(request), request.out)


def _run_02_washout_unpadded(request: RunRequest) -> None:
    """系列長を補償しない washout 掃引 (D-19 の対照)。

    かつては ``Makefile`` に埋め込んだ 9 行の python -c だった。**Makefile に
    書いた計算は、テストからもレビューからも見えない**。
    """
    config = _load_02(request)
    unpadded = dataclasses.replace(
        config, washout=dataclasses.replace(config.washout, pad_series=False)
    )
    write_washout_csv(
        run_washout_sweep(unpadded), request.out / "washout_sensitivity_unpadded.csv"
    )


def _load_03(request: RunRequest) -> Capacity03Config:
    return load_config_as(
        request.config,
        Capacity03Config,
        preset=request.preset,
        overrides=request.overrides,
    )


def _run_03(request: RunRequest) -> None:
    """実験03 (メモリ容量・情報処理容量)。"""
    config = _load_03(request)
    logger.info(
        "実験03 を実行します: %s (3-A N=%d / 3-B N=%d / n_replicates=%d)",
        request.config,
        config.mc_sweep.n_units,
        config.ipc_sweep.n_units,
        config.reservoir.n_replicates,
    )
    run_and_report_capacity(config, request.out)


def _run_03_length(request: RunRequest) -> None:
    """系列長 T の掃引だけを回す (``capacity_length.csv``)。"""
    run_and_report_length_sweep(_load_03(request), request.out)


def _run_03_operating(request: RunRequest) -> None:
    """動作点の掃引だけを回す (``narma10_operating.csv``、3-C'' / D-144)。"""
    run_and_report_narma10_operating(_load_03(request), request.out)


def _run_03_ladder_threshold(request: RunRequest) -> None:
    """閾値感度だけを回す (``capacity_topology_threshold.csv``、3-Th / D-143)。"""
    run_and_report_ladder_threshold(_load_03(request), request.out)


def _run_03_symmetry(request: RunRequest) -> None:
    """駆動入力の対称性の掃引だけを回す (``capacity_symmetry.csv``、D-116)。"""
    run_and_report_symmetry_sweep(_load_03(request), request.out)


def _run_03_ladder(request: RunRequest) -> None:
    """対照の梯子だけを回す (``capacity_topology.csv``、3-T / D-138)。"""
    run_and_report_topology_ladder(_load_03(request), request.out)


def _run_04(request: RunRequest) -> None:
    """実験04 (カオス時系列の自由走行予測)。"""
    config = load_config_as(
        request.config,
        Chaos04Config,
        preset=request.preset,
        overrides=request.overrides,
    )
    logger.info(
        "実験04 を実行します: %s (Lorenz T=%d dt=%g / n_replicates=%d)",
        request.config,
        config.lorenz.length,
        config.lorenz.rk4_step * config.lorenz.sample_interval,
        config.base.n_replicates,
    )
    run_and_report_freerun(config, request.out)


def _run_05(request: RunRequest) -> None:
    """実験05 (センサー時系列の異常検知)。"""
    config = load_config_as(
        request.config,
        Anomaly05Config,
        preset=request.preset,
        overrides=request.overrides,
    )
    logger.info(
        "実験05 を実行します: %s (source=%s / 系列 %d 本 / max_length=%d / "
        "n_replicates=%d)",
        request.config,
        config.dataset.source,
        len(config.dataset.series),
        config.dataset.max_length,
        config.reservoir.n_replicates,
    )
    run_and_report_anomaly(config, request.out)


CATALOG: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        number="01",
        name="01_what_is_rc",
        results_dir=RESULTS_DIR,
        artifacts=ARTIFACTS,
        budget_s=120.0,
        variants={MAIN: _run_01},
    ),
    ExperimentSpec(
        number="02",
        name="02_esp_and_dynamics",
        results_dir=RESULTS_DIR / "02_esp_and_dynamics",
        artifacts=ESP_ARTIFACTS,
        budget_s=600.0,
        variants={
            MAIN: _run_02,
            "threshold": _run_02_threshold,
            "washout-unpadded": _run_02_washout_unpadded,
        },
    ),
    ExperimentSpec(
        number="03",
        name="03_capacity",
        results_dir=RESULTS_DIR / "03_capacity",
        artifacts=CAPACITY_ARTIFACTS,
        budget_s=900.0,
        variants={
            MAIN: _run_03,
            "length": _run_03_length,
            "symmetry": _run_03_symmetry,
            "ladder": _run_03_ladder,
            "ladder-threshold": _run_03_ladder_threshold,
            "operating": _run_03_operating,
        },
    ),
    ExperimentSpec(
        number="04",
        name="04_chaotic_freerun",
        results_dir=RESULTS_DIR / "04_chaotic_freerun",
        artifacts=FREERUN_ARTIFACTS,
        budget_s=900.0,
        variants={MAIN: _run_04},
    ),
    ExperimentSpec(
        number="05",
        name="05_anomaly_detection",
        results_dir=RESULTS_DIR / "05_anomaly_detection",
        artifacts=ANOMALY_ARTIFACTS,
        budget_s=900.0,
        variants={MAIN: _run_05},
    ),
)
"""全実験の宣言。**実験を足すのはここへ1エントリ**。

並び順が ``--experiment`` の候補の並び順になる。
"""

BY_NUMBER: Mapping[str, ExperimentSpec] = {spec.number: spec for spec in CATALOG}
"""番号で引く索引 (``main.py`` が使う)。"""


def spec_for(number: str) -> ExperimentSpec:
    """番号から宣言を引く。**無ければ候補を並べて落とす**。

    Raises:
        ValueError: その番号の実験が無い場合。
    """
    try:
        return BY_NUMBER[number]
    except KeyError:
        known = ", ".join(sorted(BY_NUMBER))
        raise ValueError(f"実験 {number!r} はありません (既知: {known})") from None


__all__ = [
    "BY_NUMBER",
    "CATALOG",
    "MAIN",
    "ExperimentSpec",
    "RunRequest",
    "Variant",
    "spec_for",
]
