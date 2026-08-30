"""各図のパネル数 (軸の本数) を実測する (FIG-15 / ``make panels``).

``Figure.savefig`` を捕まえて ``len(figure.axes)`` を数える。**コードの読み取り
ではなく、本番設定で実際に生成される図そのもの**を数えるので、条件数に依存して
パネルが増える図 (rho 別のヒートマップなど) も正しく出る。

出力先は一時ディレクトリで、``results/`` には触れない。**``ci`` には入れない**
—— 本番設定の全実験を回すので約20分かかり、「CI は実験を回さない」という分担が
壊れる。``docs/series/図の設計方針_RC基礎編.md`` FIG-15 の表を更新するときに
手で回す。
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib.figure

logger = logging.getLogger("count_panels")

_RECORDS: list[tuple[str, int]] = []
_ORIGINAL_SAVEFIG = matplotlib.figure.Figure.savefig


def _spy(
    self: matplotlib.figure.Figure, fname: str | Path, *, format: str | None = None
) -> None:
    """保存の直前にパネル数を記録してから本来の ``savefig`` を呼ぶ。

    署名を ``plotting/style.py`` の唯一の呼び出し
    (``figure.savefig(path, format="png")``) に合わせてある。別の引数で呼ぶ
    経路が生えたら ``TypeError`` で落ちる —— 黙って素通しして「数えたつもり」に
    なるより、そこで気づくほうがよい。
    """
    _RECORDS.append((Path(fname).name, len(self.axes)))
    _ORIGINAL_SAVEFIG(self, fname, format=format)


matplotlib.figure.Figure.savefig = _spy  # type: ignore[method-assign,assignment]

from rc_basics_lab.config import (  # noqa: E402
    Anomaly05Config,
    Capacity03Config,
    Chaos04Config,
    Esp02Config,
    load_config,
    load_config_as,
)
from rc_basics_lab.experiment.anomaly_pipeline import (  # noqa: E402
    run_and_report_anomaly,
)
from rc_basics_lab.experiment.capacity_pipeline import (  # noqa: E402
    run_and_report_capacity,
)
from rc_basics_lab.experiment.esp_pipeline import run_and_report_esp  # noqa: E402
from rc_basics_lab.experiment.freerun_pipeline import (  # noqa: E402
    run_and_report_freerun,
)
from rc_basics_lab.experiment.pipeline import run_and_report  # noqa: E402

EXPERIMENTS = Path("experiments")

JOBS: tuple[tuple[str, Callable[[Path], object]], ...] = (
    (
        "01",
        lambda out: run_and_report(
            load_config(EXPERIMENTS / "01_what_is_rc" / "config.yaml"), out
        ),
    ),
    (
        "02",
        lambda out: run_and_report_esp(
            load_config_as(
                EXPERIMENTS / "02_esp_and_dynamics" / "config.yaml", Esp02Config
            ),
            out,
        ),
    ),
    (
        "03",
        lambda out: run_and_report_capacity(
            load_config_as(
                EXPERIMENTS / "03_capacity" / "config.yaml", Capacity03Config
            ),
            out,
        ),
    ),
    (
        "04",
        lambda out: run_and_report_freerun(
            load_config_as(
                EXPERIMENTS / "04_chaotic_freerun" / "config.yaml", Chaos04Config
            ),
            out,
        ),
    ),
    (
        "05",
        lambda out: run_and_report_anomaly(
            load_config_as(
                EXPERIMENTS / "05_anomaly_detection" / "config.yaml", Anomaly05Config
            ),
            out,
        ),
    ),
)
"""記事ごとの本番パイプライン。記事を1本足したらここへ1行足す。"""


def main() -> int:
    """全記事の図を一時ディレクトリへ描き、パネル数の表をログに出す。"""
    measured: dict[str, list[tuple[str, int]]] = {}
    with tempfile.TemporaryDirectory(prefix="rc-panels-") as tmp:
        for name, job in JOBS:
            _RECORDS.clear()
            job(Path(tmp) / name)
            measured[name] = list(_RECORDS)
            logger.info(
                "[%s] %d 枚 / 合計 %d パネル",
                name,
                len(measured[name]),
                sum(count for _, count in measured[name]),
            )
            for figure_name, count in measured[name]:
                logger.info("    %-34s %d パネル", figure_name, count)

    logger.info("")
    logger.info("| 記事 | 枚数 | 総パネル | 1枚の最大 |")
    logger.info("|---|---|---|---|")
    for name, items in measured.items():
        counts = [count for _, count in items]
        logger.info("| %s | %d | %d | %d |", name, len(items), sum(counts), max(counts))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
