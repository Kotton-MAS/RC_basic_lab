"""再生成した成果物のバイト不変検査 (ゴールデン) の共通機構.

``tests/test_artifact_invariance.py`` (D-74) との違いを最初に書く。あちらは
**コミット済みの ``results/`` がディスク上で変わっていないこと**を測る (再生成は
しない)。したがって「リファクタリングの前後で、同じ設定から**作り直した**成果物が
1 バイトも変わらないか」は測れず、確かめるには ``make figures-01`` 〜 ``figures-05``
を回す必要がある (03 の ``saturation-03`` だけで約 30 分)。

ここはその穴を埋める。本番と**同じ ``run_and_report_*`` 経路**を通しつつ、系列長・
格子・レプリケート数だけを縮めた設定を ``tests/golden/configs/`` に置き、その出力の
ダイジェストを ``tests/golden/manifest.json`` に固定する (**数秒**)。縮小設定から
科学的な結論は出ない。ここで見るのは同一性だけである。

**実験05 は対象外**である。``anomaly_pipeline`` は外部データセット (MGAB) を
必要とし、pytest はネットワークに一切触れない (D-60)。したがってこの検査が
覆うのは実験 01〜04 の 37 成果物であり、05 の再生成不変は ``make figures-05``
でしか確認できない。

**何を除外するか** (実測に基づく): 同一設定で 2 回走らせると PNG はバイト一致し、
CSV は ``wall_time`` 列だけが差分になる。したがって除外するのは実測 wall time
(CSV の ``wall_time*`` 列 / JSON の ``wall_time*`` キー)、git コミット、
タイムスタンプ、環境情報の 4 種だけで、残りはバイト単位で比較する。
``valid_time`` や ``lyapunov_time`` は実測時間ではなく結果値なので**残す**。

**なぜ環境フィンガープリントを持つか**: 浮動小数点の最下位ビットは BLAS 実装に
依存するため、macOS で取った基準値は Linux の CI で一致しない可能性がある。
環境が違うときは照合そのものを行わない (CI では skip される)。したがってこれは
CI の不変性保証ではなく、**ローカルでリファクタの前後を突き合わせる計器**である。

基準値の更新::

    make golden-update    # uv run python tests/golden_support.py --update
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import platform
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from rc_basics_lab import __version__, meta
from rc_basics_lab.config import (
    Capacity03Config,
    Chaos04Config,
    Esp02Config,
    load_config,
    load_config_as,
)
from rc_basics_lab.experiment.capacity_pipeline import (
    run_and_report_capacity,
    run_and_report_length_sweep,
)
from rc_basics_lab.experiment.esp_pipeline import (
    run_and_report_esp,
    run_and_report_threshold_sweep,
)
from rc_basics_lab.experiment.freerun_pipeline import run_and_report_freerun
from rc_basics_lab.experiment.pipeline import run_and_report
from rc_basics_lab.plotting.style import setup_style

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
CONFIG_DIR = GOLDEN_DIR / "configs"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"

WALL_TIME_PREFIX = "wall_time"
"""実測 wall time の列名 / キー名の接頭辞 (照合から外す)。"""

VOLATILE_JSON_KEYS: frozenset[str] = frozenset({"commit", "timestamp_utc"})
"""実行のたびに変わる ``meta.json`` のキー。"""

ENVIRONMENT_JSON_KEYS: frozenset[str] = frozenset(
    {
        "package_version",
        "python_version",
        "platform",
        "numpy_version",
        "scipy_version",
        "matplotlib_version",
        "cjk_font",
    }
)
"""環境を書き写しただけの ``meta.json`` のキー (フィンガープリント側で見る)。"""

NUMERIC_FINGERPRINT_KEYS: tuple[str, ...] = (
    "platform",
    "python_version",
    "numpy_version",
    "scipy_version",
)
"""数値 (CSV / meta.json) の一致を期待してよい条件。BLAS 実装が変わると崩れる。"""

FIGURE_FINGERPRINT_KEYS: tuple[str, ...] = (
    *NUMERIC_FINGERPRINT_KEYS,
    "matplotlib_version",
    "cjk_font",
)
"""PNG の一致を期待してよい条件。描画とフォント解決が加わる。"""


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """1 実験ぶんのゴールデン。

    Attributes:
        name: ケース名 (マニフェストのキー、pytest の id)。
        config: ``tests/golden/configs/`` 配下の設定ファイル名。
        run: ``(設定ファイル, 出力ディレクトリ) -> None``。本番と同じ
            ``run_and_report_*`` を呼ぶ薄いラッパ。
    """

    name: str
    config: str
    run: Callable[[Path, Path], None]


def _run_01(config_path: Path, out_dir: Path) -> None:
    """実験01 (``make figures-01`` と同じ経路)。"""
    run_and_report(load_config(config_path), out_dir)


def _run_02(config_path: Path, out_dir: Path) -> None:
    """実験02 (``make figures-02`` + ``make threshold-02``)。"""
    config = load_config_as(config_path, Esp02Config)
    run_and_report_esp(config, out_dir)
    run_and_report_threshold_sweep(config, out_dir)


def _run_03(config_path: Path, out_dir: Path) -> None:
    """実験03 (``make figures-03`` + ``make saturation-03``)。"""
    config = load_config_as(config_path, Capacity03Config)
    run_and_report_capacity(config, out_dir)
    run_and_report_length_sweep(config, out_dir)


def _run_04(config_path: Path, out_dir: Path) -> None:
    """実験04 (``make figures-04``)。"""
    run_and_report_freerun(load_config_as(config_path, Chaos04Config), out_dir)


PINNED_COMMIT = "0" * 40
"""ゴールデン実行中に ``git_commit()`` が返す固定値。

図の footnote には HEAD の先頭7桁が焼き込まれる (FIG-6 / D-87)。実測値なので
**コミットするたびに全 PNG が変わる**。固定しないと PNG の照合は
「直前にコミットしたか」を測るだけになり、リファクタリングの合否判定に使えない。

CSV / meta.json 側の ``commit`` は ``_strip_json`` が落とすのでここには関係しない。
"""


@contextlib.contextmanager
def pinned_commit() -> Iterator[None]:
    """``meta.git_commit()`` が ``PINNED_COMMIT`` を返す間だけ処理を実行する。

    各パイプラインは ``run_and_report_*`` の**関数本体で**
    ``from rc_basics_lab.meta import git_commit`` している (D-53 の遅延 import と
    同じ形) ため、束縛は呼び出しのたびに ``meta`` から解決される。したがって
    ``meta`` 側1箇所の差し替えで全パイプラインに効き、パイプラインが増えても
    ここは変わらない。
    """
    original = meta.git_commit
    meta.git_commit = lambda: PINNED_COMMIT
    try:
        yield
    finally:
        meta.git_commit = original


CASES: tuple[GoldenCase, ...] = (
    GoldenCase("01_what_is_rc", "01_what_is_rc.yaml", _run_01),
    GoldenCase("02_esp_and_dynamics", "02_esp_and_dynamics.yaml", _run_02),
    GoldenCase("03_capacity", "03_capacity.yaml", _run_03),
    GoldenCase("04_chaotic_freerun", "04_chaotic_freerun.yaml", _run_04),
)
"""照合するケース。実験を 1 本足したらここへ 1 行足す。"""


def _is_excluded_json_key(key: str) -> bool:
    """``meta.json`` の照合から外すキーか。"""
    return (
        key in VOLATILE_JSON_KEYS
        or key in ENVIRONMENT_JSON_KEYS
        or key.startswith(WALL_TIME_PREFIX)
    )


def _strip_json(value: object) -> object:
    """再帰的に除外キーを落とす (``config`` の内側にも ``wall_time`` は無いが、
    将来足されたときに黙って照合が緩むのを防ぐため全階層で落とす)。"""
    if isinstance(value, dict):
        return {
            str(key): _strip_json(item)
            for key, item in value.items()
            if not _is_excluded_json_key(str(key))
        }
    if isinstance(value, list):
        return [_strip_json(item) for item in value]
    return value


def normalize_json(path: Path) -> bytes:
    """``meta.json`` から除外キーを落とした正規形。"""
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(_strip_json(loaded), ensure_ascii=False, indent=2, sort_keys=True)
    return text.encode("utf-8")


def normalize_csv(path: Path) -> bytes:
    """CSV から ``wall_time*`` 列を落とした正規形。

    値そのものは文字列のまま扱う (float へ通すと丸めが入り、桁の変化を
    見逃す)。
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return b""
    keep = [
        index
        for index, name in enumerate(rows[0])
        if not name.startswith(WALL_TIME_PREFIX)
    ]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow([row[index] for index in keep])
    return buffer.getvalue().encode("utf-8")


def digest(path: Path) -> str:
    """成果物のダイジェスト (拡張子で正規化を選ぶ)。"""
    if path.suffix == ".csv":
        payload = normalize_csv(path)
    elif path.suffix == ".json":
        payload = normalize_json(path)
    else:
        payload = path.read_bytes()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def run_case(case: GoldenCase, out_dir: Path) -> dict[str, str]:
    """ケースを実行し、``相対パス -> ダイジェスト`` を返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with pinned_commit():
        case.run(CONFIG_DIR / case.config, out_dir)
    return {
        path.relative_to(out_dir).as_posix(): digest(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
    }


def environment_fingerprint() -> dict[str, str]:
    """基準値が有効な環境かを判定するための指紋。

    キー名は ``meta.json`` (``meta.collect_meta_for``) と揃える。CJK フォントの
    有無は図のラベル言語を変えるので PNG 側の条件に含まれる。
    """
    return {
        "package_version": __version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cjk_font": setup_style().cjk_font or "",
    }


def load_manifest() -> dict[str, object]:
    """``tests/golden/manifest.json`` を読む。"""
    loaded: object = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"マニフェストが dict ではありません: {MANIFEST_PATH}")
    return {str(key): value for key, value in loaded.items()}


def manifest_fingerprint(manifest: dict[str, object]) -> dict[str, str]:
    """マニフェストが記録している環境指紋。"""
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise TypeError("マニフェストに fingerprint がありません")
    return {str(key): str(value) for key, value in fingerprint.items()}


def manifest_digests(manifest: dict[str, object], case_name: str) -> dict[str, str]:
    """マニフェストが記録しているケースのダイジェスト。"""
    cases = manifest.get("cases")
    if not isinstance(cases, dict):
        raise TypeError("マニフェストに cases がありません")
    entry = cases.get(case_name)
    if not isinstance(entry, dict):
        raise KeyError(f"マニフェストに未登録のケースです: {case_name}")
    return {str(key): str(value) for key, value in entry.items()}


def mismatched_keys(
    expected: dict[str, str], actual: dict[str, str], keys: Sequence[str]
) -> tuple[str, ...]:
    """指紋のうち食い違っている項目。"""
    return tuple(key for key in keys if expected.get(key) != actual.get(key))


def build_manifest() -> dict[str, object]:
    """全ケースを実行して新しいマニフェストを組み立てる。"""
    cases: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="rc-golden-") as tmp:
        for case in CASES:
            digests = run_case(case, Path(tmp) / case.name)
            logger.info("%s: %d 成果物", case.name, len(digests))
            cases[case.name] = digests
    return {"fingerprint": environment_fingerprint(), "cases": cases}


def write_manifest(manifest: dict[str, object]) -> Path:
    """マニフェストを書き出す。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MANIFEST_PATH


def main(argv: Sequence[str] | None = None) -> int:
    """``--update`` で基準値を取り直す。"""
    parser = argparse.ArgumentParser(description="ゴールデンの基準値を更新する")
    parser.add_argument(
        "--update", action="store_true", help="manifest.json を書き直す"
    )
    namespace = parser.parse_args(argv)
    if not namespace.update:
        parser.error("--update が必要です (照合は make golden / pytest で行う)")
    path = write_manifest(build_manifest())
    logger.info("基準値を更新しました: %s", path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
