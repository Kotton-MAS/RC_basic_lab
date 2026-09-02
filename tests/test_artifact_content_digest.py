"""内容指紋 (``content_sha256``) の検査.

**「時間だけの差」と「実質的な差」を分けられること**を測る。分けられないと
再生成のたびに全行が動き、レビューで本物の変化が埋もれる (実測: 23 枚の CSV の
うち 15 枚、JSON 5 枚が実行時間を含む)。

ここが空虚になる形は「内容指紋がバイト指紋と同じものを返す」である。
それでも「一致する / しない」の検査は緑のままなので、**時間列を変えても
内容指紋が動かないこと**を正面から測る。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from _artifact_manifest import (
    TIMING_COLUMN_PREFIX,
    VOLATILE_JSON_KEYS,
    VOLATILE_JSON_SUBTREES,
    content_digest,
    volatile_json_paths,
)

_CSV_HEADER = "task,method,nrmse,wall_time_s,wall_time_mc_s"
_CSV_ROWS = ("mackey_glass,esn,0.5,1.25,0.5", "delay_parity,esn,0.7,2.5,1.0")


def _csv_bytes(rows: tuple[str, ...] = _CSV_ROWS) -> bytes:
    return ("\n".join((_CSV_HEADER, *rows)) + "\n").encode("utf-8")


def test_changing_only_the_timing_columns_keeps_the_content_digest(
    tmp_path: Path,
) -> None:
    """時間列だけを変えても内容指紋は動かない。"""
    original = tmp_path / "a.csv"
    original.write_bytes(_csv_bytes())
    retimed = tmp_path / "b.csv"
    retimed.write_bytes(
        _csv_bytes(("mackey_glass,esn,0.5,9.99,8.88", "delay_parity,esn,0.7,7.77,6.66"))
    )
    assert content_digest(original, original.read_bytes()) == content_digest(
        retimed, retimed.read_bytes()
    )
    assert original.read_bytes() != retimed.read_bytes(), "バイトは違うはずです"


def test_changing_a_measured_value_moves_the_content_digest(tmp_path: Path) -> None:
    """数値を1つ変えたら内容指紋が動く (**空虚でないことの確認**)。"""
    original = tmp_path / "a.csv"
    original.write_bytes(_csv_bytes())
    changed = tmp_path / "b.csv"
    changed.write_bytes(
        _csv_bytes(("mackey_glass,esn,0.5001,1.25,0.5", "delay_parity,esn,0.7,2.5,1.0"))
    )
    assert content_digest(original, original.read_bytes()) != content_digest(
        changed, changed.read_bytes()
    )


def test_a_csv_without_timing_columns_hashes_its_bytes(tmp_path: Path) -> None:
    """時間列が無い CSV は正規化を通さない (無駄な書き換えをしない)。"""
    path = tmp_path / "plain.csv"
    payload = b"task,method,nrmse\nmackey_glass,esn,0.5\n"
    path.write_bytes(payload)
    assert content_digest(path, payload) == hashlib.sha256(payload).hexdigest()


def test_the_volatile_keys_are_dropped_from_meta_json(tmp_path: Path) -> None:
    """``meta.json`` の実行時間と時刻が内容指紋に入らない。"""
    path = tmp_path / "meta.json"
    base = {"commit": "abc1234", "config": {"n_units": 200}}
    first = {**base, "timestamp_utc": "2026-01-01T00:00:00+00:00", "wall_time_s": 1.0}
    second = {**base, "timestamp_utc": "2026-12-31T23:59:59+00:00", "wall_time_s": 9.0}
    path.write_text(json.dumps(first), encoding="utf-8")
    digest_first = content_digest(path, path.read_bytes())
    path.write_text(json.dumps(second), encoding="utf-8")
    assert digest_first == content_digest(path, path.read_bytes())


def test_the_commit_is_excluded_from_the_content(tmp_path: Path) -> None:
    """``commit`` も内容指紋から除く。

    当初は「どのコミットで作られたかは内容の一部」として残したが、実測すると
    再生成のたびに ``meta.json`` が「内容が変わった」側に出てしまい、
    **分類の意味が無くなった** (commit は必ず動くため)。

    除いても弱くならない。commit がそろっていることは
    ``test_cycle_hygiene`` が ``meta.json`` を直接読んで見ている。
    """
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"commit": "aaa", "config": {}}), encoding="utf-8")
    first = content_digest(path, path.read_bytes())
    path.write_text(json.dumps({"commit": "bbb", "config": {}}), encoding="utf-8")
    assert first == content_digest(path, path.read_bytes())
    assert "commit" in VOLATILE_JSON_KEYS


def test_a_png_is_hashed_as_bytes(tmp_path: Path) -> None:
    """PNG は正規化しない (画素から commit だけを抜けないため)。"""
    path = tmp_path / "fig.png"
    payload = b"\x89PNG\r\n\x1a\n" + b"pixels"
    path.write_bytes(payload)
    assert content_digest(path, payload) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "path_name", ["results/03_capacity/capacity.csv", "results/comparison.csv"]
)
def test_the_real_artifacts_have_their_timing_columns_dropped(path_name: str) -> None:
    """本番の成果物で、実際に時間列が落ちていること。

    合成データだけで測ると「本番の列名が ``wall_time`` で始まらなくなった」
    ときに空振りする。
    """
    path = Path(path_name)
    payload = path.read_bytes()
    header = next(csv.reader(io.StringIO(payload.decode("utf-8"))))
    timing = [name for name in header if name.startswith(TIMING_COLUMN_PREFIX)]
    assert timing, f"{path_name} に時間列がありません (この検査が空振りします)"
    assert content_digest(path, payload) != hashlib.sha256(payload).hexdigest()


# --- 入れ子の実測値 (D-141) ---------------------------------------------

VOLATILE_PATHS_BY_ARTIFACT = {
    "results/meta.json": ("commit", "timestamp_utc", "wall_time_s"),
    "results/02_esp_and_dynamics/meta.json": (
        "commit",
        "timestamp_utc",
        "wall_time_s",
    ),
    "results/03_capacity/meta.json": (
        "commit",
        "threshold_comparison.wall_time_s",
        "timestamp_utc",
        "wall_time_breakdown",
        "wall_time_s",
    ),
    "results/04_chaotic_freerun/meta.json": (
        "commit",
        "timestamp_utc",
        "wall_time_breakdown",
        "wall_time_s",
    ),
    "results/05_anomaly_detection/meta.json": (
        "commit",
        "timestamp_utc",
        "wall_time_breakdown",
        "wall_time_s",
    ),
}
"""本番の ``meta.json`` で内容指紋から落ちるパスの**全件**。

落とす規則を広げるのは簡単で、広げすぎると「設定を変えたのに内容指紋が
動かない」状態を静かに作れる。**落ちすぎを捕まえる唯一の機械**なので、
新しいキーを足したらここも足すこと (足さなければ赤くなる)。
"""


@pytest.mark.parametrize("path_name", sorted(VOLATILE_PATHS_BY_ARTIFACT))
def test_the_stripped_paths_are_exactly_the_declared_ones(path_name: str) -> None:
    """本番の ``meta.json`` から落ちるパスが宣言と**一致する** (D-141)。"""
    found = volatile_json_paths(Path(path_name).read_bytes())
    assert found == VOLATILE_PATHS_BY_ARTIFACT[path_name], (
        f"{path_name} で落ちるパスが変わりました:\n"
        f"  余剰={sorted(set(found) - set(VOLATILE_PATHS_BY_ARTIFACT[path_name]))}\n"
        f"  不足={sorted(set(VOLATILE_PATHS_BY_ARTIFACT[path_name]) - set(found))}"
    )


def test_a_nested_timing_value_does_not_move_the_content_digest(tmp_path: Path) -> None:
    """入れ子の実測値だけを変えても内容指紋は動かない (D-141)。

    これを入れる前は 04 / 05 の ``meta.json`` が再生成のたびに「内容が
    変わった」側に出ていた。実行時間しか動いていないのに毎回説明を要する
    行が2件出ると、**本当に説明を要する行を見落とす**。
    """
    path = tmp_path / "meta.json"

    def payload(capacity: float, threshold: float) -> str:
        return json.dumps(
            {
                "config": {"n_units": 200},
                "threshold_comparison": {"wall_time_s": threshold, "n_lags": 40},
                "wall_time_breakdown": {"capacity_s": capacity, "figures_s": 1.0},
            }
        )

    path.write_text(payload(82.9, 3.9), encoding="utf-8")
    first = content_digest(path, path.read_bytes())
    path.write_text(payload(83.4, 4.1), encoding="utf-8")
    assert first == content_digest(path, path.read_bytes())


def test_a_budget_is_not_mistaken_for_a_measurement(tmp_path: Path) -> None:
    """``*_s`` で終わる**設定値**は落とさない (D-141)。

    05 は ``total_budget_s`` と ``wall_time_budget_s.*`` を持つ。名前の規則
    (``*_s`` で終われば時間) で落とす案を採ると、**予算を変えたことが内容
    指紋に出なくなる**。測った値と設定した値が同じ接尾辞を共有している以上、
    名前ではなく**どの節にあるか**で決めるしかない。
    """
    path = tmp_path / "meta.json"

    def payload(budget: float) -> str:
        return json.dumps(
            {
                "total_budget_s": budget,
                "wall_time_budget_s": {"figures_s": 20.0, "headline_s": 460.0},
                "wall_time_breakdown": {"figures_s": 1.35},
            }
        )

    path.write_text(payload(900.0), encoding="utf-8")
    first = content_digest(path, path.read_bytes())
    path.write_text(payload(1200.0), encoding="utf-8")
    assert first != content_digest(path, path.read_bytes()), (
        "予算を変えたのに内容指紋が動きません"
    )
    assert "wall_time_budget_s" not in VOLATILE_JSON_SUBTREES


def test_a_nested_configuration_value_still_moves_the_content_digest(
    tmp_path: Path,
) -> None:
    """入れ子でも設定値なら内容指紋が動く (**空虚でないことの確認**)。"""
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps({"config": {"reservoir": {"n_units": 200}}}), encoding="utf-8"
    )
    first = content_digest(path, path.read_bytes())
    path.write_text(
        json.dumps({"config": {"reservoir": {"n_units": 201}}}), encoding="utf-8"
    )
    assert first != content_digest(path, path.read_bytes())
