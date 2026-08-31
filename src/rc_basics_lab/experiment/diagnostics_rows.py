"""診断のスカラを長形式で書き出す層 (§1 の「診断結果の出口」).

**主表 (``capacity.csv`` など) の列を増やさずに、新しい診断の値を成果物へ
出すための場所**である。

診断を1本足すコストは、これまで診断本体ではなく**出口側**に集中していた。
スカラを1個 CSV に出すのに、行 dataclass にフィールドを足し、``CSV_COLUMNS``
に列を足し、``docs/design.md`` の表を直し、成果物を再生成して指紋とゴールデンを
取り直す必要があった。``CapacityRow`` は 39 列あり、**次数分布の指数を1つ測る
たびにこの表が 40 列になる**。列が増えれば既存の指紋も golden も動くので、
「バイト不変が合否判定」という一番強い規律のコストが診断のたびに払われていた。

長形式なら**既存 CSV の列が1つも動かない**。``DiagnosticResult.scalars`` は
``Mapping[str, float]`` なので、変換なしでそのまま流せる。

``capacity_profile.csv`` (D-38) が同じ形をしており、こちらはその一般化である。
違いは、あちらが (次数 x 遅延) のセルを持つのに対し、こちらは診断が返した
スカラをそのまま1行1個にする点。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rc_basics_lab.diagnostics.base import DiagnosticResult
from rc_basics_lab.experiment.report import write_rows_csv

DIAGNOSTICS_CSV = "diagnostics.csv"
"""長形式の診断スカラを書くファイル名 (全実験で共通)。"""

DIAGNOSTICS_CSV_COLUMNS = (
    "experiment",
    "condition_id",
    "replicate",
    "diagnostic",
    "key",
    "value",
)
"""列順。**この6列は増やさない** —— 増やせる形にした瞬間に、主表と同じ問題が
ここで再発する。診断固有の情報は ``condition_id`` と ``key`` の中に入れる。"""

CONDITION_SEPARATOR = "|"
"""``condition_id`` の軸どうしの区切り。"""

CONDITION_ASSIGN = "="
"""``condition_id`` の軸名と値の区切り。"""


@dataclass(frozen=True, slots=True)
class DiagnosticScalarRow:
    """``diagnostics.csv`` の1行 (長形式)。宣言順が CSV の列順。

    Attributes:
        experiment: 実験ラベル (主表の ``experiment`` 列と同じ値)。
        condition_id: 掃引で振った軸だけを並べたキー (``condition_key`` 参照)。
        replicate: レプリケート番号 (0 始まり)。
        diagnostic: ``DiagnosticResult.name``。
        key: ``scalars`` のキー。
        value: その値。
    """

    experiment: str
    condition_id: str
    replicate: int
    diagnostic: str
    key: str
    value: float


def condition_key(axes: Mapping[str, float | int | str]) -> str:
    """掃引の軸から ``condition_id`` を作る (**形式の単一の真実**、D-118)。

    ``rho=0.9|leak_rate=1.0`` のように、**軸名の昇順**で ``名前=値`` を並べる。

    昇順に固定するのは、呼び出し側が dict を作る順に依存させないためである。
    順が変わると同じ条件が別のキーになり、実験をまたいだ突き合わせが静かに
    壊れる (**この形式を後から変えると全実験の CSV が動く**ので、
    ``decisions.yaml`` に記録してある)。

    浮動小数は ``repr`` ではなく ``%g`` 相当で書く。``0.30000000000000004``
    のような表現差でキーが割れないようにするためで、**桁を落とすことによる
    衝突は呼び出し側の責任**である (同じ掃引の中で ``%g`` が衝突する軸を
    振ることは想定しない)。

    Args:
        axes: 軸名 -> 値。振っていない軸は入れないこと。

    Returns:
        ``condition_id`` の文字列。``axes`` が空なら空文字列。
    """
    parts = []
    for name in sorted(axes):
        value = axes[name]
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{name}{CONDITION_ASSIGN}{text}")
    return CONDITION_SEPARATOR.join(parts)


def scalar_rows(
    results: Iterable[DiagnosticResult],
    *,
    experiment: str,
    condition_id: str,
    replicate: int,
) -> tuple[DiagnosticScalarRow, ...]:
    """1条件ぶんの診断結果を長形式の行に展開する。

    ``scalars`` のキー順は診断側の宣言順のままにする (昇順に並べ替えない)
    —— 診断が「総量 -> 内訳」の順で返しているとき、その順序は読み手にとって
    情報だからである。

    Args:
        results: その条件で走った診断の結果。
        experiment: 実験ラベル。
        condition_id: ``condition_key`` の出力。
        replicate: レプリケート番号。

    Returns:
        行の並び (診断ごと・キーごと)。
    """
    return tuple(
        DiagnosticScalarRow(
            experiment=experiment,
            condition_id=condition_id,
            replicate=replicate,
            diagnostic=result.name,
            key=key,
            value=float(value),
        )
        for result in results
        for key, value in result.scalars.items()
    )


def rows_of(outcomes: Iterable[object]) -> tuple[DiagnosticScalarRow, ...]:
    """``diagnostics`` 属性を持つ結果の並びから、長形式の行を畳む。

    実験ごとの結果型 (``CapacityOutcome`` など) を知らずに済ませるため、
    属性の有無で受ける。**型で縛らないのは意図的**である —— 実験を1本足す
    たびにこの層が結果型を import すると、``experiment`` の依存が実験の数だけ
    増える (診断の出口は実験に依存しない層であってほしい)。

    Args:
        outcomes: ``diagnostics: Sequence[DiagnosticScalarRow]`` を持つ結果。

    Returns:
        畳んだ行。

    Raises:
        AttributeError: ``diagnostics`` を持たない要素が混ざっている場合
            (**黙って空を返さない** —— 出るはずの行が消えたことに気づけない)。
    """
    collected: list[DiagnosticScalarRow] = []
    for outcome in outcomes:
        rows = outcome.diagnostics  # type: ignore[attr-defined]
        collected.extend(rows)
    return tuple(collected)


def write_diagnostics_csv(rows: Sequence[DiagnosticScalarRow], out_dir: Path) -> Path:
    """``diagnostics.csv`` を書く (全実験で同じ関数を通す)。

    Args:
        rows: 長形式の行。
        out_dir: 実験の出力ディレクトリ。

    Returns:
        書き出したパス。
    """
    return write_rows_csv(rows, out_dir / DIAGNOSTICS_CSV, DIAGNOSTICS_CSV_COLUMNS)


__all__ = [
    "DIAGNOSTICS_CSV",
    "DIAGNOSTICS_CSV_COLUMNS",
    "DiagnosticScalarRow",
    "condition_key",
    "rows_of",
    "scalar_rows",
    "write_diagnostics_csv",
]
