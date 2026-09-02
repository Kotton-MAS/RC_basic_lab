"""実験 3-T (対照の梯子) の設定 dataclass (D-138 / D-139).

``capacity03.py`` が上限 (非空 300 行) に達したので分けた。役割としても、
梯子は ``make ladder-03`` として本番の ``figures-03`` の外で回る補助実験で
あり、``Capacity03Config`` の他のセクションとは独立している。

**水準の密度をここに書いても効かない** (D-139)。梯子は実行時に BA の
``2m/N`` へ全水準をそろえ直す —— N を掃引する以上、固定値では N=50 でしか
そろわないためである。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.reservoir.topology import (
    BarabasiAlbertConfig,
    DegreePreservingConfig,
    ErdosRenyiConfig,
    TopologyConfig,
    TopologyControlConfig,
)

_LADDER_BA_M = 2
"""梯子の既定の BA の枝数。密度をここから決める。"""


def _ladder_levels() -> tuple[TopologyConfig, ...]:
    """梯子の既定の水準 (D-138)。

    **ここに書く密度は効かない** (D-139)。密度は実行時に BA 水準の ``2m/N`` へ
    そろえ直される —— N を掃引する以上、YAML に書いた固定値では N=50 でしか
    そろわないためである (実測: N=25 で BA が ER の2倍、N=100 で半分)。
    """
    base = ErdosRenyiConfig()
    return (
        base,
        TopologyControlConfig(base=base, symmetrize=True),
        TopologyControlConfig(base=base, drop_self_loops=True),
        DegreePreservingConfig(base=BarabasiAlbertConfig(m=_LADDER_BA_M)),
        BarabasiAlbertConfig(m=_LADDER_BA_M),
    )


@dataclass(frozen=True, slots=True)
class LadderSweepConfig:
    """梯子を掛ける掃引軸を1本 (D-139)。

    ``axis`` は ``TopologyLadderConfig`` のフィールド名である
    (``reservoir.axes`` の記法をそのまま使う)。**実験計画を決める
    ``n_graphs`` / ``n_replicates`` は振れない** —— そこを振ると水準ごとに
    対の数が変わり、対応のある検定が組めなくなる。

    Attributes:
        axis: 振るフィールド名 (``n_units`` / ``state_noise`` など)。
        values: 振る値。空なら**この掃引は回らない**。
    """

    axis: str = "n_units"
    values: tuple[float, ...] = ()


def _ladder_sweeps() -> tuple[LadderSweepConfig, ...]:
    """梯子の既定の掃引 (D-139)。

    - ``n_units``: トポロジの順位は N に依存するか。**物理リザバーは小さい N
      に住む**ので、25〜100 で順位が入れ替わるなら先行より重要な結果になる
    - ``state_noise``: ノイズなしで選んだ最適トポロジは、ノイズ下でも最適か。
      トポロジ最適化の文献はほぼすべてノイズなしで行われている
    - ``rho``: 先行 (Nonlinear Dynamics 2025) の主張は「**広い rho の範囲で**
      BA が優れる」「rho > 1 でも性能を保つ」である。1点で測っている限り
      反証も追認もできないので、rho > 1 を含めて振る (D-140)
    - ``n_steps``: **ハブ型は飽和に必要な T が長いかもしれない** (D-142)。
      もしそうなら、同じ T で ER と BA を比べた文献は BA を過小評価して
      いることになる —— そして**この梯子の結論も同じ疑いを受ける**ので、
      トポロジの主張と同じ成果物の中で確かめる

    両方を**同じ CSV** に入れる (``sweep_axis`` 列で区別する)。基準点
    (N=50 / ノイズ 0) が2つの掃引に重複して入るが、そこは**同じ数が出るはず
    の点**であり、掃引点の間で状態が漏れていないことの検査になる。
    """
    return (
        LadderSweepConfig(axis="n_units", values=(25.0, 50.0, 100.0)),
        LadderSweepConfig(axis="state_noise", values=(0.0, 0.01, 0.1)),
        LadderSweepConfig(axis="rho", values=(0.5, 0.8, 0.95, 1.1, 1.3)),
        LadderSweepConfig(axis="n_steps", values=(5_000.0, 20_000.0, 80_000.0)),
    )


@dataclass(frozen=True, slots=True)
class TopologyLadderConfig:
    """実験 3-T: **交絡を1つずつ剥がす対照の梯子** (D-138)。

    「スケールフリーは記憶容量に効くか」は既に主張がある問いなので、BA を足して
    測るだけでは追試にしかならない。同じ密度で ER と BA を比べると次数分布・
    相互結合率・自己ループが同時に動くので (D-131)、**1つだけ動かした水準**を
    並べて何が効いているのかを分ける。

    水準は ``levels`` に並べた順で回る。既定の5水準:

    ===================  ==============================================
    水準                 何を変えるか
    ===================  ==============================================
    ``erdos_renyi``      基準
    ``control`` (対称)   相互結合だけを入れる
    ``control`` (対角)   自己ループだけを抜く
    ``degree_preserving``  BA の次数列だけを残す (**本命の帰無モデル**)
    ``barabasi_albert``  全部
    ===================  ==============================================

    **グラフと重みを入れ子にする** (``n_graphs`` x ``n_replicates``)。トポロジ
    比較には2種類の分散があり (グラフの実現値 / 重みの実現値)、片方だけを振ると
    「BA が良い」がグラフ間分散に埋もれているかを判定できない。実測 (N=50,
    T=20000) ではグラフ間 s.d. 0.69〜0.83、重み間 s.d. 0.71〜0.94 と**同程度**
    だった —— どちらか一方では足りない。

    ``make figures-03`` の予算の外で手動実行する (``symmetry_sweep`` と同じ扱い)。

    Attributes:
        levels: 回すトポロジ。空なら梯子を回さない。
        rho: スペクトル半径 (全水準で同じ)。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        n_units: リザバーのユニット数 N。
        n_steps: 系列長 [ステップ]。
        state_noise: 状態ノイズの標準偏差 (0 ならノイズなし)。
        n_graphs: グラフの実現値の本数 (topology ストリーム)。
        n_replicates: 1グラフあたりの重みの実現値の本数 (reservoir ストリーム)。
        sweeps: 梯子を掛ける掃引軸 (D-139)。空なら基準の1点だけを回す。
    """

    levels: tuple[TopologyConfig, ...] = field(default_factory=_ladder_levels)
    rho: float = 0.95
    leak_rate: float = 1.0
    sigma_u: float = 0.2
    state_noise: float = 0.0
    n_units: int = 50
    n_steps: int = 20_000
    n_graphs: int = 8
    n_replicates: int = 3
    sweeps: tuple[LadderSweepConfig, ...] = field(default_factory=_ladder_sweeps)


@dataclass(frozen=True, slots=True)
class LadderThresholdConfig:
    """実験 3-Th: **閾値の選び方が梯子の結論を作っていないか** (D-143)。

    容量はサロゲートのしきい値で切ってから足す (D-27)。既定の
    ``n_surrogates=100`` / ``surrogate_quantile=0.99`` は**判定基準そのもの**
    なので、値によって水準の順位が動くなら、梯子が語っているのは現象ではなく
    閾値の選び方である。02 の ``abs_tol`` 感度掃引 (``experiment/threshold.py``)
    と同じ形を容量側でも置く。

    **軌道は条件ごとに1回しか作らない。** しきい値は診断の中で容量を切る段
    だけを変えるので、状態を1回作って判定だけをやり直せる。

    ``chi2`` を格子に入れない。MC は ``chi2`` を持たない (次数1しか測らない
    ため) ので、入れると MC と IPC で違う基準を並べることになる —— 順位が
    動いたのが閾値のせいか診断の違いかを分けられなくなる。モードの比較自体は
    ``capacity_threshold.py`` が代表条件1つで既にやっている。

    実験計画を梯子より小さくしてある。問いが「順位が動くか」であって効果量の
    推定ではないためで、梯子の実測 (BA - ER = -1.92、対ごとの s.d. 0.8) なら
    8対で符号検定 p = 0.004 に届く。

    Attributes:
        n_surrogates_grid: サロゲート本数の格子 (既定 100 を中心に上下4倍)。
        quantile_grid: 分位点の格子 (既定 0.99 を中心に)。
        include_no_threshold: しきい値なし (生の容量) も並べるか。
        n_graphs: グラフの実現値の本数。
        n_replicates: 1グラフあたりの重みの実現値の本数。
    """

    n_surrogates_grid: tuple[int, ...] = (25, 100, 400)
    quantile_grid: tuple[float, ...] = (0.95, 0.99, 0.999)
    include_no_threshold: bool = True
    n_graphs: int = 4
    n_replicates: int = 2


__all__ = ["LadderSweepConfig", "LadderThresholdConfig", "TopologyLadderConfig"]
