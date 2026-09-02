"""実験 2-A / 2-B / 2-C の配線層 —— ESN と診断層をつなぐ唯一の場所.

``diagnostics/`` は行列を受け取って ``DiagnosticResult`` を返すだけで、どの行列を
渡すかは実験層が決める (01 の ``experiment/state_space.py`` と同じ分業)。
このモジュールは ``reservoir`` と ``diagnostics`` の**両方**を import してよい
唯一の場所であり、D-12 (``diagnostics`` は ``reservoir`` を知らない) はここで
アダプタを持つことで成立している。

3つの実験は「1条件 = (rho, leak_rate, sigma_u, replicate)」という同じ単位に
分解でき、``evaluate_condition`` がその1条件を回す。掃引の違いはどの軸を振るか
だけである。

**入力強度は駆動信号の標準偏差 sigma_u で定義する** (D-17)。

**状態行列の計算と、それを使う診断・回帰は別レイヤー** (F-1-005 / F-1-010)。
``simulate_reference_trajectory`` は ``ReservoirSweepConfig`` + ``DriveConfig``
+ 基底シードだけで状態行列 ``X`` を1回計算して返す。02 (ESP) はこれに比較
軌道 n_pairs 本を足す薄い層 (``simulate_condition``) を経由するが、03
(MC/IPC) のように「同じ ``X`` に対して遅延・次数ごとに読み出し回帰だけを
繰り返す」設計は ``simulate_reference_trajectory`` を1条件につき1回呼び、
返ってきた ``states`` を使い回すことで、``ESN.run`` (T に線形) の再実行を
避けられる。02 の ``_sweep`` は1条件=1回の ``evaluate_condition``呼び出しで
十分なため (本番 336 条件・実測 53.65秒)、逐次ループのままで良いが、03 の
想定規模 (delay x degree で 2395 通り) にこのパターンをそのまま複製すると
``ESN.run`` の再実行コストが支配的になる (実測: N=200, T=1e6 で約4.7秒/回。
2395 回なら約3.1時間)。
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.config import (
    DriveConfig,
    Esp02Config,
    ReservoirSweepConfig,
    esp_stream_seed,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, StatePropagator
from rc_basics_lab.diagnostics.esp import conditional_lyapunov, esp_convergence
from rc_basics_lab.diagnostics.timescale import autocorrelation_time
from rc_basics_lab.experiment.diagnostics_rows import DiagnosticScalarRow
from rc_basics_lab.experiment.esp_reservoir import BIAS_SCALE, build_esn_config
from rc_basics_lab.experiment.esp_rows import (
    ESP_CSV_COLUMNS,
    EspRow,
    esp_diagnostic_rows,
)
from rc_basics_lab.reservoir.protocol import Reservoir
from rc_basics_lab.reservoir.registry import build_reservoir
from rc_basics_lab.reservoir.topology import TopologyConfig
from rc_basics_lab.seeds import SeedStream, make_rng_for
from rc_basics_lab.types import FloatArray

logger = logging.getLogger(__name__)

UNIFORM = "uniform"
"""``DriveConfig.distribution`` が受理する唯一の値 (仕様 §8 の前提)。"""

EXPERIMENT_DECAY = "2A_decay"
EXPERIMENT_TIMESCALE = "2B_timescale"
EXPERIMENT_ESP_MAP = "2C_esp_map"


ESP_DISTANCE_WASHOUT = 0
"""``esp_convergence`` に渡す ctx の washout。**0 に固定する** (実装メモ)。

``esp_convergence`` で washout が効くのは減衰率の当てはめ開始位置だけであり、
判定 (末尾 window の中央値) には効かない。一方 2-A は**過渡そのものを見せる
図**であって、距離が丸めの床 (``EspConfig.floor``) に届く前に当てはめを始め
ないと減衰率が測れない。実測: 無入力 rho=0.5 の距離は t≈46 で 1e-14 を割る
ため、当てはめ開始が ``washout(200) + fit_skip(50) = 250`` だと当てはめ点が
0 点になり ``decay_rate_per_step`` が ``nan`` になる (rho=0.8 も同様)。
過渡を捨てるのは λ と自己相関の側の要求なので、そちらの ctx にだけ
``drive.washout`` を渡す。当てはめ開始の微調整は ``esp.fit_skip`` が担う。
"""

BOUNDARY_LAMBDA = 0.01
"""λ の符号と ESP 判定の整合を問わない境界帯の幅 (仕様 T3 受け入れ基準)。"""

STRONG_DRIVE_SIGMA = 0.5
"""この強度以上の駆動では λ の符号と ESP 判定が完全に一致する (D-20 予定)。

実測 (本番格子 369 行のうち境界近傍を除く 332 件): ``sigma_u >= 0.5`` の
158 条件で不一致 0 件。不一致 27 件はすべて ``sigma_u <= 0.2`` かつ
``rho in [1.1, 1.6]`` に限局し、向きも「λ<0 なのに非収束」の一方向だけである。
"""

_N_INPUTS = 1
"""駆動入力の次元 (i.i.d. 一様乱数の1系列)。"""

_SQRT3 = math.sqrt(3.0)
"""一様分布 ``U[-a, a]`` の標準偏差 ``a / sqrt(3)`` からの換算係数 (D-17)。"""


def make_drive(
    sigma: float,
    n_steps: int,
    rng: np.random.Generator,
    *,
    distribution: str = UNIFORM,
    offset: float = 0.0,
) -> FloatArray:
    """i.i.d. 駆動入力 ``(n_steps, 1)`` を作る。**強度は標準偏差** (D-17)。

    Args:
        sigma: 駆動信号の標準偏差 sigma_u。``0.0`` は厳密なゼロ系列。
        n_steps: 系列長 [ステップ]。
        rng: 乱数生成器 (``SeedStream.TASK`` から引く)。
        distribution: 分布名。``"uniform"`` 以外は ``ValueError``
            (黙って一様として扱わない。設定値が効かない実験を作らないため)。
        offset: 駆動信号に加える定数 (**平均のずれ**、D-116)。標準偏差も分布の
            形 (一様) も変えないので Legendre 基底は正規直交のまま (D-28)。

    Returns:
        駆動入力 ``(n_steps, 1)``。一様分布なので振幅は ``sqrt(3) * sigma``。

    Raises:
        ValueError: ``distribution`` が未対応、``sigma`` が負、
            ``n_steps`` が 1 未満の場合。
    """
    if distribution != UNIFORM:
        raise ValueError(
            f"未対応の駆動信号の分布です (現在は {UNIFORM!r} のみ): {distribution!r}"
        )
    if sigma < 0.0:
        raise ValueError(f"sigma_u は 0 以上である必要があります: {sigma!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps は 1 以上である必要があります: {n_steps!r}")
    if sigma == 0.0:
        zeros: FloatArray = np.full((n_steps, _N_INPUTS), offset, dtype=np.float64)
        return zeros
    amplitude = _SQRT3 * sigma
    drive: FloatArray = rng.uniform(-amplitude, amplitude, (n_steps, _N_INPUTS))
    # offset=0.0 のときは加算しない。`+ 0.0` でも値は同じだが、既存の成果物と
    # のバイト一致を「浮動小数の加算が恒等である」という性質に依存させない。
    if offset != 0.0:
        return drive + offset
    return drive


def make_initial_states(
    n_units: int, n_pairs: int, rng: np.random.Generator
) -> tuple[FloatArray, ...]:
    """``U[-1, 1]^N`` から ``n_pairs + 1`` 本の初期状態を独立に引く (D-16)。

    先頭が参照軌道の初期状態、残りが比較軌道の初期状態。**片方をゼロ状態に
    しない**。無入力ではゼロが不動点なので、片方を 0 にすると「2軌道の分離」
    ではなく「単一軌道の原点への収束」を測ることになり ESP と別の量に化ける。

    Raises:
        ValueError: ``n_units`` が 1 未満、または ``n_pairs`` が 1 未満の場合。
    """
    if n_units < 1:
        raise ValueError(f"n_units は 1 以上である必要があります: {n_units!r}")
    if n_pairs < 1:
        raise ValueError(f"n_pairs は 1 以上である必要があります: {n_pairs!r}")
    return tuple(
        np.asarray(rng.uniform(-1.0, 1.0, n_units), dtype=np.float64)
        for _ in range(n_pairs + 1)
    )


def require_deterministic_esn(
    state_noise: float,
    *,
    what: str,
    why: str,
    forbidden: str,
    remedy: str,
) -> None:
    """``state_noise`` が 0 でなければ ``ValueError`` にする (D-47 / D-48 共有)。

    ノイズを 02 経路へ入れたときの壊れ方は2種類あり、直し方も別々である
    (伝播器: ADR 0001 §2 / 比較軌道: 同 §3)。判定と**4点そろったメッセージの
    組み立て**をここ1本に集約する。
    Args:
        state_noise: 検査する値。呼び出し側の引数でも ``esn.config`` 由来でもよい。
        what: 何を拒否したか。
        why: なぜか (測っている量がどう変わるか)。
        forbidden: やってはいけない直し方。
        remedy: 正しい経路。

    Raises:
        ValueError: ``state_noise`` が 0 でない場合。

    Note:
        比較は ``> 0`` ではなく ``!= 0`` である。``ESNConfig`` を経た値は
        ``ESN.__init__`` が負を弾いているので両者は同値だが、``simulate_condition``
        の引数は ESN を通らずに拒否されるため、負の値も「受理しない」側に倒す。
    """
    if state_noise != 0.0:
        raise ValueError(
            f"{what}: {why} / "
            f"やってはいけない直し方: {forbidden} / "
            f"正しい経路: {remedy} "
            f"(実際の state_noise={state_noise!r})"
        )


def esn_propagator(esn: Reservoir, u: FloatArray) -> StatePropagator:
    """``X[t]`` から ``X[t+1]`` を返す伝播器を作る (D-18 / D-48)。

    ``ESN.run`` の規約により ``X[t]`` は ``u[t]`` を**処理した後**の状態なので、
    1ステップ進めるのに使う入力は ``u[t + 1]`` である。``u[t]`` を渡すと λ が
    "それらしい値" で出てレビューでは気づけないため、``conditional_lyapunov``
    は既定でこの一致を実行時に検査する (``check_propagator``)。

    **``state_noise > 0`` の ESN は受理しない** (D-48)。D-36 の「``ESN.run``
    には常に ``rng`` を渡す」は**軌道を作る**呼び出しの規律であり、伝播器は
    そこに含めない —— したがって ``esn.step`` に ``rng`` を渡して
    ``ValueError`` を黙らせるのは誤りである (F-3b1-2-006)。

    判定は**伝播器を作る時点**で行う (D-48)。誤診の実測は
    ``tests/test_experiment_esp.py::test_noise_free_clone_fails_the_propagator_check``
    に残してある。

    Raises:
        ValueError: ``esn`` の ``state_noise`` が 0 でない場合 (D-48)。
    """
    require_deterministic_esn(
        esn.config.state_noise,
        what="state_noise>0 の ESN からは伝播器を作れません (D-48)",
        why=(
            "conditional_lyapunov は同じ写像を2本の近接した状態に当てて摂動の"
            "成長率を測るので伝播器は決定的でなければならず、ノイズを入れると"
            "測る量が『摂動 + ノイズ実現値の差の成長率』という別物になります"
        ),
        forbidden=(
            "esn.step に rng を渡して黙らせること / "
            "ノイズ無しの複製 ESN で伝播すること "
            "(参照軌道はノイズ有りなので D-18 の check_propagator が "
            "propagator_tol を桁で超えて必ず落ち、しかも誤った診断が出ます)"
        ),
        remedy=(
            "state_noise=0 で ESN を構成して伝播器を作る。ノイズ下の条件付き "
            "Lyapunov 指数が成果物の列として必要になったら、ノイズ実現値を"
            "凍結する案 (ADR 0001 §2.4 の案C) を正面から検討する"
        ),
    )

    def propagate(x: FloatArray, t: int) -> FloatArray:
        # rng を渡さないのは意図的 (上記 docstring / D-48)。ここへ rng を足す
        # 変異は test_propagator_refuses_a_noisy_esn が落とす。
        return esn.step(x, u[t + 1])

    return propagate


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """参照軌道1本と、それを作った ESN・駆動入力 (F-1-005)。

    ``simulate_condition`` (ESP) は比較軌道 n_pairs 本も併せて作るが、03
    (MC/IPC) の読み出し回帰には参照軌道1本があれば足りる。比較軌道を作らない
    ぶん ``simulate_condition`` の ``n_pairs + 1`` 倍の計算 (本番 n_pairs=10 で
    11 倍) を避けられる。

    ``rng`` は重み生成に使った ``Generator`` そのもの (状態ノイズと同じ
    ストリームの続き)。D-36 の『``ESN.run`` には常に ``rng`` を渡す』を
    ``simulate_condition`` の比較軌道ループでも満たすために外へ出す
    (F-3b1-1-001)。``ReferenceTrajectory`` がこれを持たないと、比較軌道側は
    続きのストリームを受け取る手段が無く『常に』が2箇所のうち1箇所でしか
    成立しないリーキーな抽象化になる。
    """

    esn: Reservoir
    drive: FloatArray
    states: FloatArray
    rng: np.random.Generator


def simulate_reference_trajectory(
    reservoir: ReservoirSweepConfig,
    drive_config: DriveConfig,
    *,
    reservoir_seed: int,
    drive_seed: int,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
    x0: FloatArray | None = None,
    state_noise: float = 0.0,
    drive_offset: float = 0.0,
    topology: TopologyConfig | None = None,
) -> ReferenceTrajectory:
    """参照軌道1本を作る (``Esp02Config`` を要らない。F-1-005)。

    ``ReservoirSweepConfig`` + ``DriveConfig`` + 基底シードだけで呼べるので、
    03 (MC/IPC) は ``Esp02Config`` の3ストリーム配線 (D-14) を写経せずに
    再利用できる。比較軌道の初期状態対 (``SeedStream.PROBE``) は ESP 判定専用
    なのでここでは作らない。``x0`` を省略すると ``ESN.run`` の既定 (零ベクトル)。

    Args:
        reservoir: リザバー構造 (掃引軸を除く4フィールド)。
        drive_config: 駆動入力の共通条件。
        reservoir_seed: リザバー重みストリームの基底シード。
        drive_seed: 駆動入力ストリームの基底シード。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        replicate: レプリケート番号 (0 始まり)。
        x0: 初期状態 ``(N,)``。``None`` なら ``ESN.run`` の既定 (零ベクトル)。
        drive_offset: 駆動入力に加える定数 (**平均のずれ**、D-116)。3-S だけが振る。
        state_noise: tanh 内部に加えるガウスノイズの標準偏差 (D-36)。**既定値
            つきキーワード**なので 02 の呼び出しは書き換えない。0 のとき
            ``ESN`` は乱数を1個も引かず、既存の成果物はバイト単位で不変である。

    Returns:
        ESN・駆動入力・状態系列 ``(T, N)``。
    """
    drive_rng = make_rng_for(drive_seed, SeedStream.TASK, replicate)
    u = make_drive(
        sigma_u,
        drive_config.n_steps,
        drive_rng,
        distribution=drive_config.distribution,
        offset=drive_offset,
    )
    reservoir_rng = make_rng_for(reservoir_seed, SeedStream.RESERVOIR, replicate)
    esn = build_reservoir(
        build_esn_config(
            reservoir, rho, leak_rate, state_noise=state_noise, topology=topology
        ),
        reservoir_rng,
        n_inputs=_N_INPUTS,
    )
    # 状態ノイズ用の rng は**常に**渡す (D-36)。
    # state_noise=0 では1個も引かれないので既存の結果は変わらない。``rng`` を
    # ``ReferenceTrajectory`` にも載せて返すのは、``simulate_condition`` の
    # 比較軌道ループが同じ規律に従えるようにするため (F-3b1-1-001)。
    return ReferenceTrajectory(
        esn=esn, drive=u, states=esn.run(u, x0=x0, rng=reservoir_rng), rng=reservoir_rng
    )


@dataclass(frozen=True, slots=True)
class Trajectories:
    """1条件ぶんの軌道と、伝播器を作るのに要る材料。

    ``esp_convergence`` は ``(states, companions)`` の純関数なので、判定基準
    (``EspConfig``) を変えるだけの掃引 (2-C の閾値感度。``experiment/threshold.py``)
    は軌道を作り直す必要が無い。その再利用のために ``evaluate_condition`` から
    シミュレーション部分だけを切り出したのがこの型である。
    """

    esn: Reservoir
    drive: FloatArray
    states: FloatArray
    companions: tuple[FloatArray, ...]


_NOISE_REJECTION_WHY = (
    "(a) 比較軌道が『初期状態もノイズ実現値も違う』軌道になり、ESP 判定が測る"
    "はずの『初期状態だけを振った差』に D-14 の3ストリーム分離の外側から"
    "4本目の未制御な変動が混ざる (b) 各軌道が引く乱数の個数と位置が参照軌道の"
    "消費量に依存するため、結果が**評価順**に依存する"
)
_NOISE_REJECTION_FORBIDDEN = (
    "ノイズ実現値用に"
    "5本目の乱数ストリームを新設すること "
    "(D-14 の3ストリームは 01・02・03 の成果物の再現性の土台で、"
    "本数を増やす判断は 04a の範囲外です)"
)
_NOISE_REJECTION_REMEDY = (
    "simulate_reference_trajectory(..., state_noise=...) を使う "
    "(03 の 3-B' と同じ経路。比較軌道を作らないので上の2つが起きない)"
)


def simulate_condition(
    config: Esp02Config,
    *,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
    state_noise: float = 0.0,
) -> Trajectories:
    """1条件 (rho, leak_rate, sigma_u, replicate) の軌道を作る。

    参照軌道の生成は ``simulate_reference_trajectory`` に委譲する (F-1-005)。
    ここで足すのは ESP 専用の比較軌道 n_pairs 本と、その初期状態対 (D-14 の
    ``PROBE`` ストリーム) だけ。3ストリームが独立なので「初期状態だけを振った
    ときに判定が変わるか」を重みを固定したまま測れる。

    **この経路は ``state_noise > 0`` を受理しない** (D-47)。

    検査は二重に置く (経路非依存):

    1. 引数 ``state_noise``。呼び出し側が明示的に渡した値をその場で拒否する
    2. ``config`` から組んだ ESN の ``state_noise``。``build_esn_config`` に
       ノイズを流す別経路 (monkeypatch / 将来の設定フィールド) が生えても、
       比較軌道ループへ入る前に落ちる

    Args:
        config: 02 の実験設定。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        replicate: レプリケート番号 (0 始まり)。
        state_noise: **0 以外を受理しない** (D-47)。

    Raises:
        ValueError: ``state_noise`` が 0 でない場合、または ``config`` から
            組んだ ESN の ``state_noise`` が 0 でない場合 (D-47)。
    """
    require_deterministic_esn(
        state_noise,
        what="simulate_condition は state_noise を受理しません (D-47)",
        why=_NOISE_REJECTION_WHY,
        forbidden=_NOISE_REJECTION_FORBIDDEN,
        remedy=_NOISE_REJECTION_REMEDY,
    )
    probe_rng = make_rng_for(
        esp_stream_seed(config.seeds, SeedStream.PROBE), SeedStream.PROBE, replicate
    )
    initial_states = make_initial_states(
        config.reservoir.n_units, config.drive.n_pairs, probe_rng
    )
    reference = simulate_reference_trajectory(
        config.reservoir,
        config.drive,
        reservoir_seed=esp_stream_seed(config.seeds, SeedStream.RESERVOIR),
        drive_seed=esp_stream_seed(config.seeds, SeedStream.TASK),
        rho=rho,
        leak_rate=leak_rate,
        sigma_u=sigma_u,
        replicate=replicate,
        x0=initial_states[0],
    )
    # 経路非依存の検査 (副)。比較軌道ループの**直前**に置く —— ループへ入って
    # からでは D-14 の外側の乱数消費が既に起きている。引数側の検査だけを消す
    # 変異はここで、ここだけを消す変異は引数側で落ちる (二重化が空虚でない証明:
    # test_simulate_condition_rejects_a_noisy_esn_from_any_route)。
    require_deterministic_esn(
        reference.esn.config.state_noise,
        what=(
            "simulate_condition の比較軌道は state_noise>0 の ESN では作れません (D-47)"
        ),
        why=_NOISE_REJECTION_WHY,
        forbidden=_NOISE_REJECTION_FORBIDDEN,
        remedy=_NOISE_REJECTION_REMEDY,
    )
    return Trajectories(
        esn=reference.esn,
        drive=reference.drive,
        states=reference.states,
        # F-3b1-2-007 / D-47: state_noise=0 では比較軌道は乱数を1個も引かないので
        # D-14 の3ストリーム分離 (RESERVOIR / TASK / PROBE) が保たれる。
        # state_noise>0 で何が壊れるか (4本目の未制御な変動 / 評価順依存) は
        # 上の require_deterministic_esn のメッセージが説明しており、04a T2 で
        # 「この経路は受理しない」と決めた (5本目のストリームは新設しない)。
        companions=tuple(
            reference.esn.run(reference.drive, x0=x0, rng=reference.rng)
            for x0 in initial_states[1:]
        ),
    )


@dataclass(frozen=True, slots=True)
class ConditionOutcome:
    """1条件ぶんの結果。行に加えて図が必要とする曲線を持つ。

    仕様 §4 T3 は ``evaluate_condition -> EspRow`` と書いていたが、2-A の図は
    距離曲線そのものが主役であり、行だけ返すと図のために全条件をもう一度
    回すことになる (実測 77 秒が倍になる)。``row`` が ``EspRow`` である以上
    CSV 列順の単一の真実は変わらない。
    """

    row: EspRow
    distance: FloatArray
    acf: FloatArray
    diagnostics: tuple[DiagnosticScalarRow, ...] = ()
    """診断のスカラを長形式で運ぶ (D-118)。``esp_diagnostics.csv`` の列は動かさない。"""


def evaluate_condition(
    config: Esp02Config,
    *,
    experiment: str,
    rho: float,
    leak_rate: float,
    sigma_u: float,
    replicate: int,
) -> ConditionOutcome:
    """1条件 (rho, leak_rate, sigma_u, replicate) を回して3診断を取る。

    乱数は3ストリームに分ける (D-14)。リザバー重み ``RESERVOIR`` / 駆動信号
    ``TASK`` / 初期状態対 ``PROBE`` が独立なので、「初期状態だけを振ったときに
    判定が変わるか」を重みを固定したまま測れる。

    Args:
        config: 02 の設定。
        experiment: CSV の ``experiment`` 列 (``EXPERIMENT_*``)。
        rho: スペクトル半径。
        leak_rate: リーク率。
        sigma_u: 駆動信号の標準偏差 (D-17)。
        replicate: レプリケート番号 (0 始まり)。

    Returns:
        ``EspRow`` と、2-A / 2-B の図が使う距離曲線・自己相関曲線。
    """
    started = time.perf_counter()
    drive_config = config.drive
    reservoir_config = config.reservoir

    trajectories = simulate_condition(
        config, rho=rho, leak_rate=leak_rate, sigma_u=sigma_u, replicate=replicate
    )
    u = trajectories.drive
    states = trajectories.states

    # ESP の距離当てはめは過渡そのものが測定対象なので washout を 0 にする。
    # λ と自己相関は過渡を捨てたいので drive.washout を使う (両者は別の要求)。
    distance_ctx = DiagnosticContext(
        washout=ESP_DISTANCE_WASHOUT, companion_states=trajectories.companions
    )
    dynamics_ctx = DiagnosticContext(
        washout=drive_config.washout,
        propagator=esn_propagator(trajectories.esn, u),
    )
    esp = esp_convergence(states, ctx=distance_ctx, cfg=config.esp)
    lyapunov = conditional_lyapunov(states, ctx=dynamics_ctx, cfg=config.lyapunov)
    timescale = autocorrelation_time(states, ctx=dynamics_ctx, cfg=config.timescale)

    row = EspRow(
        experiment=experiment,
        replicate=replicate,
        seed_reservoir=config.seeds.reservoir,
        seed_drive=config.seeds.drive,
        seed_probe=config.seeds.probe,
        rho=rho,
        leak_rate=leak_rate,
        input_scale=reservoir_config.input_scale,
        sigma_u=sigma_u,
        input_amplitude=_SQRT3 * sigma_u,
        input_drive_std=float(np.std(u)),
        n_units=reservoir_config.n_units,
        density=reservoir_config.density,
        n_steps=drive_config.n_steps,
        washout=drive_config.washout,
        window=config.esp.window,
        n_pairs=drive_config.n_pairs,
        d_initial=esp.scalars["d_initial"],
        d_tail=esp.scalars["d_tail"],
        converged=int(esp.scalars["converged"]),
        decay_rate_per_step=esp.scalars["decay_rate_per_step"],
        lyapunov_per_step=lyapunov.scalars["lyapunov_per_step"],
        lyapunov_per_time=lyapunov.scalars["lyapunov_per_time"],
        tau_1e=timescale.scalars["tau_1e"],
        tau_censored=timescale.scalars["tau_censored"],
        tau_integrated=timescale.scalars["tau_integrated"],
        wall_time_s=time.perf_counter() - started,
    )
    logger.debug(
        "experiment=%s rep=%d rho=%.3f leak=%.2f sigma_u=%.3f "
        "converged=%d lambda=%+.4f tau_1e=%.3f (%.3fs)",
        row.experiment,
        row.replicate,
        row.rho,
        row.leak_rate,
        row.sigma_u,
        row.converged,
        row.lyapunov_per_step,
        row.tau_censored,
        row.wall_time_s,
    )
    return ConditionOutcome(
        row=row,
        distance=esp.arrays["distance"],
        acf=timescale.arrays["acf"],
        # 診断のスカラは長形式へ逃がす (D-118)。主表の列は1つも動かさない。
        diagnostics=esp_diagnostic_rows(row, (esp, lyapunov, timescale)),
    )


def run_decay_sweep(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-A: rho を振って状態距離の減衰曲線を得る (受け入れ条件1)。"""
    section = config.decay
    return _sweep(
        config,
        EXPERIMENT_DECAY,
        tuple((rho, section.leak_rate, section.sigma_u) for rho in section.rho_grid),
    )


def run_timescale_sweep(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-B: リーク率を振って実効時定数を得る (受け入れ条件4)。"""
    section = config.timescale_sweep
    return _sweep(
        config,
        EXPERIMENT_TIMESCALE,
        tuple(
            (section.rho, leak_rate, section.sigma_u)
            for leak_rate in section.leak_rate_grid
        ),
    )


def run_esp_map(config: Esp02Config) -> tuple[ConditionOutcome, ...]:
    """実験 2-C: rho x 入力強度 の ESP 成立領域 (受け入れ条件2。記事の目玉)。"""
    section = config.esp_map
    return _sweep(
        config,
        EXPERIMENT_ESP_MAP,
        tuple(
            (rho, section.leak_rate, sigma_u)
            for rho in section.rho_grid
            for sigma_u in section.sigma_grid
        ),
    )


def _sweep(
    config: Esp02Config,
    experiment: str,
    axes: tuple[tuple[float, float, float], ...],
) -> tuple[ConditionOutcome, ...]:
    """``(rho, leak_rate, sigma_u)`` の並びをレプリケートぶん回す共通ループ。"""
    n_replicates = config.reservoir.n_replicates
    if n_replicates < 1:
        raise ValueError(
            f"reservoir.n_replicates は 1 以上である必要があります: {n_replicates}"
        )
    started = time.perf_counter()
    outcomes = tuple(
        evaluate_condition(
            config,
            experiment=experiment,
            rho=rho,
            leak_rate=leak_rate,
            sigma_u=sigma_u,
            replicate=replicate,
        )
        for replicate in range(n_replicates)
        for rho, leak_rate, sigma_u in axes
    )
    logger.info(
        "experiment=%s 条件数=%d (軸=%d x レプリケート=%d) wall_time=%.2fs",
        experiment,
        len(outcomes),
        len(axes),
        n_replicates,
        time.perf_counter() - started,
    )
    return outcomes


@dataclass(frozen=True, slots=True)
class EspResults:
    """3実験ぶんの結果 (図はそれぞれ別の実験の結果だけを見る)。"""

    decay: tuple[ConditionOutcome, ...]
    timescale: tuple[ConditionOutcome, ...]
    esp_map: tuple[ConditionOutcome, ...]

    @property
    def outcomes(self) -> tuple[ConditionOutcome, ...]:
        """3実験ぶんを宣言順につなげたもの。"""
        return self.decay + self.timescale + self.esp_map

    @property
    def rows(self) -> tuple[EspRow, ...]:
        """``esp_diagnostics.csv`` に書く行 (3実験ぶん)。"""
        return tuple(outcome.row for outcome in self.outcomes)


def run_esp_experiment(config: Esp02Config) -> EspResults:
    """実験 2-A / 2-B / 2-C をこの順に回す。"""
    return EspResults(
        decay=run_decay_sweep(config),
        timescale=run_timescale_sweep(config),
        esp_map=run_esp_map(config),
    )


@dataclass(frozen=True, slots=True)
class VerdictAgreement:
    """λ の符号と ESP 判定の整合の要約 (``meta.json`` に載せる)。

    **整合の要求は非対称である** (D-20)。

    - ``λ > 0`` なのに収束 (**偽の ESP**) は起きてはならない
    - ``λ < 0`` なのに非収束は起きうる

    どこで起きたかを追えるように sigma_u と rho の分布まで残す。
    """

    boundary_lambda: float
    strong_drive_sigma: float
    n_rows: int
    n_near_boundary: int
    n_compared: int
    n_false_esp: int
    n_local_but_not_global: int
    disagreement_rate: float
    n_compared_strong_drive: int
    n_disagreement_strong_drive: int
    disagreement_by_sigma: tuple[tuple[float, int], ...]
    disagreement_by_rho: tuple[tuple[float, int], ...]

    def to_summary(self) -> dict[str, object]:
        """``meta.json`` に載せるプレーンな dict。"""
        return {
            "boundary_lambda": self.boundary_lambda,
            "strong_drive_sigma": self.strong_drive_sigma,
            "n_rows": self.n_rows,
            "n_near_boundary": self.n_near_boundary,
            "n_compared": self.n_compared,
            "n_false_esp": self.n_false_esp,
            "n_local_but_not_global": self.n_local_but_not_global,
            "disagreement_rate": self.disagreement_rate,
            "n_compared_strong_drive": self.n_compared_strong_drive,
            "n_disagreement_strong_drive": self.n_disagreement_strong_drive,
            "disagreement_by_sigma": [
                {"sigma_u": sigma, "n": count}
                for sigma, count in self.disagreement_by_sigma
            ],
            "disagreement_by_rho": [
                {"rho": rho, "n": count} for rho, count in self.disagreement_by_rho
            ],
        }


def _is_disagreement(row: EspRow) -> bool:
    """境界帯の外で λ の符号と ESP 判定が食い違っているか。"""
    return (row.lyapunov_per_step < 0.0) != (row.converged == 1)


def summarize_verdict_agreement(
    rows: tuple[EspRow, ...],
    *,
    boundary_lambda: float = BOUNDARY_LAMBDA,
    strong_drive_sigma: float = STRONG_DRIVE_SIGMA,
) -> VerdictAgreement:
    """λ の符号と ESP 判定の整合を集計する (受け入れ条件3 の一次資料)。"""
    compared = [
        row
        for row in rows
        if math.isfinite(row.lyapunov_per_step)
        and abs(row.lyapunov_per_step) > boundary_lambda
    ]
    disagreements = [row for row in compared if _is_disagreement(row)]
    strong = [row for row in compared if row.sigma_u >= strong_drive_sigma]
    return VerdictAgreement(
        boundary_lambda=boundary_lambda,
        strong_drive_sigma=strong_drive_sigma,
        n_rows=len(rows),
        n_near_boundary=len(rows) - len(compared),
        n_compared=len(compared),
        n_false_esp=sum(1 for row in disagreements if row.lyapunov_per_step > 0.0),
        n_local_but_not_global=sum(
            1 for row in disagreements if row.lyapunov_per_step < 0.0
        ),
        disagreement_rate=len(disagreements) / len(compared) if compared else 0.0,
        n_compared_strong_drive=len(strong),
        n_disagreement_strong_drive=sum(1 for row in strong if _is_disagreement(row)),
        disagreement_by_sigma=tuple(
            sorted(Counter(row.sigma_u for row in disagreements).items())
        ),
        disagreement_by_rho=tuple(
            sorted(Counter(row.rho for row in disagreements).items())
        ),
    )


def esp_defaults(config: Esp02Config) -> dict[str, object]:
    """02 の判定基準と、YAML に出てこない実装側の固定値 (``meta.json`` 用)。

    T5 が ``docs/design.md`` §9 に既定値の根拠を書くときの一次資料になる。
    YAML から設定できる値 (``config`` 全体) は ``meta.json`` の ``config``
    に別途載るので、ここには**コード側にしか無い値**を明示的に並べる。
    """
    return {
        "bias_scale": BIAS_SCALE,
        "esp_distance_washout": ESP_DISTANCE_WASHOUT,
        "boundary_lambda": BOUNDARY_LAMBDA,
        "strong_drive_sigma": STRONG_DRIVE_SIGMA,
        "drive_distribution": config.drive.distribution,
        "input_amplitude_per_sigma": _SQRT3,
        "esp": {
            "abs_tol": config.esp.abs_tol,
            "rel_tol": config.esp.rel_tol,
            "window": config.esp.window,
            "fit_skip": config.esp.fit_skip,
            "floor": config.esp.floor,
        },
        "lyapunov": {
            "method": config.lyapunov.method,
            "delta": config.lyapunov.delta,
            "renorm_interval": config.lyapunov.renorm_interval,
            "max_growth": config.lyapunov.max_growth,
            "check_propagator": config.lyapunov.check_propagator,
            "propagator_tol": config.lyapunov.propagator_tol,
        },
        "timescale": {"max_lag": config.timescale.max_lag},
    }


__all__ = [
    "BIAS_SCALE",
    "BOUNDARY_LAMBDA",
    "ESP_CSV_COLUMNS",
    "ESP_DISTANCE_WASHOUT",
    "EXPERIMENT_DECAY",
    "EXPERIMENT_ESP_MAP",
    "EXPERIMENT_TIMESCALE",
    "STRONG_DRIVE_SIGMA",
    "UNIFORM",
    "ConditionOutcome",
    "EspResults",
    "EspRow",
    "ReferenceTrajectory",
    "Trajectories",
    "VerdictAgreement",
    "build_esn_config",
    "esn_propagator",
    "esp_defaults",
    "evaluate_condition",
    "make_drive",
    "make_initial_states",
    "require_deterministic_esn",
    "run_decay_sweep",
    "run_esp_experiment",
    "run_esp_map",
    "run_timescale_sweep",
    "simulate_condition",
    "simulate_reference_trajectory",
    "summarize_verdict_agreement",
]
