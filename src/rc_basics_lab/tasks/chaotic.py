"""カオス時系列の課題 (実験04) —— Lorenz の生成と、標準化の単一の真実 (D-41).

Lorenz 系::

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

を古典的 RK4 (刻み ``rk4_step``) で積分し、``sample_interval`` ステップごとに
サブサンプルする。パラメータは **(sigma, rho, beta) = (10, 28, 8/3)** に固定
する (D-41): この1点だけが「蝶形アトラクタ」と文献値 lambda_max = 0.9056
(``config.chaos04.LORENZ_LYAPUNOV_REFERENCE``) を同時に指すので、設定
フィールドにすると照合値の意味が黙って失われる (``tasks/narma.py`` の係数を
設定にしない D-29 と同じ流儀)。

**Mackey-Glass は再実装しない**。``generate_standardized_mackey_glass`` は
``tasks/mackey_glass.py`` の ``generate_mackey_glass`` へ委譲し、04 が要求する
標準化だけを足す薄い adapter である。

**標準化係数は訓練区間から推定した1組を全区間で使う** (D-41、仕様 §10-2)。
自走中や評価区間で再推定すると、モデルが当てられていない区間でも予測と真値の
平均・分散が揃うため「予測が当たっているように見える」壊れ方をし、
**位相図でも有効予測時間でも検出できない**。``Standardizer`` を値として持ち回る
のはこのためで、係数を作れる場所を ``Standardizer.from_training_prefix``
1箇所に閉じてある。

確保軸 (D-34 の規律「確保より前に落とす」) は2本あり、``_validate`` が**生成前**に
1本ずつ検査する:

1. ``(length + horizon + integration_burn_in) * sample_interval`` = 積分ステップ数
2. ``length * LORENZ_STATE_DIM`` = 真の軌道の配列要素数
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_basics_lab.config import LorenzConfig, MackeyGlassConfig
from rc_basics_lab.tasks.base import TaskData
from rc_basics_lab.tasks.mackey_glass import generate_mackey_glass
from rc_basics_lab.types import FloatArray

TASK_NAME_LORENZ = "lorenz"

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0
"""古典的な Lorenz (1963) のカオス域のパラメータ (D-41)。**設定にしない**。"""

LORENZ_STATE_DIM = 3
"""Lorenz 系の状態次元 (確保軸2 の係数)。"""

LORENZ_INITIAL_STATE = (1.0, 1.0, 1.0)
"""初期状態の中心 (文献の慣例値)。アトラクタ上ではないので burn-in が要る。"""

LORENZ_INITIAL_JITTER = 0.1
"""初期状態に task ストリームで与える一様揺らぎの幅。

レプリケート間で軌道を変えるためのもの。カオス系なので、この幅の違いは
バーンイン後には初期値の記憶を残さない (MG の ``INITIAL_JITTER`` と同じ役割)。
"""

_MAX_INTEGRATION_STEPS = 20_000_000
"""確保軸1: RK4 の総積分ステップ数の上書き不能な絶対上限 (CWE-400)。

積分は逐次計算でベクトル化できない (仕様 §10-1) ため、時間は
``(length + horizon + integration_burn_in) * sample_interval`` に比例する。
本番設定は ``(8000 + 1 + 1000) * 10`` = 90,010 ステップで、上限 2e7 は 222 倍の
余裕を残しつつ、真の軌道の生成を予算 (60 秒) の内側に抑える (実測: 1 ステップ
あたり約 1.5 マイクロ秒 -> 2e7 ステップで約 30 秒)。``length`` 単体ではなく
``sample_interval`` との積で縛るのは、``sample_interval`` だけを 1000 倍する
設定変更が ``length`` の検査をすり抜けるためである。
"""

_MAX_TRAJECTORY_ELEMENTS = 50_000_000
"""確保軸2: 真の軌道の配列要素数 (``length * LORENZ_STATE_DIM``) の絶対上限。

``float64`` なので 5e7 要素 = 400 MB。ピークメモリ予算 4 GB の 1/10 で、
本番設定 (8000 * 3 = 24,000 要素) の 2000 倍の余裕がある。軸1 とは**別の軸**で
ある —— ``sample_interval=1`` にすれば軸1 を通したまま ``length`` だけを
1000 万オーダーへ伸ばせるので、片方だけでは塞げない。
"""


def _validate(cfg: LorenzConfig) -> None:
    """値域と確保軸を**生成前に**検査する (D-09 / D-34)。"""
    if cfg.rk4_step <= 0.0:
        raise ValueError(f"rk4_step は正である必要があります: {cfg.rk4_step}")
    if cfg.sample_interval < 1:
        raise ValueError(
            f"sample_interval は 1 以上である必要があります: {cfg.sample_interval}"
        )
    if cfg.integration_burn_in < 0:
        raise ValueError(
            "integration_burn_in は 0 以上である必要があります: "
            f"{cfg.integration_burn_in}"
        )
    if cfg.length < 1:
        raise ValueError(f"length は 1 以上である必要があります: {cfg.length}")
    if cfg.horizon < 1:
        raise ValueError(f"horizon は 1 以上である必要があります: {cfg.horizon}")
    if cfg.standardize_steps < 2:
        raise ValueError(
            f"standardize_steps は 2 以上である必要があります: {cfg.standardize_steps}"
        )
    if cfg.standardize_steps > cfg.length:
        raise ValueError(
            "standardize_steps が系列長を超えています "
            f"(訓練区間の内側でなければならない、D-41): "
            f"standardize_steps={cfg.standardize_steps} > length={cfg.length}"
        )
    _validate_allocation_bounds(cfg)


def _validate_allocation_bounds(cfg: LorenzConfig) -> None:
    """確保軸1・2 を1本ずつ検査する (**確保より前に**落とす、D-34)。"""
    n_integration_steps = (
        cfg.length + cfg.horizon + cfg.integration_burn_in
    ) * cfg.sample_interval
    if n_integration_steps > _MAX_INTEGRATION_STEPS:
        raise ValueError(
            "積分ステップ数が上限を超えています: "
            f"{n_integration_steps} > {_MAX_INTEGRATION_STEPS} "
            "((length + horizon + integration_burn_in) * sample_interval。"
            "RK4 は逐次計算なので時間がこの積に比例する)"
        )
    n_elements = cfg.length * LORENZ_STATE_DIM
    if n_elements > _MAX_TRAJECTORY_ELEMENTS:
        raise ValueError(
            "真の軌道の配列要素数が上限を超えています: "
            f"{n_elements} > {_MAX_TRAJECTORY_ELEMENTS} "
            "(length * 状態次元。確保する前に検査で落とす)"
        )


@dataclass(frozen=True, slots=True)
class Standardizer:
    """平均0・分散1へ移す成分ごとのアフィン変換 (D-41)。

    **係数を作れる場所を ``from_training_prefix`` 1本に閉じる**。値として
    持ち回れば、自走中や評価区間で「その区間から推定し直した」係数が紛れ込む
    余地が構造上なくなる (仕様 §10-2)。

    Attributes:
        mean: 成分ごとの平均 ``(D,)``。
        scale: 成分ごとの標準偏差 ``(D,)`` (ddof=0)。
    """

    mean: FloatArray
    scale: FloatArray

    @classmethod
    def from_training_prefix(cls, series: FloatArray, n_steps: int) -> Standardizer:
        """先頭 ``n_steps`` 行から係数を推定する (D-41)。

        Args:
            series: ``(T, D)`` の系列。
            n_steps: 係数の推定に使う先頭行数。**訓練区間の内側**であること。

        Raises:
            ValueError: 形状不正、``n_steps`` が範囲外、または推定した標準偏差に
                0 が含まれる場合 (定数成分は標準化できない)。
        """
        array = np.asarray(series, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"series は (T, D) の2次元配列が必要です: {array.shape}")
        if not 2 <= n_steps <= array.shape[0]:
            raise ValueError(
                "n_steps は 2 以上・系列長以下である必要があります: "
                f"n_steps={n_steps}, T={array.shape[0]}"
            )
        prefix = array[:n_steps]
        scale: FloatArray = np.std(prefix, axis=0)
        if not np.all(scale > 0.0):
            raise ValueError(
                f"標準偏差が 0 の成分があるため標準化できません: scale={scale!r}"
            )
        return cls(mean=np.mean(prefix, axis=0), scale=scale)

    def apply(self, series: FloatArray) -> FloatArray:
        """``(series - mean) / scale``。"""
        standardized: FloatArray = (
            np.asarray(series, dtype=np.float64) - self.mean
        ) / self.scale
        return standardized

    def invert(self, series: FloatArray) -> FloatArray:
        """``apply`` の逆変換 (自走の予測を物理量へ戻すときに使う)。"""
        original: FloatArray = (
            np.asarray(series, dtype=np.float64) * self.scale + self.mean
        )
        return original


def _derivative(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Lorenz の右辺 (状態は3つのスカラで持つ)。"""
    return (
        LORENZ_SIGMA * (y - x),
        x * (LORENZ_RHO - z) - y,
        x * y - LORENZ_BETA * z,
    )


def _rk4_step(x: float, y: float, z: float, step: float) -> tuple[float, float, float]:
    """古典的 RK4 の1ステップ。

    numpy 配列ではなく Python の float 3個で持つ。3次元の系では配列の生成
    コストが演算より大きく、逐次ループ (仕様 §10-1: ベクトル化できない) では
    そのオーバーヘッドがそのまま実行時間になる。
    """
    k1x, k1y, k1z = _derivative(x, y, z)
    half = 0.5 * step
    k2x, k2y, k2z = _derivative(x + half * k1x, y + half * k1y, z + half * k1z)
    k3x, k3y, k3z = _derivative(x + half * k2x, y + half * k2y, z + half * k2z)
    k4x, k4y, k4z = _derivative(x + step * k3x, y + step * k3y, z + step * k3z)
    sixth = step / 6.0
    return (
        x + sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + sixth * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + sixth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


def initial_state(rng: np.random.Generator) -> FloatArray:
    """レプリケートごとの初期状態 ``(3,)`` を task ストリームから引く。"""
    jitter: FloatArray = rng.uniform(
        -LORENZ_INITIAL_JITTER, LORENZ_INITIAL_JITTER, LORENZ_STATE_DIM
    )
    return np.asarray(LORENZ_INITIAL_STATE, dtype=np.float64) + jitter


def lorenz_sample_step(cfg: LorenzConfig, state: FloatArray) -> FloatArray:
    """状態を**1サンプル** (``sample_interval`` 積分ステップ) 進める。

    ``diagnostics/lyapunov.py`` の ``ctx.propagator`` はこれを渡す。
    ``integrate_lorenz`` とまったく同じ演算列を通るので、
    ``propagator(X[t], t)`` は ``X[t+1]`` とビット単位で一致する
    (``conditional_lyapunov`` の伝播器整合検査 D-18 が実行時に確かめる)。
    """
    array = np.asarray(state, dtype=np.float64)
    if array.shape != (LORENZ_STATE_DIM,):
        raise ValueError(
            f"状態は ({LORENZ_STATE_DIM},) である必要があります: {array.shape}"
        )
    x, y, z = float(array[0]), float(array[1]), float(array[2])
    for _ in range(cfg.sample_interval):
        x, y, z = _rk4_step(x, y, z, cfg.rk4_step)
    return np.array((x, y, z), dtype=np.float64)


def integrate_lorenz(cfg: LorenzConfig, x0: FloatArray, n_samples: int) -> FloatArray:
    """``x0`` から積分し、バーンイン後の ``n_samples`` サンプルを返す。

    ``x0`` を引数に取るのは、独立実装 (``scipy.integrate.solve_ivp``) との
    突き合わせ (D-41 の guard_test) で**同じ初期条件**から出発できるように
    するためである。乱数から初期状態を引く経路は ``initial_state`` にある。

    Args:
        cfg: 生成パラメータ (刻み・サンプリング間隔・バーンイン)。
        x0: 初期状態 ``(3,)``。
        n_samples: 返すサンプル数 (バーンインぶんは含まない)。

    Returns:
        ``(n_samples, 3)``。``k`` 行目は時刻
        ``(integration_burn_in + k + 1) * sample_interval * rk4_step`` の状態
        (``x0`` 自身は含まない。MG の ``integrate`` と同じ規約)。

    Raises:
        ValueError: 設定が値域外・確保軸を超える、または ``n_samples < 1``。
    """
    _validate(cfg)
    if n_samples < 1:
        raise ValueError(f"n_samples は 1 以上である必要があります: {n_samples}")
    array = np.asarray(x0, dtype=np.float64)
    if array.shape != (LORENZ_STATE_DIM,):
        raise ValueError(
            f"x0 は ({LORENZ_STATE_DIM},) である必要があります: {array.shape}"
        )

    x, y, z = float(array[0]), float(array[1]), float(array[2])
    step = cfg.rk4_step
    interval = cfg.sample_interval
    for _ in range(cfg.integration_burn_in * interval):
        x, y, z = _rk4_step(x, y, z, step)
    trajectory: FloatArray = np.empty((n_samples, LORENZ_STATE_DIM), dtype=np.float64)
    for index in range(n_samples):
        for _ in range(interval):
            x, y, z = _rk4_step(x, y, z, step)
        trajectory[index, 0] = x
        trajectory[index, 1] = y
        trajectory[index, 2] = z
    return trajectory


def lorenz_params(cfg: LorenzConfig) -> dict[str, str]:
    """``TaskData.params`` / ``meta.json`` に載せる生成パラメータ。"""
    return {
        "sigma": str(LORENZ_SIGMA),
        "rho": str(LORENZ_RHO),
        "beta": str(LORENZ_BETA),
        "rk4_step": str(cfg.rk4_step),
        "sample_interval": str(cfg.sample_interval),
        "dt": str(sampling_interval(cfg)),
        "integration_burn_in": str(cfg.integration_burn_in),
        "horizon": str(cfg.horizon),
        "standardize_steps": str(cfg.standardize_steps),
    }


def sampling_interval(cfg: LorenzConfig) -> float:
    """サンプリング間隔 ``Delta t = rk4_step * sample_interval`` [時間]。

    Lyapunov 時間正規化の分母 (``DiagnosticContext.dt``) はここが単一の真実。
    """
    return cfg.rk4_step * cfg.sample_interval


def generate_lorenz(cfg: LorenzConfig, rng: np.random.Generator) -> TaskData:
    """Lorenz の ``horizon`` ステップ先予測の課題を作る (標準化込み、D-41)。

    ``u[t] = x_std[t]``, ``y[t] = x_std[t + horizon]``。標準化係数は
    **先頭 ``cfg.standardize_steps`` サンプル (訓練区間) から推定した1組**を
    ``u`` と ``y`` の全区間へ使う。``u`` と ``y`` で別の係数を使うと、自走
    (出力を入力へ戻す) の時点で単位が食い違う。

    Args:
        cfg: 生成パラメータ。
        rng: task ストリームの Generator (D-06)。初期状態の揺らぎに使う。
    """
    _validate(cfg)
    series = integrate_lorenz(cfg, initial_state(rng), cfg.length + cfg.horizon)
    standardizer = Standardizer.from_training_prefix(series, cfg.standardize_steps)
    standardized = standardizer.apply(series)
    return TaskData(
        u=standardized[: cfg.length],
        y=standardized[cfg.horizon : cfg.horizon + cfg.length],
        name=TASK_NAME_LORENZ,
        params=lorenz_params(cfg),
    )


def generate_standardized_mackey_glass(
    cfg: MackeyGlassConfig, rng: np.random.Generator, *, standardize_steps: int
) -> TaskData:
    """MG 課題に 04 の標準化 (D-41) だけを足す薄い adapter。

    **生成は ``tasks/mackey_glass.py`` へ委譲する** (再実装しない)。04 が足すのは
    「訓練区間から推定した1組の係数を全区間で使う」ことだけで、遅延微分方程式の
    積分・履歴の補間・``tau / rk4_step`` の整数性検査は 01 の実装が単一の真実で
    ある。

    ``u`` は元系列の先頭 ``length`` 行、``y`` は同じ系列を ``horizon`` だけ
    ずらしたものなので、``u`` の先頭から推定した係数を両方へ当てるのが
    「訓練区間から推定した1組」の定義そのものになる。

    Args:
        cfg: MG の生成パラメータ (``Chaos04Config.base.mackey_glass``)。
        rng: task ストリームの Generator (D-06)。
        standardize_steps: 係数の推定に使う先頭サンプル数。

    Raises:
        ValueError: ``standardize_steps`` が範囲外の場合。
    """
    data = generate_mackey_glass(cfg, rng)
    standardizer = Standardizer.from_training_prefix(data.u, standardize_steps)
    return TaskData(
        u=standardizer.apply(data.u),
        y=standardizer.apply(data.y),
        name=data.name,
        params={**data.params, "standardize_steps": str(standardize_steps)},
    )


__all__ = [
    "LORENZ_BETA",
    "LORENZ_INITIAL_JITTER",
    "LORENZ_INITIAL_STATE",
    "LORENZ_RHO",
    "LORENZ_SIGMA",
    "LORENZ_STATE_DIM",
    "TASK_NAME_LORENZ",
    "Standardizer",
    "generate_lorenz",
    "generate_standardized_mackey_glass",
    "initial_state",
    "integrate_lorenz",
    "lorenz_params",
    "lorenz_sample_step",
    "sampling_interval",
]
