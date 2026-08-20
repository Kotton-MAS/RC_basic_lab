"""自走軌道を数値で評価する純関数層 (D-43 / D-45 / D-46).

このモジュールは**設定も ESN も課題も知らない**。受け取るのは配列だけで、
返すのは数値と判定である。実験の配線 (どの軌道を渡すか) は
``experiment/freerun.py`` と ``experiment/stability.py`` が持つ。

分けてある理由は3つある。

1. **3態分類を図や目視で決めない** (仕様 §5 禁止する構造6 / D-45)。分類が
   純関数であれば、判定の根拠は引数と定数だけになり「図を見て閾値を動かす」
   経路が構造上存在しない。
2. **アトラクタ再現を視覚評価で結論しない** (D-46)。距離指標を2本 (リターン
   マップの点集合距離・パワースペクトルの距離) 返し、**真の軌道のシャッフル
   代替**と比べる形をここに閉じる。
3. **確保軸4 と確保軸7 の在り処を1か所にする**。長時間統計の系列長
   (``stats_steps``) には上書き不能な絶対上限を置き (``_MAX_STATS_STEPS``)、
   FFT 長は ``stats_steps`` に**従属**させて独立した設定軸にしない
   (仕様 §5 確保軸7。リターンマップ側はビンも分位点も持たない)。

**発散の判定は ``isfinite`` だけでは足りない** (T4 実装メモ 5)。``float64`` の
範囲内で 1e200 まで伸びる軌道は有限値のまま「破綻」しているので、分類器は
**振幅そのもの**を真の軌道と比べる (``AMPLITUDE_RATIO_MAX``)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rc_basics_lab.types import FloatArray

_MAX_STATS_STEPS = 1_000_000
"""確保軸4: 長時間統計に使う自走ステップ数 ``freerun.stats_steps`` の絶対上限。

自走は逐次計算でベクトル化できない (仕様 §10-1) ので、時間はこの値に比例する。
本番設定は 20,000 ステップ (= 200 時間単位) で、上限 1e6 は 50 倍の余裕を残し
つつ、1本あたりの自走を約 25 秒 (実測 25 マイクロ秒/ステップ) に抑える。
**上書き不能な定数**であり設定からは動かせない (D-34 の規律)。

軸3 (``free_run_steps * n_units``、状態行列の確保量) とは**別の軸**である ——
軸3 は行列の要素数を縛るので ``n_units`` を小さくすればいくらでもステップ数を
伸ばせる。時間の側を縛るのがこの軸である。
"""

MIN_STATS_STEPS = 64
"""長時間統計に意味を与える最小のステップ数 (これ未満は ``ValueError``)。"""

VALID_TIME_THRESHOLD_GRID: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
"""有効予測時間の閾値感度 (仕様 §8)。``docs/design.md`` §12 の感度表の一次資料。

本番の閾値 (``freerun.valid_time_threshold``) はこの格子とは**独立**に設定から
来る。格子は「閾値の取り方で結論が変わらないこと」を示すためだけに使う。
"""

AMPLITUDE_RATIO_MAX = 5.0
"""発散と判定する振幅比 (自走のピーク振幅 / 真の軌道のピーク振幅)。

**``isfinite`` では捕まらない破綻を捕まえるための軸**である (T4 実装メモ 5)。
``float64`` の範囲内で 1e200 まで伸びる軌道は有限値のままなので、振幅を見ない
分類器は「発散していないが破綻した」軌道をアトラクタ再現に数える。値 5.0 は
T4 の Delta t 較正が使ったのと同じ基準 (真値の最大振幅の5倍)。
"""

COLLAPSE_STD_RATIO = 1.0 / AMPLITUDE_RATIO_MAX
"""周期軌道 (不動点を含む) と判定する標準偏差比の上限 (= 0.2)。

自走が1点へ潰れると標準偏差が 0 に近づく。不動点は周期1の軌道なので3態の
「周期軌道への落ち込み」に入る。

**``AMPLITUDE_RATIO_MAX`` の逆数にしてある**。判定の倍率を2本持つと片方だけを
動かせてしまい、「図が良く見えるまで閾値を動かす」余地が残る (D-45 が禁じて
いるのはまさにそれ)。「真の軌道の5倍より大きければ発散、1/5 より小さければ
潰れた」という**1つの倍率**で両側を決める。

実測の分離 (Lorenz / ``stats_steps`` = 20,000、``docs/design.md`` §12):
自走が続く ESN は 0.80〜1.01、緩やかに不動点へ落ちる対照 (線形・遅延線) は
0.087〜0.139 で、0.2 は**どちらの側からも遠い**。
"""

PERIODIC_AUTOCORR = 0.95
"""周期軌道と判定する自己相関のピーク値 (最初のゼロ交差以降の最大)。

ラグ1 の自己相関は Delta t = 0.01 では 0.999 に達する (滑らかにサンプルして
いるだけ) ので、**最初のゼロ交差より手前は見ない**。真の Lorenz 軌道での実測
は ``docs/design.md`` §12 に載せる (閾値が真の軌道を周期と誤判定しないことの
実測)。
"""

MIN_RETURN_MAP_POINTS = 8
"""リターンマップの点集合距離を定義できる最小の点数。

これ未満のときは距離を ``nan`` にする。0 を返すと「潰れた軌道ほど真の軌道に
近い」という逆向きの結論になり、nan なら比較 (``<``) が必ず False になるので
「再現できていない」側へ倒れる。
"""

REGIME_DIVERGED = "diverged"
"""3態: 発散 (有限でない値、または真値の ``AMPLITUDE_RATIO_MAX`` 倍超の振幅)。"""

REGIME_PERIODIC = "periodic"
"""3態: 周期軌道への落ち込み (不動点を含む)。"""

REGIME_ATTRACTOR = "attractor"
"""3態: 有界かつ非周期 —— アトラクタ再現の候補。

**再現の良し悪しはここでは測らない** (D-46 の2指標が別途測る)。3態分類は
「発散でも周期でもない」ことしか言わない。
"""

REGIMES: tuple[str, ...] = (REGIME_DIVERGED, REGIME_PERIODIC, REGIME_ATTRACTOR)
"""3態の全体 (``classify_regime`` はこのいずれか1つを必ず返す: 排他かつ網羅)。"""


def validate_stats_bounds(stats_steps: int) -> None:
    """確保軸4 を**自走を1ステップも回す前に**検査する (D-34)。

    Args:
        stats_steps: 長時間統計に使う自走ステップ数。

    Raises:
        ValueError: ``MIN_STATS_STEPS`` 未満、または ``_MAX_STATS_STEPS`` 超過。
    """
    if stats_steps < MIN_STATS_STEPS:
        raise ValueError(
            f"stats_steps は {MIN_STATS_STEPS} 以上である必要があります: "
            f"{stats_steps} (これ未満では長時間統計が定義できない)"
        )
    if stats_steps > _MAX_STATS_STEPS:
        raise ValueError(
            f"stats_steps が上限を超えています: {stats_steps} > {_MAX_STATS_STEPS} "
            "(自走は逐次計算でベクトル化できないため、時間がこの値に比例する)"
        )


def _as_series(series: FloatArray, name: str) -> FloatArray:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} は (T, D) の2次元配列が必要です: {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} が空です: {array.shape}")
    return array


# --- D-43: 有効予測時間 --------------------------------------------------------


def normalized_error_curve(truth: FloatArray, prediction: FloatArray) -> FloatArray:
    """各ステップの誤差を **NRMSE 比**で返す (D-43 / D-02 と同じ正規化)。

    ``e[k] = rms_j(prediction[k, j] - truth[k, j]) / std(truth)`` で、分母は
    ``metrics.nrmse`` と同じ「真の系列の標準偏差 (ddof=0、全成分をまとめて)」
    である。指標の定義を 04 で作り直さないことで、``onestep.csv`` の ``nrmse``
    と有効予測時間の誤差が同じ尺度になる。

    ``prediction`` に ``nan`` (自走が打ち切られた後の行) があれば、その行の
    誤差も ``nan`` になる。``valid_time_from_errors`` は ``nan`` を「閾値を
    超えた」と扱うので、打ち切りが有効予測時間を伸ばすことはない。

    Args:
        truth: 真の軌道 ``(K, D)``。
        prediction: 自走の予測 ``(K, D)``。

    Returns:
        ``(K,)`` の誤差列。

    Raises:
        ValueError: 形状不一致、または ``std(truth) == 0`` の場合。
    """
    true_array = _as_series(truth, "truth")
    pred_array = _as_series(prediction, "prediction")
    if true_array.shape != pred_array.shape:
        raise ValueError(
            f"truth と prediction の形状が違います: "
            f"{true_array.shape} != {pred_array.shape}"
        )
    scale = float(np.std(true_array))
    if scale == 0.0:
        raise ValueError("std(truth) が 0 のため NRMSE 比を定義できません")
    residual: FloatArray = pred_array - true_array
    curve: FloatArray = np.sqrt(np.mean(residual**2, axis=1)) / scale
    return curve


@dataclass(frozen=True, slots=True)
class ValidTime:
    """有効予測時間 (D-43)。**打ち切りを無かったことにしない**。

    Attributes:
        steps: 誤差が閾値を初めて超えるまでのステップ数。
        censored: 自走長の最後まで閾値を超えなかったか (右側打ち切り)。
            ``True`` の行を「その値ちょうどで誤差が超えた」と読むと、
            分布の右端が実際より小さく見える。
        threshold: 判定に使った閾値 (NRMSE 比)。
    """

    steps: int
    censored: bool
    threshold: float


def valid_time_from_errors(errors: FloatArray, threshold: float) -> ValidTime:
    """誤差列から有効予測時間を求める (**純関数**、D-43)。

    ``nan`` は「閾値を超えた」と扱う —— 自走が打ち切られた後の行が
    「まだ当たっている」に化けるのを防ぐ。

    Args:
        errors: ``normalized_error_curve`` の出力 ``(K,)``。
        threshold: NRMSE 比の閾値。

    Returns:
        ``ValidTime``。

    Raises:
        ValueError: ``errors`` が空、または ``threshold`` が正でない場合。
    """
    curve = np.asarray(errors, dtype=np.float64)
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError(f"errors は非空の1次元配列が必要です: {curve.shape}")
    if not threshold > 0.0:
        raise ValueError(f"threshold は正である必要があります: {threshold}")
    exceeded = ~(curve <= threshold)
    if not bool(np.any(exceeded)):
        return ValidTime(steps=int(curve.size), censored=True, threshold=threshold)
    return ValidTime(
        steps=int(np.argmax(exceeded)), censored=False, threshold=threshold
    )


def lyapunov_normalized(steps: int, dt: float, lyapunov_time: float) -> float:
    """ステップ数を **Lyapunov 時間**の単位へ直す (D-43)。

    ``steps * dt / lyapunov_time``。生のステップ数だけで報告することは
    禁止されている (仕様 §5 禁止する構造5) ので、変換をここ1本に閉じる。

    Args:
        steps: 有効予測時間 [ステップ]。
        dt: サンプリング間隔 [時間] (``tasks.chaotic.sampling_interval``)。
        lyapunov_time: ``1 / lambda_max`` [時間] (D-42 の数値推定)。

    Returns:
        Lyapunov 時間で正規化した有効予測時間。``lyapunov_time`` が有限で
        正でなければ ``nan`` (負の Lyapunov 時間で割らない)。
    """
    if not math.isfinite(lyapunov_time) or lyapunov_time <= 0.0:
        return math.nan
    return steps * dt / lyapunov_time


# --- D-46: 長時間統計 (2指標) --------------------------------------------------


def successive_maxima(series: FloatArray, component: int = -1) -> FloatArray:
    """成分の**極大値**を出現順に返す (Lorenz のリターンマップの座標)。

    Lorenz のリターンマップ (z の連続する極大値 ``z_n -> z_(n+1)``) は、
    **時間順序を壊すと再現できない**統計量である。値の分布だけを見る指標だと
    「真の軌道をシャッフルした代替」と区別できない (シャッフルは周辺分布を
    1ビットも変えない) ため、D-46 の1本目にはこれを使う。

    Args:
        series: ``(T, D)``。
        component: どの成分の極大を取るか (既定は最終成分 —— Lorenz なら
            古典的な z、Mackey-Glass なら唯一の成分)。

    Returns:
        極大値 ``(M,)`` (出現順)。
    """
    array = _as_series(series, "series")
    values: FloatArray = array[:, component]
    if values.size < 3:
        return np.empty(0, dtype=np.float64)
    center = values[1:-1]
    is_max = (center > values[:-2]) & (center >= values[2:])
    maxima: FloatArray = center[is_max]
    return maxima


def return_map_points(series: FloatArray, component: int = -1) -> FloatArray:
    """リターンマップの点列 ``(z_n, z_(n+1))`` ``(M-1, 2)`` を返す。

    図 (``fig_freerun_stats.png``) はこの点列を描く。距離指標は
    ``distribution_distance`` が極大値の分布に対して測る。
    """
    maxima = successive_maxima(series, component)
    if maxima.size < 2:
        return np.empty((0, 2), dtype=np.float64)
    points: FloatArray = np.stack([maxima[:-1], maxima[1:]], axis=1)
    return points


_POINT_CHUNK = 2048
"""点集合距離を計算するときの1度に持つ候補点数 (確保量を標本数に比例させない)。"""


def _mean_nearest(left: FloatArray, right: FloatArray) -> float:
    """``left`` の各点から ``right`` の最近傍までの距離の平均。"""
    total = 0.0
    for start in range(0, left.shape[0], _POINT_CHUNK):
        block: FloatArray = left[start : start + _POINT_CHUNK]
        distances: FloatArray = np.linalg.norm(
            block[:, None, :] - right[None, :, :], axis=2
        )
        total += float(np.sum(np.min(distances, axis=1)))
    return total / float(left.shape[0])


def point_set_distance(left: FloatArray, right: FloatArray) -> float:
    """2次元の点集合間の**対称 chamfer 距離** (D-46 の1本目)。

    ``0.5 * (mean_p min_q |p - q| + mean_q min_p |p - q|)``。対称にするのは、
    片側だけだと**潰れた軌道が真のリターンマップ上に乗っているだけ**で距離 0
    になるためである (周期軌道は真の曲線の1点に乗りうる)。

    **2次元ヒストグラムの全変動距離を採らなかったのは実測による**: 真の軌道の
    極大値は 106 点しかなく、同一分布からの独立標本 (真の点列を半分ずつに割った
    もの) でも TV が 0.23〜0.45 (ビン 8〜32) に達して、自走 (0.14〜0.26) と
    重なる。同じ対照で chamfer は 0.030 (同一分布) / 0.011 (自走) /
    0.393 (シャッフル代替) と分離する。数字は ``docs/design.md`` §12。

    ビン数も分位点数も持たないので、確保軸7 (「ビン数を独立の軸にしない」) は
    そもそも軸が存在しない形で満たす。確保量は ``_POINT_CHUNK`` で抑える。

    Args:
        left: 点集合 ``(M, 2)``。
        right: 点集合 ``(M', 2)``。

    Returns:
        距離。どちらかの点数が ``MIN_RETURN_MAP_POINTS`` 未満なら ``nan``
        (「距離 0」にすると潰れた軌道ほど近いという逆向きの結論になる)。
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError(f"点集合は (M, 2) が必要です: {a.shape} / {b.shape}")
    if min(a.shape[0], b.shape[0]) < MIN_RETURN_MAP_POINTS:
        return math.nan
    return 0.5 * (_mean_nearest(a, b) + _mean_nearest(b, a))


def spectrum_length(n_samples: int) -> int:
    """FFT 長 (``n_samples`` 以下の最大の2冪)。**設定軸にしない** (確保軸7)。

    Args:
        n_samples: 使える標本数。

    Returns:
        FFT に使う長さ。``n_samples`` が 2 未満なら 0。
    """
    if n_samples < 2:
        return 0
    return 1 << (int(n_samples).bit_length() - 1)


def power_spectrum(series: FloatArray, dt: float) -> tuple[FloatArray, FloatArray]:
    """成分平均の**正規化パワースペクトル**を返す (D-46 の2本目)。

    各成分について平均を引き Hann 窓をかけて ``rfft`` を取り、パワーを総和1へ
    正規化してから成分平均する。総和1にそろえるのは、自走の振幅が真の軌道と
    多少ずれていても**形の一致**を測るためで、振幅そのものは3態分類
    (``AMPLITUDE_RATIO_MAX``) が別に見ている。

    Args:
        series: ``(T, D)``。
        dt: サンプリング間隔 [時間] (周波数軸の単位を決める)。

    Returns:
        ``(周波数 (F,), 正規化パワー (F,))``。``F = spectrum_length(T)//2 + 1``。

    Raises:
        ValueError: 標本数が足りず FFT 長を取れない場合。
    """
    array = _as_series(series, "series")
    n_fft = spectrum_length(array.shape[0])
    if n_fft < 4:
        raise ValueError(f"パワースペクトルに足りる標本がありません: {array.shape}")
    block: FloatArray = array[:n_fft]
    window: FloatArray = np.hanning(n_fft)
    total: FloatArray = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for index in range(block.shape[1]):
        centered: FloatArray = (block[:, index] - np.mean(block[:, index])) * window
        power: FloatArray = np.abs(np.fft.rfft(centered)) ** 2
        norm = float(np.sum(power))
        if norm > 0.0:
            total += power / norm
    frequencies: FloatArray = np.fft.rfftfreq(n_fft, d=dt)
    scale = float(np.sum(total))
    if scale > 0.0:
        total = total / scale
    return frequencies, total


def spectrum_distance(left: FloatArray, right: FloatArray) -> float:
    """正規化パワースペクトル間の全変動距離 (``0.5 * L1``)。

    0 が完全一致、1 が重なりゼロ。**距離**にそろえてあるので、リターンマップ
    側と同じ向き (小さいほど近い) で比較できる。

    Raises:
        ValueError: 長さが違う場合 (別の FFT 長で測ったスペクトルは比べない)。
    """
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"スペクトルの長さが違います: {a.shape} != {b.shape}")
    return float(0.5 * np.sum(np.abs(a - b)))


def shuffled_surrogate(
    series: FloatArray, rng: np.random.Generator, n_samples: int
) -> FloatArray:
    """真の軌道の**シャッフル代替** (時間順序だけを壊した対照)。

    行 (時刻) を並べ替えるので、成分ごとの周辺分布は1ビットも変わらず
    **時間構造だけ**が消える。「自走がアトラクタを再現している」という主張は、
    この対照より有意に近いことで初めて成立する (D-46) —— 分布だけを見る指標
    なら代替も同じ値になるので、指標が時間構造を見ていることの検査にもなる。

    Args:
        series: 真の軌道 ``(T, D)``。
        rng: 並べ替え用 Generator (**新しいストリームは作らない**、D-14)。
        n_samples: 返す行数 (自走側と同じ標本数にそろえる)。

    Returns:
        ``(min(n_samples, T), D)``。

    Raises:
        ValueError: ``n_samples`` が 1 未満の場合。
    """
    array = _as_series(series, "series")
    if n_samples < 1:
        raise ValueError(f"n_samples は 1 以上である必要があります: {n_samples}")
    order = rng.permutation(array.shape[0])[: min(n_samples, array.shape[0])]
    surrogate: FloatArray = array[order]
    return surrogate


@dataclass(frozen=True, slots=True)
class AttractorDistance:
    """アトラクタ再現の距離2本 (D-46)。**視覚評価は結論に使わない**。

    Attributes:
        return_map: リターンマップ (連続する極大値の対) の点集合距離
            (対称 chamfer)。点が足りなければ ``nan``。
        spectrum: 正規化パワースペクトルの全変動距離。
        n_return_map_points: 距離の推定に使ったリターンマップの点数。
        n_spectrum_bins: スペクトルのビン数 (``stats_steps`` に従属、確保軸7)。
    """

    return_map: float
    spectrum: float
    n_return_map_points: int
    n_spectrum_bins: int


def attractor_distance(
    reference: FloatArray, candidate: FloatArray, dt: float
) -> AttractorDistance:
    """真の軌道と候補軌道の距離を2本まとめて測る (D-46)。

    Args:
        reference: 真の軌道 ``(T, D)``。
        candidate: 比べる軌道 ``(K, D)`` (自走、またはシャッフル代替)。
        dt: サンプリング間隔 [時間]。

    Returns:
        ``AttractorDistance``。スペクトルは**短い方の FFT 長**にそろえて測る
        (長さが違うスペクトルは比べない)。
    """
    reference_array = _as_series(reference, "reference")
    candidate_array = _as_series(candidate, "candidate")
    n_common = min(reference_array.shape[0], candidate_array.shape[0])
    if spectrum_length(n_common) < 4:
        spectrum = math.nan
        n_bins = 0
    else:
        _, reference_power = power_spectrum(reference_array[:n_common], dt)
        _, candidate_power = power_spectrum(candidate_array[:n_common], dt)
        spectrum = spectrum_distance(reference_power, candidate_power)
        n_bins = int(reference_power.size)
    candidate_points = return_map_points(candidate_array)
    return AttractorDistance(
        return_map=point_set_distance(
            candidate_points, return_map_points(reference_array)
        ),
        spectrum=spectrum,
        n_return_map_points=int(candidate_points.shape[0]),
        n_spectrum_bins=n_bins,
    )


# --- D-45: 3態分類 ------------------------------------------------------------


def _autocorrelation(series: FloatArray, component: int) -> FloatArray:
    """ラグ 0 から ``T//2`` までの自己相関 (``r[0] = 1``)。空なら長さ0。

    ラグ 0 を 1.0 にそろえるため、相関に実際に寄与する区間の二乗和で割る
    (全区間の二乗和で割ると ``r[0]`` が約 0.5 になり、閾値の意味が変わる)。
    """
    array = _as_series(series, "series")
    values: FloatArray = array[:, component]
    n_samples = values.size
    if n_samples < 8:
        return np.empty(0, dtype=np.float64)
    centered: FloatArray = values - np.mean(values)
    max_lag = n_samples // 2
    window: FloatArray = centered[: n_samples - max_lag]
    variance = float(np.dot(window, window))
    if variance <= 0.0:
        return np.empty(0, dtype=np.float64)
    correlation: FloatArray = (
        np.correlate(centered, window, mode="valid")[:max_lag] / variance
    )
    return correlation


def first_autocorrelation_zero(series: FloatArray, component: int = 0) -> int:
    """自己相関が初めて 0 以下になるラグ (遅延座標埋め込みの標準的な選び方)。

    1変数の課題 (Mackey-Glass) の位相図は遅延座標 ``(u[t], u[t - lag])`` で
    描く。``lag`` を設定値にすると「図が良く見えるまで動かす」経路ができるので、
    **系列そのものから決める** (真の軌道から決めた 1 個を自走側にも使う ——
    別々に決めると同じ座標系で重ね描きできない)。

    Returns:
        ラグ。決められないときは 1。
    """
    correlation = _autocorrelation(series, component)
    negative = np.nonzero(correlation <= 0.0)[0]
    if negative.size == 0:
        return 1
    return max(1, int(negative[0]))


def autocorrelation_peak(series: FloatArray, component: int = 0) -> float:
    """**最初のゼロ交差以降**の自己相関の最大値 (周期性の尺度)。

    ラグ1 の自己相関は Delta t = 0.01 では 0.999 に達する (滑らかにサンプル
    しているだけ) ので、そのまま最大を取ると全部が周期軌道になる。減衰して
    符号が変わる点より先だけを見ることで「戻ってくる」度合いだけを測る。

    Args:
        series: ``(T, D)``。
        component: 見る成分。

    Returns:
        自己相関のピーク。標本が足りない / 分散が 0 のときは 1.0 (= 変化して
        いない軌道は周期側へ倒す)。
    """
    correlation = _autocorrelation(series, component)
    if correlation.size == 0:
        return 1.0
    negative = np.nonzero(correlation <= 0.0)[0]
    if negative.size == 0:
        return 1.0
    tail: FloatArray = correlation[int(negative[0]) :]
    if tail.size == 0:
        return 1.0
    return float(np.max(tail))


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    """3態分類の結果と、その根拠になった数値 (D-45)。

    判定は ``regime`` だけだが、根拠の数値も返すのは「図から決めていない」
    ことを成果物の列として残すためである (``stability.csv`` の
    ``amplitude_ratio`` / ``std_ratio`` / ``autocorr_peak``)。

    Attributes:
        regime: ``REGIMES`` のいずれか (排他かつ網羅)。
        amplitude_ratio: 自走のピーク振幅 / 真の軌道のピーク振幅。
        std_ratio: 自走の標準偏差 / 真の軌道の標準偏差。
        autocorr_peak: 最初のゼロ交差以降の自己相関のピーク。
        n_samples: 判定に使った (有限な) 行数。
    """

    regime: str
    amplitude_ratio: float
    std_ratio: float
    autocorr_peak: float
    n_samples: int


def classify_regime(
    trajectory: FloatArray, *, reference: FloatArray, diverged: bool
) -> RegimeVerdict:
    """自走軌道を発散 / 周期軌道 / アトラクタの3態へ分類する (**純関数**、D-45)。

    判定は上から順に**排他**で、最後の分岐が残り全部を受けるので**網羅**する。

    1. ``diverged`` (有限でない値が出て打ち切られた) または
       ``amplitude_ratio > AMPLITUDE_RATIO_MAX`` -> ``REGIME_DIVERGED``。
       **振幅を見るのが要点** —— ``float64`` の範囲内で 1e200 まで伸びる軌道は
       有限値のままなので、``isfinite`` だけの判定は破綻を見逃す
       (T4 実装メモ 5)。
    2. ``std_ratio < COLLAPSE_STD_RATIO`` (1点へ潰れた = 不動点) または
       ``autocorrelation_peak >= PERIODIC_AUTOCORR`` -> ``REGIME_PERIODIC``。
    3. それ以外 -> ``REGIME_ATTRACTOR`` (有界かつ非周期)。

    **図も目視も使わない** (仕様 §5 禁止する構造6)。閾値3つはモジュール定数で、
    真の軌道での実測値は ``docs/design.md`` §12 に載せる。

    Args:
        trajectory: 自走軌道の**有限な行だけ** ``(K, D)`` (空でもよい)。
        reference: 真の軌道 ``(T, D)`` (振幅・分散の基準)。
        diverged: 自走が有限でない値で打ち切られたか。

    Returns:
        ``RegimeVerdict``。

    Raises:
        ValueError: ``reference`` が空、または振幅・標準偏差が 0 の場合。
    """
    reference_array = _as_series(reference, "reference")
    reference_amplitude = float(np.max(np.abs(reference_array)))
    reference_std = float(np.std(reference_array))
    if reference_amplitude <= 0.0 or reference_std <= 0.0:
        raise ValueError("真の軌道の振幅・標準偏差が 0 です (基準にできません)")

    candidate = np.asarray(trajectory, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[0] == 0:
        return RegimeVerdict(
            regime=REGIME_DIVERGED,
            amplitude_ratio=math.nan,
            std_ratio=math.nan,
            autocorr_peak=math.nan,
            n_samples=0,
        )
    amplitude_ratio = float(np.max(np.abs(candidate))) / reference_amplitude
    std_ratio = float(np.std(candidate)) / reference_std
    peak = autocorrelation_peak(candidate)
    if diverged or amplitude_ratio > AMPLITUDE_RATIO_MAX:
        regime = REGIME_DIVERGED
    elif std_ratio < COLLAPSE_STD_RATIO or peak >= PERIODIC_AUTOCORR:
        regime = REGIME_PERIODIC
    else:
        regime = REGIME_ATTRACTOR
    return RegimeVerdict(
        regime=regime,
        amplitude_ratio=amplitude_ratio,
        std_ratio=std_ratio,
        autocorr_peak=peak,
        n_samples=int(candidate.shape[0]),
    )


__all__ = [
    "AMPLITUDE_RATIO_MAX",
    "COLLAPSE_STD_RATIO",
    "MIN_RETURN_MAP_POINTS",
    "MIN_STATS_STEPS",
    "PERIODIC_AUTOCORR",
    "REGIMES",
    "REGIME_ATTRACTOR",
    "REGIME_DIVERGED",
    "REGIME_PERIODIC",
    "VALID_TIME_THRESHOLD_GRID",
    "AttractorDistance",
    "RegimeVerdict",
    "ValidTime",
    "attractor_distance",
    "autocorrelation_peak",
    "classify_regime",
    "first_autocorrelation_zero",
    "lyapunov_normalized",
    "normalized_error_curve",
    "point_set_distance",
    "power_spectrum",
    "return_map_points",
    "shuffled_surrogate",
    "spectrum_distance",
    "spectrum_length",
    "successive_maxima",
    "valid_time_from_errors",
    "validate_stats_bounds",
]
