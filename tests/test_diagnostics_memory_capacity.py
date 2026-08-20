"""線形メモリ容量 (MC) と容量カーネルのテスト (仕様 rc-basics-03 T1)。

状態系列はすべて**このパッケージの ``reservoir`` を通さず**テスト内で作る
(受け入れ条件6)。``diagnostics`` が移植可能であることの実体はここにある。
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

import rc_basics_lab.diagnostics._capacity as capacity_module
from rc_basics_lab.diagnostics._capacity import (
    HERMITE,
    LEGENDRE,
    NORMAL,
    NORMAL_HERMITE,
    UNIFORM,
    UNIFORM_LEGENDRE,
    CapacityProblem,
    InputMeasure,
    bounded_chunk_size,
    capacity_of_targets,
    orthonormal_basis,
)
from rc_basics_lab.diagnostics.base import DiagnosticContext, DiagnosticResult
from rc_basics_lab.diagnostics.memory_capacity import (
    THRESHOLD_NONE,
    THRESHOLD_SURROGATE,
    MemoryCapacityConfig,
    _iter_delay_chunks,
    memory_capacity,
)
from rc_basics_lab.readout.ridge import fit_ridge_from_gram
from rc_basics_lab.types import FloatArray

CTX_SEED = 20240303
"""サロゲート閾値に使う ``ctx.seed`` (D-27: 乱数源はこれだけ)。"""

INPUT_SCALE = 0.1
"""テスト用リザバーの入力スケール。

小さくしてあるのは意図的で、``tanh`` を飽和させない準線形領域に置くため。
飽和領域 (``input_scale=1``) では ``rho`` を上げても線形記憶が伸びず、
「rho が上がると記憶が伸びる」という受け入れ条件1 の現象そのものが消える
(実測: ``input_scale=1.0``, T=20000 で ``mc_effective_delay`` の
rho=0.99 / rho=0.5 比は 5 シード中 4 シードで 1.5 を下回る)。
"""


def _external_reservoir_states(
    rho: float,
    *,
    n_units: int,
    n_steps: int,
    seed: int,
    input_scale: float = INPUT_SCALE,
    state_noise: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """``rc_basics_lab.reservoir`` を使わずに作る状態系列と入力 (受け入れ条件6)。

    ``x[t] = tanh(W x[t-1] + w_in u[t])`` という素の ESN。診断が外部由来の
    状態系列で動くことを示すのが目的なので、本体の ESN 実装は一切通さない。
    """
    rng = np.random.default_rng(seed)
    weights: FloatArray = rng.standard_normal((n_units, n_units))
    weights *= rho / max(abs(np.linalg.eigvals(weights)))
    w_in: FloatArray = rng.uniform(-1.0, 1.0, size=n_units) * input_scale
    inputs: FloatArray = rng.uniform(-1.0, 1.0, size=(n_steps, 1))
    states: FloatArray = np.empty((n_steps, n_units), dtype=np.float64)
    state: FloatArray = np.zeros(n_units, dtype=np.float64)
    for step in range(n_steps):
        state = np.tanh(weights @ state + w_in * inputs[step, 0])
        if state_noise > 0.0:
            state = state + rng.normal(0.0, state_noise, size=n_units)
        states[step] = state
    return states, inputs


@functools.cache
def _cached_states(
    rho: float, n_units: int, n_steps: int, seed: int
) -> tuple[FloatArray, FloatArray]:
    """状態生成は掃引テストで使い回す (1本 5 秒以内の予算を守るため)。"""
    return _external_reservoir_states(rho, n_units=n_units, n_steps=n_steps, seed=seed)


def _scalars(result: DiagnosticResult) -> dict[str, float]:
    return {key: float(value) for key, value in result.scalars.items()}


# --------------------------------------------------------------------------
# 受け入れ基準1: 保存則 (MC <= N)
# --------------------------------------------------------------------------


_MC_RATIO_LOWER_BOUND = {0.5: 0.25, 0.9: 0.4, 0.99: 0.4}
"""rho ごとの ``mc_ratio`` 下限 (F-03-2-019)。

一律 0.2 は rho=0.5 (実測ベースライン 0.376) では約48〜50%喪失で検出できて
いたが、rho=0.9/0.99 (ベースライン 0.620/0.653) では約65〜70%喪失まで
``capacity_of_chunks`` の戻り値を潰す変異が素通りしていた。IPC の同種テスト
(``saturation_ratio >= 0.5``、ベースライン 0.6831 に対し約30%喪失で検出) と
同水準に揃えるため、``capacity_of_chunks`` を factor 倍する変異を rho ごとに
注入して破断点を実測し (``/tmp`` の使い捨てスクリプト、in-process
monkeypatch)、rho=0.5 は 0.25 (約33%喪失で検出)、rho=0.9/0.99 は 0.4
(約36%/39%喪失で検出) に個別値化した。詳細は
docs/review-findings-03.md の F-03-2-019 対応記録を参照。
"""


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99])
def test_mc_total_does_not_exceed_n_units(rho: float) -> None:
    """N=30, T=5000 で総容量が N を (1.02 倍の余裕込みで) 超えない。

    Dambre 2012 の保存則の次数1成分。上限は状態次元 N であり、これを超えたら
    基底の正規化か行合わせ (D-24) が壊れている。1.02 倍の余裕は有限標本の
    ゆらぎのぶん。

    上限側だけでなく下限も検査する (F-03-1-023)。上限と ``> 0.0`` だけだと、
    容量測定がほぼ意味を成さない状態 (深刻な回帰バグ) でも通ってしまう:
    実測で ``capacity_of_chunks`` の戻り値を 0.02 倍に潰す変異を注入すると
    ``mc_ratio`` は 0.011 まで落ちるが、上限チェックと ``> 0.0`` はどちらも
    成立してこのテストを素通りした。下限は rho ごとに個別値を置く
    (``_MC_RATIO_LOWER_BOUND``、F-03-2-019): IPC の同種テスト
    (``test_ipc_total_does_not_exceed_n_units`` の ``saturation_ratio >=
    0.5``) は約30%喪失で検出するが、MC 側は一律 0.2 だと rho=0.9/0.99 で
    約65〜70%喪失まで検出できなかった (docstring 上の『IPC の相当品』という
    説明は強度差に触れていなかった)。
    """
    n_units = 30
    states, inputs = _cached_states(rho, n_units, 5000, 7)
    result = memory_capacity(
        states, inputs, ctx=DiagnosticContext(washout=100, seed=CTX_SEED)
    )
    scalars = _scalars(result)
    assert scalars["mc_total"] <= n_units * 1.02, (
        f"rho={rho}: mc_total={scalars['mc_total']} が上限 N={n_units} を超えました"
    )
    assert scalars["mc_ratio"] <= 1.02
    assert scalars["mc_total"] > 0.0
    lower_bound = _MC_RATIO_LOWER_BOUND[rho]
    assert scalars["mc_ratio"] >= lower_bound, (
        f"rho={rho}: mc_ratio={scalars['mc_ratio']} が下限 {lower_bound} を"
        " 下回りました (容量測定がほぼ意味を成さない状態を示唆、F-03-1-023)"
    )
    assert scalars["n_delays"] == 400.0
    assert result.arrays["mc_profile"].shape == (400,)


def test_mc_profile_lengthens_with_spectral_radius() -> None:
    """rho を上げると記憶プロファイルが伸びる (受け入れ条件1)。

    ``mc_effective_delay`` (容量重心) が rho について単調非減少で、
    rho=0.99 は rho=0.5 の 1.5 倍以上であること。

    T は保存則テスト (T=5000) より長い 20000 を使う。有限標本による容量の
    かさ上げは ``F/T`` (ここでは 31/T) の桁で入り、T=5000 では 400 本の遅延に
    薄く散った偽陽性が重心を押し上げて rho の効果を覆い隠す (実測: T=5000 では
    5 シード中 5 シードでこの assert が落ちる。T=20000 では元の5シードすべて
    で通る)。3-A の本番設定も T=20000 (仕様 §4 T3)。

    比の下限を 1.5 に固定した根拠 (F-03-1-024): 元の docstring は5シードの
    観測から「比の最小は 1.75」としていたが、これは楽観的だった。より広い
    62シード (任意シード12種 + seed=1000〜1049 の連番50個) で実測すると
    比の最小は 1.559 (閾値 1.5 に対して約4%の余裕しかない) で、62シード中
    失敗は0件。閾値自体を下げる必要はないが (実測で不安定ではない)、
    「1.75」という数値は実際の余裕を過大に伝えていたので実測値に訂正する。
    """
    effective: list[float] = []
    for rho in (0.5, 0.9, 0.99):
        states, inputs = _cached_states(rho, 30, 20000, 7)
        result = memory_capacity(
            states, inputs, ctx=DiagnosticContext(washout=100, seed=CTX_SEED)
        )
        effective.append(_scalars(result)["mc_effective_delay"])

    assert effective == sorted(effective), (
        f"mc_effective_delay が rho に対して単調非減少ではありません: {effective}"
    )
    assert effective[-1] >= 1.5 * effective[0], (
        "rho=0.99 の実効遅延が rho=0.5 の 1.5 倍に届きません: "
        f"{effective[-1]} vs {effective[0]}"
    )


# --------------------------------------------------------------------------
# 受け入れ基準2: 基底の正規直交性 (D-28)
# --------------------------------------------------------------------------


def _basis_gram(
    values: FloatArray, measure: InputMeasure, max_degree: int
) -> FloatArray:
    columns = [
        orthonormal_basis(values, degree, measure) for degree in range(max_degree + 1)
    ]
    matrix: FloatArray = np.column_stack(columns)
    gram: FloatArray = matrix.T @ matrix / float(matrix.shape[0])
    return gram


def test_basis_is_orthonormal_under_declared_input_distribution() -> None:
    """宣言された入力分布に対して基底が正規直交である (D-28)。

    直交していない基底で目標を作ると容量が目標間で二重計上され、保存則が
    「N をわずかに超える」という穏やかな形で破れる —— 図では正常に見えるので
    ここで数値として固定する。

    次数の上限を分布で変えているのは、基底の性質ではなく**推定のばらつき**の
    問題である。IPC が実際に使う組 (uniform x legendre) は次数4まで T=200000 で
    許容差 0.02 に収まる (実測: 3 シードで 0.005 / 0.005 / 0.013)。一方
    normal x hermite は ``He_n`` の2乗が重い裾を持つため ``E[psi_n^2]`` の標本
    誤差そのものが大きく、次数4では T=200000 で 0.03〜0.18、T=2000000 でも
    0.012〜0.021 と、T とともに縮みはするが 0.02 には収まらない。基底の定義
    そのもの (バイアス) は
    ``test_polynomial_family_is_exactly_orthonormal_under_quadrature`` が
    求積法で次数4まで誤差 1e-10 以下で固定しており、そちらには標本誤差が無い。
    """
    n_steps = 200_000
    rng = np.random.default_rng(4649)
    tolerance = 0.02

    half_width = np.sqrt(3.0) * 1.7  # sigma_u = 1.7 の一様分布 (D-17)
    uniform_input: FloatArray = rng.uniform(-half_width, half_width, size=n_steps)
    uniform_gram = _basis_gram(uniform_input, UNIFORM_LEGENDRE, 4)
    assert np.max(np.abs(uniform_gram - np.eye(5))) < tolerance, uniform_gram

    normal_input: FloatArray = rng.normal(0.0, 0.4, size=n_steps)
    normal_gram = _basis_gram(normal_input, NORMAL_HERMITE, 2)
    assert np.max(np.abs(normal_gram - np.eye(3))) < tolerance, normal_gram


def test_polynomial_family_is_exactly_orthonormal_under_quadrature() -> None:
    """基底の定義そのものを求積法で厳密に固定する (D-28、標本誤差ゼロ)。

    上のモンテカルロ検査は「実測の平均・標準偏差で正規化する経路」まで含めて
    見る代わりに標本誤差を抱える。ここでは Gauss-Legendre / Gauss-Hermite
    求積で内積を厳密に評価し、次数4まで ``<psi_i, psi_j> = delta_ij`` を
    1e-10 で固定する。係数 (``sqrt(2n+1)`` / ``1/sqrt(n!)``) を1つ落とすと
    ここが落ちる。
    """
    nodes, weights = np.polynomial.legendre.leggauss(24)
    legendre_matrix: FloatArray = np.column_stack(
        [capacity_module._legendre_normalized(nodes, degree) for degree in range(5)]
    )
    legendre_gram = (legendre_matrix * weights[:, None]).T @ legendre_matrix / 2.0
    np.testing.assert_allclose(legendre_gram, np.eye(5), rtol=0.0, atol=1.0e-10)

    nodes, weights = np.polynomial.hermite_e.hermegauss(24)
    hermite_matrix: FloatArray = np.column_stack(
        [capacity_module._hermite_normalized(nodes, degree) for degree in range(5)]
    )
    hermite_gram = (
        (hermite_matrix * weights[:, None]).T @ hermite_matrix / np.sqrt(2.0 * np.pi)
    )
    np.testing.assert_allclose(hermite_gram, np.eye(5), rtol=0.0, atol=1.0e-10)


def test_input_measure_rejects_unsupported_pairs() -> None:
    """未対応の ``(input_distribution, basis)`` は**構築時点**で ValueError (D-28)。

    黙って Legendre として扱うと、正規入力に対して直交しない基底で目標が
    作られ、容量が二重計上される。04a T3 以前は ``orthonormal_basis`` の中で
    検査しており、対の片方だけを渡す呼び方 (第3引数に既定値があった) が
    型検査を素通りしていた (F-03-1-006)。対を1つの値にまとめた以上、検査は
    **値を作る場所1箇所**にしか無い —— ``orthonormal_basis`` に届いた
    ``InputMeasure`` は定義上すべて対応済みの組である。
    """
    for distribution, basis in ((NORMAL, LEGENDRE), (UNIFORM, HERMITE)):
        with pytest.raises(ValueError, match="D-28"):
            InputMeasure(distribution, basis)
    # 対応する組は作れて、値がそのまま読める (拒否だけの実装で緑にしない)。
    assert (UNIFORM_LEGENDRE.distribution, UNIFORM_LEGENDRE.basis) == (
        UNIFORM,
        LEGENDRE,
    )
    assert (NORMAL_HERMITE.distribution, NORMAL_HERMITE.basis) == (NORMAL, HERMITE)


def test_orthonormal_basis_requires_an_explicit_measure() -> None:
    """``orthonormal_basis`` の第3引数に**既定値が無い** (D-28 の実体)。

    「片方だけ渡す」呼び方 (``orthonormal_basis(u, 2, NORMAL)`` のように
    分布だけを渡し basis は既定の Legendre のまま) が書けてしまうことが
    F-03-1-006 の指摘そのものだった。対を1つの値にしても、第3引数に既定値を
    戻せば同じ罠が復活する (測度を渡し忘れた呼び出しが黙って既定の測度で
    走る)。署名で固定する。
    """
    signature = inspect.signature(orthonormal_basis)
    names = list(signature.parameters)
    assert names == ["u_lagged", "degree", "measure"], names
    measure_param = signature.parameters["measure"]
    assert measure_param.default is inspect.Parameter.empty, (
        "measure に既定値が付いています (片方だけ渡す呼び方が復活します)"
    )
    assert measure_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    # 測度を渡さない呼び出しは実行時にも通らない (署名検査だけで緑にしない)。
    values: FloatArray = np.linspace(-1.0, 1.0, 100)
    with pytest.raises(TypeError):
        # 測度なしの呼び出しは mypy が拒否するので ignore を付ける。この
        # ignore を消せる (= 型検査が通る) 状態になったら既定値が復活した
        # サインであり、上の assert が先に落ちる。
        orthonormal_basis(values, 2)  # type: ignore[call-arg]


def test_degree_one_basis_is_the_same_for_both_distributions() -> None:
    """次数1は分布によらず ``(u - mean) / sigma`` に一致する。

    MC が ``input_distribution`` を設定項目に持たないことの根拠
    (``memory_capacity`` は次数1しか使わない)。
    """
    rng = np.random.default_rng(11)
    values: FloatArray = rng.uniform(-2.0, 2.0, size=5000)
    legendre = orthonormal_basis(values, 1, UNIFORM_LEGENDRE)
    hermite = orthonormal_basis(values, 1, NORMAL_HERMITE)
    standardized = (values - values.mean()) / values.std()
    assert np.allclose(legendre, standardized, rtol=0.0, atol=1e-12)
    assert np.allclose(hermite, standardized, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 受け入れ基準3: alpha 単調性 (D-25)
# --------------------------------------------------------------------------


def test_capacity_is_monotone_decreasing_in_alpha() -> None:
    """容量は alpha に対して単調非増加で、大きな alpha では実際に減る (D-25)。

    正則化は「線形読み出しで到達可能な最大の説明率」という容量の定義を
    系統的に過小評価する。この向きが崩れていたら、容量測定に検証分割の
    alpha 選択 (D-04) を持ち込んではいけないという D-25 の前提が壊れている。

    D-25 の guard_test は IPC 側 (``tests/test_diagnostics_ipc.py``) に置く
    予定だが、T2 が未着手なので当面ここで固定する。
    """
    states, inputs = _cached_states(0.9, 20, 4000, 31)
    ctx = DiagnosticContext(washout=100, seed=CTX_SEED)
    totals = [
        _scalars(
            memory_capacity(
                states,
                inputs,
                ctx=ctx,
                cfg=MemoryCapacityConfig(
                    max_delay=60, alpha=alpha, threshold_mode=THRESHOLD_NONE
                ),
            )
        )["mc_total_raw"]
        for alpha in (1.0e-9, 1.0e-6, 1.0e-3, 1.0, 100.0)
    ]
    for smaller, larger in itertools.pairwise(totals):
        assert larger <= smaller + 1.0e-9, f"alpha に対して容量が増えました: {totals}"
    assert totals[-1] < totals[0], f"alpha=100 で容量が減っていません: {totals}"


# --------------------------------------------------------------------------
# 受け入れ基準4: chunk_size は結果を変えない (仕様 §10-2)
# --------------------------------------------------------------------------


def test_chunk_size_does_not_change_results() -> None:
    """``chunk_size`` は性能パラメータで、結果を1ビットも変えてはいけない。

    他の設定フィールドは「変えたら出力が変わる」ことを要求されるが、これだけは
    **逆向きの要求**である (仕様 §5・§10-2)。チャンク分割にバグ (列の取り違え、
    サロゲート乱数のチャンク依存) があるとここが落ちる。
    """
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    ctx = DiagnosticContext(washout=50, seed=CTX_SEED)
    base_cfg = MemoryCapacityConfig(max_delay=120, n_surrogates=40, chunk_size=256)
    reference = memory_capacity(states, inputs, ctx=ctx, cfg=base_cfg)

    # F-03-2-018: 20_000 は bounded_chunk_size の 128MiB 予算を実際に超え、
    # 無条件切り詰め (キャップ) が発動する (n_samples=1880 で budget<9000)。
    # 「キャップが発動しない規模」しか通っていなかった既存テストに、発動する
    # 規模のケースを1件足す。
    n_samples = int(reference.params["n_samples"])
    capped_chunk_size = 20_000
    assert bounded_chunk_size(capped_chunk_size, n_samples) < capped_chunk_size, (
        "この chunk_size ではキャップが発動しません (テストの前提が崩れています)"
    )

    for chunk_size in (1, 7, 64, 1000, capped_chunk_size):
        other = memory_capacity(
            states,
            inputs,
            ctx=ctx,
            cfg=dataclasses.replace(base_cfg, chunk_size=chunk_size),
        )
        assert set(_scalars(other)) == set(_scalars(reference))
        for key, value in _scalars(reference).items():
            assert other.scalars[key] == pytest.approx(value, rel=1.0e-10), (
                f"chunk_size={chunk_size} で {key} が変わりました"
            )
        np.testing.assert_allclose(
            other.arrays["mc_profile"],
            reference.arrays["mc_profile"],
            rtol=1.0e-10,
            atol=0.0,
        )


def test_memory_capacity_config_fields_change_output() -> None:
    """``chunk_size`` 以外の全フィールドは出力を変える (「設定したのに効かない」除け)。

    ``chunk_size`` だけは逆向きの要求を持つので
    ``test_chunk_size_does_not_change_results`` が別に担当する。フィールドを
    足したのに配線を忘れた場合、この完全性チェックが赤くなる。
    """
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    ctx = DiagnosticContext(washout=50, seed=CTX_SEED)
    base_cfg = MemoryCapacityConfig(max_delay=40, n_surrogates=20, chunk_size=16)
    reference = _scalars(memory_capacity(states, inputs, ctx=ctx, cfg=base_cfg))

    changed: dict[str, MemoryCapacityConfig] = {
        "max_delay": dataclasses.replace(base_cfg, max_delay=55),
        "alpha": dataclasses.replace(base_cfg, alpha=10.0),
        "threshold_mode": dataclasses.replace(base_cfg, threshold_mode=THRESHOLD_NONE),
        "n_surrogates": dataclasses.replace(base_cfg, n_surrogates=60),
        "surrogate_quantile": dataclasses.replace(base_cfg, surrogate_quantile=0.5),
    }
    covered = set(changed) | {"chunk_size"}
    actual = {field.name for field in dataclasses.fields(MemoryCapacityConfig)}
    assert covered == actual, (
        "MemoryCapacityConfig のフィールドに対する検査が不足しています: "
        f"{sorted(actual - covered)}"
    )

    for name, cfg in changed.items():
        other = _scalars(memory_capacity(states, inputs, ctx=ctx, cfg=cfg))
        assert other != reference, f"{name} を変えても出力が変わりません"


# --------------------------------------------------------------------------
# 受け入れ基準5: 行集合の共有 (D-24) と入力要件
# --------------------------------------------------------------------------


def test_all_delays_share_identical_rows() -> None:
    """行集合は ``t0 = max(washout, max_delay)`` で決まり、全遅延で同一 (D-24)。

    ``washout`` が ``max_delay`` より小さい間は結果が一切変わらないことで
    「基準点が単一である」ことを観測可能にする。遅延ごとに使える行を変える
    実装 (深い遅延ほど標本が減る) では、washout を動かした瞬間に浅い遅延の
    容量だけが変わるのでここが落ちる。
    """
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    cfg = MemoryCapacityConfig(max_delay=100, n_surrogates=20, chunk_size=32)
    results = [
        memory_capacity(
            states,
            inputs,
            ctx=DiagnosticContext(washout=washout, seed=CTX_SEED),
            cfg=cfg,
        )
        for washout in (0, 37, 100)
    ]
    for result in results:
        assert result.params["t0"] == "100"
        assert result.params["n_samples"] == "1900"
    for other in results[1:]:
        np.testing.assert_array_equal(
            other.arrays["mc_profile"], results[0].arrays["mc_profile"]
        )

    deeper = memory_capacity(
        states,
        inputs,
        ctx=DiagnosticContext(washout=300, seed=CTX_SEED),
        cfg=cfg,
    )
    assert deeper.params["t0"] == "300", "washout が max_delay を超えたら t0 は washout"


def test_row_alignment_needs_no_state_matrix() -> None:
    """行合わせの検査は**状態行列を経由せずに書ける** (D-24、案C の採用条件)。

    このテストは ``RowAlignment`` が無いと書けない。04a T3 以前、窓計算
    ``series[t0 - delay : t0 - delay + n_samples]`` は
    ``CapacityProblem.lagged`` にあり、``CapacityProblem`` は状態行列と
    その Gram を必ず伴う。そのため窓計算だけを見たいテストは、値を一切
    使わないダミーの状態行列 (``np.zeros``、特異な Gram) を構築していた
    (IPC 側の ``_dummy_problem``)。行合わせを ``t0`` と ``n_samples`` だけの
    値に切り出したので、その必要が消えた ——
    ``RowAlignment(t0=42, n_samples=458)`` を直接構築すればよい。
    ``RowAlignment`` を消して ``lagged`` を ``CapacityProblem`` へ戻す変異は、
    実行時ではなく**収集時**にこのテストを落とす (import が解決しない)。

    値そのものの guard は F-03-1-001 の再発防止でもある: 窓の式は かつて
    MC (2箇所) と IPC (1箇所) が共有カーネルの外でそれぞれ書いており、
    行数さえ合えば任意のオフセットを通した (実測: MC の窓を1ステップ
    ずらしても ``test_all_delays_share_identical_rows`` を含む22テストが
    全て緑のまま通った)。``u[t] = t`` の単調系列なら、返った窓の値から
    入力の index を逆算できる。
    """
    # 状態行列も Gram も作らない。持っているのは2つの整数だけである。
    assert [field.name for field in dataclasses.fields(RowAlignment)] == [
        "t0",
        "n_samples",
    ], "RowAlignment が行合わせ以外のものを持ち始めています (責務が混ざったサイン)"

    n_steps = 500
    ramp: FloatArray = np.arange(n_steps, dtype=np.float64)
    t0 = 42
    n_samples = n_steps - t0
    rows = RowAlignment(t0=t0, n_samples=n_samples)
    for delay in (0, 1, 5, 20, t0):
        window = rows.lagged(ramp, delay)
        assert window.shape == (n_samples,)
        expected: FloatArray = np.arange(
            t0 - delay, t0 - delay + n_samples, dtype=np.float64
        )
        np.testing.assert_array_equal(window, expected)

    # 1ステップずらした変異は D-24 の guard で確実に落ちる (完了条件4)。
    mutated = rows.lagged(ramp, 5 - 1)
    assert not np.array_equal(mutated, rows.lagged(ramp, 5))

    with pytest.raises(ValueError, match="D-24"):
        rows.lagged(ramp, t0 + 1)  # 範囲外 (窓の先頭が負になる)。
    with pytest.raises(ValueError, match="0 以上"):
        rows.lagged(ramp, -1)


def test_row_alignment_from_series_is_the_only_base_point_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """基準点 ``t0`` の算出は MC / IPC が**同じ1本**を呼ぶ (D-24)。

    F-03-1-001 が潰したのは窓の式だけで、``t0 = max(washout, 最大遅延)`` と
    「系列が短すぎます」の拒否は ``memory_capacity`` と ``ipc`` に2箇所
    複製されたまま残っていた。片方だけを手書きへ戻す変異 (例: MC 側だけ
    ``max(...)`` を直接書く) は、両診断の出力が一致している限り既存の
    テストでは検出できない。ここでは ``from_series`` が実際に呼ばれたことを
    数えるので、複製が復活した瞬間に落ちる。
    """
    calls: list[tuple[int, int, int]] = []
    original = RowAlignment.from_series

    def spy(*, n_steps: int, washout: int, max_delay: int) -> RowAlignment:
        calls.append((n_steps, washout, max_delay))
        return original(n_steps=n_steps, washout=washout, max_delay=max_delay)

    monkeypatch.setattr(RowAlignment, "from_series", staticmethod(spy))

    states, inputs = _cached_states(0.9, 15, 2000, 5)
    memory_capacity(
        states,
        inputs,
        ctx=DiagnosticContext(washout=50, seed=CTX_SEED),
        cfg=MemoryCapacityConfig(max_delay=30, n_surrogates=5, chunk_size=16),
    )
    assert calls == [(2000, 50, 30)], (
        f"MC が RowAlignment.from_series を1度だけ呼んでいません: {calls}"
    )

    calls.clear()
    ipc(
        states,
        inputs,
        ctx=DiagnosticContext(washout=50, seed=CTX_SEED),
        cfg=IpcConfig(
            max_delay_by_degree=(20, 8), max_variables=2, n_surrogates=5, chunk_size=16
        ),
    )
    assert calls == [(2000, 50, 20)], (
        f"IPC が RowAlignment.from_series を1度だけ呼んでいません: {calls}"
    )


def test_capacity_problem_rejects_inconsistent_row_alignment() -> None:
    """``RowAlignment`` の行数と状態から切り出した行数が違えば ``ValueError``。

    ``CapacityProblem`` は行合わせを内包するが、目標側は ``rows.n_samples``
    行で作られ、設計行列は ``X[t0:]`` の行数で組まれる。両者が食い違ったまま
    進むと「どちらが本物か」が実行時に決まってしまうので、構築時に一致を
    要求する (D-24 の担い手を1つに保つための構造ガード)。
    """
    rng = np.random.default_rng(31)
    states: FloatArray = rng.standard_normal((300, 5))
    with pytest.raises(ValueError, match="D-24"):
        CapacityProblem.from_states(states, rows=RowAlignment(t0=10, n_samples=289))
    with pytest.raises(ValueError, match="D-24"):
        CapacityProblem.from_states(states, rows=RowAlignment(t0=10, n_samples=291))
    # 一致していれば通る (すべて拒否する実装で緑にしない)。
    problem = CapacityProblem.from_states(
        states, rows=RowAlignment(t0=10, n_samples=290)
    )
    assert problem.n_samples == 290
    assert problem.rows.t0 == 10


def test_solve_and_block_widths_share_one_budget_function() -> None:
    """性能軸と確保軸は ``bounded_chunk_size`` **1本**へ委譲する (D-33)。

    128 MiB の予算を2箇所に持つと、片方だけを動かしても誰も気づけない。
    ``solve_width`` は「運用者が指定した列数」を、``block_width`` は
    「実体化したい列数」を入力に取るという違いしか無く、予算による切り詰めは
    同一の純関数が行う。
    """
    rows = RowAlignment(t0=200, n_samples=999_800)
    budget = bounded_chunk_size(10**9, rows.n_samples)
    for configured in (1, 7, 16, 256, 20_000):
        assert rows.solve_width(configured) == bounded_chunk_size(
            configured, rows.n_samples
        )
    for n_columns in (1, 4, 16, 4096):
        assert rows.block_width(n_columns) == bounded_chunk_size(
            n_columns, rows.n_samples
        )
    # 同じ入力なら同じ値になる (= 予算が1本である) ことを直接示す。
    for width in (1, 16, 1000):
        assert rows.solve_width(width) == rows.block_width(width)
    assert budget == 16, f"128 MiB 予算の実効列数が変わりました: {budget}"


def test_block_width_is_capped_by_the_memory_budget() -> None:
    """確保軸も 128 MiB 予算で切り詰まる (D-33、キャップを外す変異で赤)。

    ``block_width`` は ``cfg.chunk_size`` を読まないが、**予算を読まない**
    わけではない。代表目標が大量にある設定 (``n_surrogate_targets`` に上限は
    無い) で予算を無視すると、F-03-2-015 の一括確保 (実測 peak RSS 3.23GB)
    がそのまま復活する。
    """
    rows = RowAlignment(t0=200, n_samples=999_800)
    assert rows.block_width(400) == 16, "128 MiB 予算で切り詰まっていません"
    # 予算より少ない列数はそのまま (必要以上に確保も分割もしない)。
    assert rows.block_width(4) == 4
    assert rows.block_width(1) == 1
    # 小さい T_eff では予算が効かない (切り詰めが常時発動する実装ではない)。
    small = RowAlignment(t0=50, n_samples=1_950)
    assert small.block_width(400) == 400


def test_iter_delay_chunks_matches_expected_offset() -> None:
    """MC が実際に使う ``_iter_delay_chunks`` の出力値そのものを固定する。

    ``CapacityProblem.lagged`` 単体の正しさを固定するだけでは、
    ``_iter_delay_chunks`` (``memory_capacity`` が実際に呼ぶ関数) 側で
    ``lagged`` に渡す ``delay`` を取り違えるミス (例: ``delay - 1``) までは
    検出できない。``lagged`` に触れず ``_iter_delay_chunks`` を丸ごと
    差し替える形の変異 (関数全体の monkeypatch) でこの穴を実際に確認した:
    ``lagged`` 自身のガードは無傷のまま、``psi[t0-delay+1 : ...]`` 相当へ
    ずらしても既存22テストは全て緑のまま通った。ここで実際に MC が呼ぶ
    ``_iter_delay_chunks`` の出力を直接検査することで、その穴を閉じる。
    """
    n_steps = 500
    ramp: FloatArray = np.arange(n_steps, dtype=np.float64)
    t0 = 42
    n_samples = n_steps - t0
    problem = CapacityProblem.from_states(np.zeros((n_steps, 3)), t0=t0)
    delays = (1, 5, 20)
    chunks = list(_iter_delay_chunks(problem, ramp, delays, chunk_size=2))
    columns: FloatArray = np.concatenate(chunks, axis=1)
    assert columns.shape == (n_samples, len(delays))
    for index, delay in enumerate(delays):
        expected: FloatArray = np.arange(
            t0 - delay, t0 - delay + n_samples, dtype=np.float64
        )
        np.testing.assert_array_equal(columns[:, index], expected)


def test_memory_capacity_requires_single_channel_input() -> None:
    """``u`` が無い / 多変数だと ValueError (安く緑にする逃げ道を塞ぐ)。"""
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    ctx = DiagnosticContext(seed=CTX_SEED)
    cfg = MemoryCapacityConfig(max_delay=20, n_surrogates=5)
    with pytest.raises(ValueError, match="入力系列 u"):
        memory_capacity(states, None, ctx=ctx, cfg=cfg)
    with pytest.raises(ValueError, match="1変数入力"):
        memory_capacity(states, np.repeat(inputs, 2, axis=1), ctx=ctx, cfg=cfg)


def test_series_shorter_than_max_delay_raises_instead_of_truncating() -> None:
    """``max_delay`` が系列長を超えたら黙って切り詰めず ValueError。"""
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    with pytest.raises(ValueError, match="系列が短すぎます"):
        memory_capacity(
            states,
            inputs,
            ctx=DiagnosticContext(seed=CTX_SEED),
            cfg=MemoryCapacityConfig(max_delay=2000),
        )


def test_unknown_threshold_mode_raises() -> None:
    """未知の ``threshold_mode`` は既定へフォールバックせず ValueError。"""
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    with pytest.raises(ValueError, match="threshold_mode"):
        memory_capacity(
            states,
            inputs,
            ctx=DiagnosticContext(seed=CTX_SEED),
            cfg=MemoryCapacityConfig(max_delay=20, threshold_mode="bonferroni"),
        )


def test_surrogate_threshold_requires_ctx_seed_and_is_reproducible() -> None:
    """サロゲート閾値は ``ctx.seed`` を必須にし、同じ seed で再現する (D-27)。

    閾値が黙って非再現になると、しきい値法の比較 (受け入れ条件3) の記録が
    意味を失う。
    """
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    cfg = MemoryCapacityConfig(max_delay=40, n_surrogates=25, chunk_size=8)
    with pytest.raises(ValueError, match=r"ctx\.seed"):
        memory_capacity(states, inputs, ctx=DiagnosticContext(), cfg=cfg)

    first = memory_capacity(
        states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=cfg
    )
    same = memory_capacity(
        states, inputs, ctx=DiagnosticContext(seed=CTX_SEED), cfg=cfg
    )
    other = memory_capacity(
        states, inputs, ctx=DiagnosticContext(seed=CTX_SEED + 1), cfg=cfg
    )
    assert first.scalars["mc_threshold"] == same.scalars["mc_threshold"]
    assert first.scalars["mc_threshold"] != other.scalars["mc_threshold"]

    none_mode = memory_capacity(
        states,
        inputs,
        ctx=DiagnosticContext(),
        cfg=dataclasses.replace(cfg, threshold_mode=THRESHOLD_NONE),
    )
    assert none_mode.scalars["mc_total"] == pytest.approx(
        none_mode.scalars["mc_total_raw"]
    )
    assert first.scalars["mc_total"] < first.scalars["mc_total_raw"], (
        "サロゲート閾値が1本も落としていません (閾値が効いていない疑い)"
    )
    assert first.params["threshold_mode"] == THRESHOLD_SURROGATE


# --------------------------------------------------------------------------
# 受け入れ基準6: 外部生成の状態系列で完走する
# --------------------------------------------------------------------------


def test_diagnostic_accepts_arbitrary_external_state_series() -> None:
    """リザバーですらない任意の X (独立な乱数) でも完走する (受け入れ条件6)。

    入力と無関係な状態なので容量はほぼ 0 になるはずで、サロゲート閾値が
    効いていれば ``mc_total`` は生の総容量よりずっと小さくなる。
    """
    rng = np.random.default_rng(2718)
    states: FloatArray = rng.standard_normal((1500, 12))
    inputs: FloatArray = rng.uniform(-1.0, 1.0, size=(1500, 1))
    result = memory_capacity(
        states,
        inputs,
        ctx=DiagnosticContext(washout=10, seed=CTX_SEED),
        cfg=MemoryCapacityConfig(max_delay=50, n_surrogates=50, chunk_size=16),
    )
    scalars = _scalars(result)
    assert all(np.isfinite(value) for value in scalars.values())
    assert scalars["mc_total"] < 0.05 * scalars["mc_total_raw"] + 0.5, (
        "独立な乱数状態に対して容量が残っています: "
        f"total={scalars['mc_total']}, raw={scalars['mc_total_raw']}"
    )


# --------------------------------------------------------------------------
# 容量カーネルそのもの (D-26 の構造)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _CountingMatrix:
    """``@`` の回数を数えるために ``CapacityProblem.x`` へ差し込むプロキシ。

    ``capacity_of_targets`` が使う配列の機能 (``shape`` / ``.T`` / ``@``)
    だけを持つ。``X`` を2回走査する実装 (例: 予測 ``Phi @ W`` を実体化して
    残差を取る) に変えると呼び出し回数が増える。
    """

    array: FloatArray
    calls: list[str]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def T(self) -> _CountingMatrix:
        return _CountingMatrix(self.array.T, self.calls)

    def __matmul__(self, other: FloatArray) -> FloatArray:
        self.calls.append("matmul")
        product: FloatArray = self.array @ other
        return product


def test_capacity_of_targets_touches_x_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容量は Gram 量だけで閉じ、``X`` を1回しか走査しない (D-26 / F-03-1-013)。

    ``rhs = [sum(Z,0); X.T @ Z]`` のうち ``X`` に触れるのは ``X.T @ Z`` の
    **1回だけ** (バイアス行は ``Z`` の縮約だけで作るので ``X`` には触れない)。
    予測 ``Phi @ W`` を作って ``Z - Phi W`` から残差を取る素直な実装だと2回に
    なり、目標数 K に比例する ``O(T F K)`` の走査がもう1本増える (IPC の
    20万目標で効く)。``fit_ridge_from_gram`` の呼び出しもチャンクあたり1回に
    固定する。

    ``CapacityProblem`` は F-03-1-013 で ``Phi`` を実体化しなくなり、状態
    ``X`` のビューだけを持つようになった。旧テストは ``problem.phi`` を
    プロキシに差し替えていたが、``phi`` が無くなったのでこの guard の意図
    (「触れるのは1回だけ」) を保ったまま ``problem.x`` を差し替える形に
    書き換えてある。
    """
    rng = np.random.default_rng(99)
    states: FloatArray = rng.standard_normal((400, 6))
    problem = CapacityProblem.from_states(states, t0=10)
    targets: FloatArray = rng.standard_normal((problem.n_samples, 12))

    solves: list[str] = []
    original = fit_ridge_from_gram

    def counting_solve(
        gram: FloatArray, rhs: FloatArray, alpha: float, *, bias_column: int | None
    ) -> FloatArray:
        solves.append("solve")
        return original(gram, rhs, alpha, bias_column=bias_column)

    monkeypatch.setattr(capacity_module, "fit_ridge_from_gram", counting_solve)

    calls: list[str] = []
    proxied = dataclasses.replace(
        problem, x=cast(FloatArray, _CountingMatrix(problem.x, calls))
    )
    capacities = capacity_of_targets(proxied, targets, 1.0e-9)

    assert calls == ["matmul"], f"X の走査回数が1回ではありません: {len(calls)}"
    assert solves == ["solve"], f"solve の回数が1回ではありません: {len(solves)}"
    np.testing.assert_allclose(
        capacities, capacity_of_targets(problem, targets, 1.0e-9)
    )


def test_capacity_matches_direct_least_squares_residual() -> None:
    """Gram 展開の容量が、素直な残差計算と一致する (代数の裏取り)。

    ``C = 1 - ||z - Phi w||^2 / ||z||^2`` を実際に予測を作って計算した値と
    突き合わせる。Gram 展開の符号ミス (``-2 w.T rhs`` を ``+2`` にする等) は
    値が「それらしく」出るためレビューでは気づけない。``Phi`` は本体では
    F-03-1-013 により実体化しなくなったため、ここでは検証専用に
    ``problem.x`` からその場で組み立てる (本体の経路には影響しない)。
    """
    rng = np.random.default_rng(1234)
    states: FloatArray = rng.standard_normal((600, 8))
    problem = CapacityProblem.from_states(states, t0=5)
    targets: FloatArray = rng.standard_normal((problem.n_samples, 4))
    alpha = 1.0e-8

    capacities = capacity_of_targets(problem, targets, alpha)
    phi: FloatArray = np.concatenate(
        (np.ones((problem.n_samples, 1), dtype=np.float64), problem.x), axis=1
    )
    weights = fit_ridge_from_gram(
        problem.gram, phi.T @ targets, alpha, bias_column=problem.bias_column
    )
    residual: FloatArray = targets - phi @ weights
    direct: FloatArray = 1.0 - np.sum(residual**2, axis=0) / np.sum(targets**2, axis=0)
    np.testing.assert_allclose(capacities, direct, rtol=1.0e-8, atol=1.0e-10)


def test_capacity_problem_rejects_short_series() -> None:
    """行数が特徴数以下なら ValueError (見かけ上「完璧な」容量が出るのを防ぐ)。"""
    rng = np.random.default_rng(5)
    states: FloatArray = rng.standard_normal((20, 30))
    with pytest.raises(ValueError, match="特徴数以下"):
        CapacityProblem.from_states(states, t0=0)


def test_capacity_problem_x_is_read_only() -> None:
    """``problem.x`` への書き込みは ``ValueError`` (F-03-3-006)。

    ``x`` は元の ``X`` のビューで、``gram`` は構築時点の ``X`` から作った
    スナップショット。``x`` が書き込み可能なままだと、診断内部や呼び出し側の
    コードが ``problem.x[...] = ...`` と書いても素通りし、``gram`` と無言で
    desync する (F-03-2-003 の契約)。``from_states`` が保持するビュー自身を
    読み取り専用にすることで、この経路の半分 (呼び出し側が渡す前の ``X`` 自身
    への書き込みは 3b の受け入れ条件) を閉じる。
    """
    rng = np.random.default_rng(9)
    states: FloatArray = rng.standard_normal((300, 5))
    problem = CapacityProblem.from_states(states, t0=10)
    with pytest.raises(ValueError, match="read-only"):
        problem.x[0, 0] = 1.0


def test_capacity_of_targets_rejects_mismatched_rows() -> None:
    """目標の行数が設計行列と違えば ValueError (D-24 の行合わせ)。"""
    rng = np.random.default_rng(6)
    states: FloatArray = rng.standard_normal((300, 5))
    problem = CapacityProblem.from_states(states, t0=10)
    bad: FloatArray = rng.standard_normal((problem.n_samples - 1, 3))
    with pytest.raises(ValueError, match="D-24"):
        capacity_of_targets(problem, bad, 1.0e-9)


def _chunk_counts(sizes: Sequence[int], chunk_size: int) -> int:
    return sum(-(-size // chunk_size) for size in sizes)


def test_solve_count_is_driven_by_chunks_not_by_target_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """solve の回数はチャンク数で決まり、遅延本数には比例しない (D-26)。

    遅延を 4 倍にしても、``chunk_size`` が同じなら solve はチャンク数ぶんしか
    増えない。「(delay, degree) ごとに回帰し直す」構造に戻ると、この関係が
    崩れて回数が目標数に比例する。
    """
    states, inputs = _cached_states(0.9, 15, 2000, 5)
    ctx = DiagnosticContext(washout=50, seed=CTX_SEED)
    original = fit_ridge_from_gram

    for max_delay in (40, 160):
        solves: list[str] = []

        def counting_solve(
            gram: FloatArray,
            rhs: FloatArray,
            alpha: float,
            *,
            bias_column: int | None,
            _sink: list[str] = solves,
        ) -> FloatArray:
            _sink.append("solve")
            return original(gram, rhs, alpha, bias_column=bias_column)

        monkeypatch.setattr(capacity_module, "fit_ridge_from_gram", counting_solve)
        cfg = MemoryCapacityConfig(max_delay=max_delay, n_surrogates=20, chunk_size=16)
        memory_capacity(states, inputs, ctx=ctx, cfg=cfg)
        # 遅延 max_delay 本 + サロゲート 20 本を chunk_size=16 で切った枚数。
        assert len(solves) == _chunk_counts((max_delay, 20), 16)


# --------------------------------------------------------------------------
# bounded_chunk_size 単体 (F-03-2-018: BLOCKER の実体的な安全機構)
# --------------------------------------------------------------------------


def test_bounded_chunk_size_keeps_configured_when_under_budget() -> None:
    """configured が予算内なら変更しない (小さい chunk_size を明示指定した
    呼び出し側の意図を尊重する)。"""
    assert bounded_chunk_size(100, 4_000) == 100
    assert bounded_chunk_size(1, 999_600) == 1


def test_bounded_chunk_size_truncates_when_over_budget() -> None:
    """configured が予算を超えていれば切り詰める (BLOCKER の実体)。

    T=1e6 級の本番規模で configured=256 のまま (無条件) だと1チャンクが
    2.05GB に達する (_MAX_CHUNK_BYTES のモジュール docstring 参照)。
    """
    assert bounded_chunk_size(100_000, 2_000_000) == 8
    assert bounded_chunk_size(256, 999_600) == 16


def test_bounded_chunk_size_defends_against_non_positive_n_samples() -> None:
    """``n_samples <= 0`` はバイト数を計算できないため configured をそのまま
    返す (防御的分岐、coverage で Missing だった行)。"""
    assert bounded_chunk_size(256, 0) == 256
    assert bounded_chunk_size(256, -5) == 256


def test_bounded_chunk_size_never_returns_less_than_one() -> None:
    """``n_samples`` がどれだけ大きくても下限は 1 (0 列のチャンクは作らない)。"""
    assert bounded_chunk_size(500, 1_000_000_000) == 1
    assert bounded_chunk_size(1, 1_000_000_000) == 1


def test_capacity_problem_effective_chunk_size_delegates_to_bounded_chunk_size() -> (
    None
):
    """``CapacityProblem.effective_chunk_size`` は自身の ``n_samples`` で
    ``bounded_chunk_size`` を呼ぶだけ (F-03-2-001)。

    呼び出し側 (``ipc.py`` / ``memory_capacity.py``) が ``n_samples`` を
    取り出して個別に ``bounded_chunk_size`` を呼ぶ形の複製をこのメソッドが
    肩代わりすることを、直接の委譲として固定する。
    """
    rng = np.random.default_rng(42)
    states: FloatArray = rng.standard_normal((5_000, 4))
    problem = CapacityProblem.from_states(states, t0=10)
    for configured in (1, 256, 100_000):
        assert problem.effective_chunk_size(configured) == bounded_chunk_size(
            configured, problem.n_samples
        )


def test_capacity_problem_lagged_rejects_multi_dimensional_series() -> None:
    """``series`` が1次元でなければ ``ValueError`` (F-03-2-020、coverage で
    Missing だった分岐)。"""
    rng = np.random.default_rng(7)
    states: FloatArray = rng.standard_normal((300, 5))
    problem = CapacityProblem.from_states(states, t0=10)
    series_2d: FloatArray = rng.standard_normal((300, 2))
    with pytest.raises(ValueError, match="1次元"):
        problem.lagged(series_2d, 1)
