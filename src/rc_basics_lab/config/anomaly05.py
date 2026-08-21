"""実験05 (センサー時系列の異常検知) の設定 dataclass 群 (D-13).

T2 が合成源の設定 (``SyntheticAnomalyConfig``) を、T3 が実験1本ぶんの
``Anomaly05Config`` を置いた。**非空 300 行の上限**
(``tests/test_config_package_layout.py``) まで余裕が少ないので、5-C / 5-D の
掃引設定を足す T4 はこのモジュールを割ることになる。

``SyntheticAnomalyConfig`` を ``tasks/anomaly.py`` 側に置かなかったのは
import の向きのためである: 既存の課題層は ``tasks -> config``
(``tasks/mackey_glass.py:21`` など5モジュール) の一方向で、逆向きの辺
(``config -> tasks``) を1本でも引くと ``config/__init__`` の初期化中に
``tasks`` が ``rc_basics_lab.config`` を import し直す循環になる。
``diagnostics`` / ``reservoir`` が自分の設定 dataclass を持てるのは、
その2つが ``config`` を import しないからである (D-12)。

**Mackey-Glass の生成パラメータの既定値をここで書き直さない** (04 の
``chaos04.py`` と同じ規律)。既定値の単一の真実は 01 の ``MackeyGlassConfig``
のままで、``SyntheticMackeyGlassConfig`` は**葉の集合だけを絞った**器である
(既定値は ``_MACKEY_GLASS_DEFAULTS`` = ``MackeyGlassConfig()`` から引く。
D-69)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.config.anomaly05_sweep import (
    AnomalyProtocolSweepConfig,
    AnomalySizeSweepConfig,
)
from rc_basics_lab.config.experiment01 import DEFAULT_ALPHA_GRID, MackeyGlassConfig
from rc_basics_lab.reservoir.esn import ESNConfig
from rc_basics_lab.seeds import SeedStream

_MACKEY_GLASS_DEFAULTS = MackeyGlassConfig()
"""既定値の単一の真実 (01 の ``MackeyGlassConfig``)。

``SyntheticMackeyGlassConfig`` の7葉の既定はすべてここから引く。数値を
リテラルで書き写すと 01 と 05 で tau が食い違っても誰も気づかない —— 絞るのは
**葉の集合**であって既定値ではない (D-69)。
"""

_SYNTHETIC_HORIZON = 1
"""合成源が ``generate_mackey_glass`` に渡す ``horizon`` (固定)。

合成源は生成された系列の ``u`` 列しか使わない (1ステップ先の対 ``y`` は捨てる)
ので、``horizon`` は最小値で足りる。設定軸にすると「変えても出力が1バイトも
変わらない死んだ葉」になる (D-69)。
"""


@dataclass(frozen=True, slots=True)
class SyntheticMackeyGlassConfig:
    """合成源が使う MG 生成パラメータ (**``length`` / ``horizon`` を持たない**).

    **絞った器で受けることで、この2葉は構造的に存在しなくなる** (D-69)。

    既定値は ``_MACKEY_GLASS_DEFAULTS`` (= 01 の ``MackeyGlassConfig()``) から
    引くので、01 の既定を変えれば 05 の合成源も追随する。

    Attributes:
        tau: 遅延時間。``tau / rk4_step`` が整数でない設定は生成側で
            ``ValueError`` になる (``tasks/mackey_glass.py`` の ``delay_steps``)。
        beta: 生成項の係数。
        gamma: 減衰項の係数。
        exponent: 生成項の非線形指数 n。
        rk4_step: RK4 の刻み h。
        sample_interval: サブサンプル間隔 (積分ステップ数)。
        integration_burn_in: 捨てるサンプル数 (初期履歴の記憶を落とす)。
    """

    tau: float = _MACKEY_GLASS_DEFAULTS.tau
    beta: float = _MACKEY_GLASS_DEFAULTS.beta
    gamma: float = _MACKEY_GLASS_DEFAULTS.gamma
    exponent: int = _MACKEY_GLASS_DEFAULTS.exponent
    rk4_step: float = _MACKEY_GLASS_DEFAULTS.rk4_step
    sample_interval: int = _MACKEY_GLASS_DEFAULTS.sample_interval
    integration_burn_in: int = _MACKEY_GLASS_DEFAULTS.integration_burn_in

    def to_mackey_glass(self, *, length: int) -> MackeyGlassConfig:
        """01 の ``MackeyGlassConfig`` を組み立てる**唯一の口** (D-70)。

        Args:
            length: 生成を頼むサンプル数 (合成源が「除去するぶんを足した長さ」
                として計算した値)。

        Returns:
            7葉 + ``length`` + ``horizon=_SYNTHETIC_HORIZON`` の生成パラメータ。
        """
        return MackeyGlassConfig(
            tau=self.tau,
            beta=self.beta,
            gamma=self.gamma,
            exponent=self.exponent,
            rk4_step=self.rk4_step,
            sample_interval=self.sample_interval,
            integration_burn_in=self.integration_burn_in,
            length=length,
            horizon=_SYNTHETIC_HORIZON,
        )


@dataclass(frozen=True, slots=True)
class SyntheticAnomalyConfig:
    """合成異常源 (MGAB と同じ手続き) の設定。純データ。値域検証は使う側 (D-09)。

    MGAB (Thill et al., CC0-1.0) は Mackey-Glass 系列から「値と微分が一致する
    2点」で挟まれたセグメントを取り除いて縫合し、**縫合点の近傍を異常として
    標識する**。値も微分も連続なので、目視でも単純な閾値でも見つからない異常に
    なる。ここはその手続きをそのまま合成源として持つ (D-60: 既定データ源は
    合成にし、pytest をネットワークから切り離す)。

    Attributes:
        length: 合成後の系列長 T。
        n_anomalies: 挿入する異常の個数。実測の MGAB は 100,000 点に 10 個。
        segment_length: 縫合点を中心に**異常として標識する幅** [点]。
            実測の MGAB は 401 点 (異常率 0.0401)。除去するセグメントの長さは
            この値から派生させる (``_REMOVED_SPAN_FACTORS``) —— 独立した設定軸に
            すると「標識幅は変えたが除去幅は変えていない」条件が作れてしまい、
            異常の強さと評価の甘さが同じ軸で動かなくなる。
        ignore_margin: 異常区間の**前後**で評価から除外する点数 (D-58 の
            MGAB 実測メモにあるとおり、縫合の近傍は正常とも異常とも言えない)。
        mackey_glass: 土台の Mackey-Glass 生成パラメータ。系列長は
            ``SyntheticAnomalyConfig.length`` 側が単一の真実なので、この器は
            ``length`` / ``horizon`` を**持たない** (D-69)。
    """

    length: int = 20000
    n_anomalies: int = 5
    segment_length: int = 200
    ignore_margin: int = 50
    mackey_glass: SyntheticMackeyGlassConfig = field(
        default_factory=SyntheticMackeyGlassConfig
    )


@dataclass(frozen=True, slots=True)
class AnomalyDatasetConfig:
    """系列源の選択と、系列1本の使い方 (5-A の行の素)。

    Attributes:
        source: 源の識別子。実体への対応づけは実験層の ``build_sources``
            1箇所だけが持つ (源の具象名で分岐してよい唯一の場所、D-71)。
        series: 使う系列名。合成源では系列を区別する札、MGAB では系列番号、
            UCR ではファイル名。**行数がこの長さで決まる**。
        max_length: 系列の打ち切り長 [点]。予算調整の第一の軸 (仕様 §3)。
        train_ratio: 学習に使う割合。学習区間は「異常が1点も無いことが
            保証された前半」(``AnomalySeries.train_end``) の内側でなければ
            ならず、実験層が突き合わせて検査する。
        calibration_ratio: 運用閾値を決める区間の割合 (D-56)。残りがテスト。
            **較正区間のラベルは1ビットも読まない**。
    """

    source: str = "synthetic"
    series: tuple[str, ...] = ("s1", "s2", "s3")
    max_length: int = 20000
    train_ratio: float = 0.25
    calibration_ratio: float = 0.15


@dataclass(frozen=True, slots=True)
class AnomalyPreprocessConfig:
    """全手法が共有する前処理とスコア整形 (D-57)。

    Attributes:
        normalize: ``AnomalyPreprocessor`` の方式 (``NORMALIZE_METHODS`` の4値)。
        standardize_steps: 係数を推定する先頭行数。**学習区間の内側**である
            ことを実験層が検査する (テスト区間から推定させない、D-57)。
        input_window: 遅延線の ``n_lags`` と移動統計の窓幅。両者を1軸に
            まとめてあるので、「入力が何点届くか」が手法間でそろう。
        score_smoothing: 異常スコアの後方移動平均の窓 [点]。``1`` で平滑化
            なし。**全手法に同じ窓を掛ける** (片方だけ平滑化された比較を
            作れないようにするため)。
    """

    normalize: str = "zscore"
    standardize_steps: int = 3000
    input_window: int = 16
    score_smoothing: int = 8


@dataclass(frozen=True, slots=True)
class AnomalyReservoirConfig:
    """05 のリザバー構造と実行の粒度 (D-08: 検証分割で調整しない)。

    01 の ``ESNConfig`` を内包しない —— ``bias_scale`` / ``activation`` /
    ``state_noise`` は 05 の軸ではなく、内包すると死葉が3つ増える (D-69)。

    Attributes:
        n_units: リザバーのユニット数 N (5-D の掃引軸)。
        spectral_radius: 再帰行列のスペクトル半径。
        leak_rate: 漏れ率。
        input_scale: 入力重みの幅。
        density: 再帰行列の非零率。
        washout: 初期過渡として全手法で捨てる行数 (``compute_t0``、D-05)。
        n_replicates: 1系列あたりのレプリケート数。予算超過時に落としてよい
            唯一の値 (仕様 §3)。
    """

    n_units: int = 200
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scale: float = 0.5
    density: float = 0.1
    washout: int = 200
    n_replicates: int = 5

    def to_esn(self) -> ESNConfig:
        """``ESNConfig`` を組み立てる**唯一の口** (``to_mackey_glass`` と同型)。"""
        return ESNConfig(
            n_units=self.n_units,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            input_scale=self.input_scale,
            density=self.density,
        )


@dataclass(frozen=True, slots=True)
class AnomalyRidgeConfig:
    """残差系スコアのリッジ回帰 (D-04: 全手法が同一格子を読む)。

    01 の ``RidgeConfig`` を内包しないのは ``n_lags_grid`` のため —— 05 の
    遅延線は ``preprocess.input_window`` で決まるので死葉になる (D-69)。

    Attributes:
        alpha_grid: 全手法・全系列が共有する単一の探索格子 (D-04)。
    """

    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID


@dataclass(frozen=True, slots=True)
class AnomalyThresholdConfig:
    """運用閾値の決め方と、テスト側最適化の**参考値** (D-56)。

    Attributes:
        target_false_alarm_rate: 較正区間で許す警報率。閾値はこの分位点で
            決まり、テスト区間では固定する。
        report_test_optimal: ``f1_test_optimal`` 列を出すか。**別列**であって
            ``f1_calibrated`` を置き換えることはない (D-56)。
        sweep_points: 5-B の閾値掃引の点数 (``anomaly_threshold.csv`` の
            1条件あたりの行数)。
    """

    target_false_alarm_rate: float = 0.01
    report_test_optimal: bool = True
    sweep_points: int = 21


@dataclass(frozen=True, slots=True)
class AnomalyEvaluationConfig:
    """評価の作法 (D-55 の point-adjust と ignore マスク)。

    Attributes:
        report_point_adjust: PA%K を報告するか。``True`` のとき ``pa_f1_k*``
            と ``pa_f1_random_k*`` が**必ず対で**現れる (D-55)。
        pa_k_grid: PA%K の K [%]。``0`` が従来の point-adjust。
        ignore_transition: ``AnomalySeries.ignore`` の点を点単位指標から
            落とすか (PA 系はマスク前の系列で計算する)。
    """

    report_point_adjust: bool = True
    pa_k_grid: tuple[float, ...] = (0.0, 20.0)
    ignore_transition: bool = True


@dataclass(frozen=True, slots=True)
class AnomalySeedConfig:
    """05 が使う4ストリームの基底シード (D-06 / D-14)。

    01 の ``SeedConfig`` とは別クラスにする (D-13)。``control`` は
    ``EspSeedConfig.drive`` が ``SeedStream.TASK`` へ載るのと同じ流儀で
    **``SeedStream.PROBE`` へ載せる** —— 05 は初期状態プローブを使わないので
    空いており、5本目のストリームを足さずに対照だけを独立に振れる。

    Attributes:
        reservoir: リザバー重み (``SeedStream.RESERVOIR``)。
        task: 系列生成 (``SeedStream.TASK``)。
        split: 分割オフセット (``SeedStream.SPLIT``)。
        control: 一様乱数対照 (``SeedStream.PROBE``)。**これを変えたときに
            動くのは対照の行だけ**であることを配線テストが実測する (D-61)。
    """

    reservoir: int = 0
    task: int = 1
    split: int = 2
    control: int = 5


def anomaly_stream_seed(seeds: AnomalySeedConfig, stream: SeedStream) -> int:
    """05 の設定からストリームの基底シードを取り出す (``esp_stream_seed`` と同型)。

    他ストリームのシードを一切参照しないことが独立性の根拠なので、
    ``getattr`` ではなく明示的な分岐で書く。
    """
    match stream:
        case SeedStream.RESERVOIR:
            return seeds.reservoir
        case SeedStream.TASK:
            return seeds.task
        case SeedStream.SPLIT:
            return seeds.split
        case SeedStream.PROBE:
            return seeds.control


@dataclass(frozen=True, slots=True)
class Anomaly05Config:
    """実験05 (5-A / 5-B / 5-C / 5-D) 1本ぶんの設定 (D-13)。

    ``ExperimentConfig`` には1フィールドも足さない。5-C (プロトコル掃引) と
    5-D (N 掃引) の格子は T4 が掃引の実装 (``experiment/anomaly_sweep.py``)
    と**同時に**足した —— 掃引を回す実装が無い時点で置くと、その葉は
    「値を変えても出力が1バイトも変わらない」状態で全葉被覆テストに載り、
    D-69 が ``length`` / ``horizon`` で潰したのと同じ死葉になる。

    Attributes:
        name: 実験名 (``meta.json`` に出るだけで結果行は変えない)。
        dataset: 系列源と分割の割合。
        synthetic: 合成源の設定 (``dataset.source == "synthetic"`` のときだけ
            読まれる)。被覆は ``test_each_synthetic_leaf_changes_the_generated_series``
            へ委譲する (D-69)。
        preprocess: 全手法共通の前処理とスコア整形 (D-57)。
        reservoir: リザバー構造とレプリケート数。
        ridge: 残差系スコアのリッジ格子 (D-04)。
        threshold: 運用閾値と参考値 (D-56)。
        evaluation: PA%K と ignore マスク (D-55)。
        protocol_sweep: 5-C の格子。既定の格子は ``preprocess`` の既定値を
            含む (含まない格子は ``run_protocol_sweep`` が ``ValueError``、
            D-79)。
        size_sweep: 5-D の格子。既定の格子は ``reservoir.n_units`` を含む。
        seeds: 4ストリームの基底シード。
    """

    name: str = "05_anomaly_detection"
    dataset: AnomalyDatasetConfig = field(default_factory=AnomalyDatasetConfig)
    synthetic: SyntheticAnomalyConfig = field(default_factory=SyntheticAnomalyConfig)
    preprocess: AnomalyPreprocessConfig = field(default_factory=AnomalyPreprocessConfig)
    reservoir: AnomalyReservoirConfig = field(default_factory=AnomalyReservoirConfig)
    ridge: AnomalyRidgeConfig = field(default_factory=AnomalyRidgeConfig)
    threshold: AnomalyThresholdConfig = field(default_factory=AnomalyThresholdConfig)
    evaluation: AnomalyEvaluationConfig = field(default_factory=AnomalyEvaluationConfig)
    protocol_sweep: AnomalyProtocolSweepConfig = field(
        default_factory=AnomalyProtocolSweepConfig
    )
    size_sweep: AnomalySizeSweepConfig = field(default_factory=AnomalySizeSweepConfig)
    seeds: AnomalySeedConfig = field(default_factory=AnomalySeedConfig)


__all__ = [
    "Anomaly05Config",
    "AnomalyDatasetConfig",
    "AnomalyEvaluationConfig",
    "AnomalyPreprocessConfig",
    "AnomalyProtocolSweepConfig",
    "AnomalyReservoirConfig",
    "AnomalyRidgeConfig",
    "AnomalySeedConfig",
    "AnomalySizeSweepConfig",
    "AnomalyThresholdConfig",
    "SyntheticAnomalyConfig",
    "SyntheticMackeyGlassConfig",
    "anomaly_stream_seed",
]
