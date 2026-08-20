"""実験05 (センサー時系列の異常検知) の設定 dataclass 群 (D-13).

T2 (データ層) が必要とする**合成源の設定だけ**をここに置く。実験1本ぶんの
``Anomaly05Config`` は T3 が同じモジュールへ足す。

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

from rc_basics_lab.config.experiment01 import MackeyGlassConfig

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

    01 の ``MackeyGlassConfig`` をそのまま内包していたときは、``length`` と
    ``horizon`` の2葉が ``generate_synthetic_anomalies`` に必ず上書きされる
    **死んだ設定**だった (YAML から設定できるのに出力が1バイトも変わらない)。
    絞った器で受けることで、この2葉は構造的に存在しなくなる (D-69)。

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

        ``tasks/chaotic.py`` の ``Standardizer.from_training_prefix`` (D-41) と
        同じ流儀で、**作れる場所を1本に閉じる**。呼び出し側が自前で
        ``MackeyGlassConfig`` を組み立てられると、そこで ``horizon`` や他の葉を
        黙って別の値にする経路が復活する。

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


__all__ = ["SyntheticAnomalyConfig", "SyntheticMackeyGlassConfig"]
