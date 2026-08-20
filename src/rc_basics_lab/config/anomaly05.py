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

**Mackey-Glass の生成パラメータをここで再定義しない** (04 の ``chaos04.py``
と同じ規律)。単一の真実は 01 の ``MackeyGlassConfig`` で、合成源はその値を
``dataclasses.replace`` で長さだけ差し替えて使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rc_basics_lab.config.experiment01 import MackeyGlassConfig


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
        mackey_glass: 土台の Mackey-Glass 生成パラメータ。``length`` は
            合成源が「除去するぶんを足した長さ」へ差し替えるので、ここに書いた
            ``length`` は**使われない** (単一の真実は ``SyntheticAnomalyConfig.
            length`` 側)。
    """

    length: int = 20000
    n_anomalies: int = 5
    segment_length: int = 200
    ignore_margin: int = 50
    mackey_glass: MackeyGlassConfig = field(default_factory=MackeyGlassConfig)


__all__ = ["SyntheticAnomalyConfig"]
