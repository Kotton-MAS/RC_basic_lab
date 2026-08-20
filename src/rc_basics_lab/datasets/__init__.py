"""データ層 —— 外部データセットの取得・照合・読み取り (D-58 / D-59 / D-60).

**ネットワークとファイル I/O を持つ唯一のパッケージ**である。課題層
(``tasks/``) と指標層 (``metrics_detection.py``) は純関数のまま保ち、
依存の向きは ``datasets -> tasks`` の一方向にする (D-59)。逆向きの辺を1本でも
引くと、課題層がステートフルな I/O 層に化けて移植性 (D-12 が守る性質) が
失われる。

- ``fetch``: HTTPS 限定のダウンロード・SHA256 照合・キャッシュ・ZIP 展開
- ``manifest``: ``manifests/*.csv`` (出典 URL / ライセンス / SHA256) の読み取り
- ``mgab``: MGAB (CC0-1.0)。既定の実データ源
- ``ucr``: UCR Anomaly Archive (**ライセンス未指定**。本体は同梱しない)

**``cli`` はここに載せない** (D-72)。``cli.py`` は
``from rc_basics_lab.datasets import mgab, ucr`` でこのパッケージへ戻るので、
``__init__`` が ``cli`` を import すると ``__init__ -> cli -> __init__`` の辺が
できる。現状は submodule import 機構のおかげで動いているだけで、``cli`` が
``__init__`` の**再エクスポートを1つでも使い始めた瞬間に ``ImportError``**
になる。CLI は ``python -m rc_basics_lab.datasets``
(``__main__.py``) か ``rc_basics_lab.datasets.cli`` から直接呼ぶ。

データ本体はリポジトリに入れない (D-58)。``data/`` は ``.gitignore`` 済みで、
**pytest はキャッシュが無ければ skip する** (D-60)。
"""

from rc_basics_lab.datasets import fetch, manifest, mgab, ucr

__all__ = [
    "fetch",
    "manifest",
    "mgab",
    "ucr",
]
