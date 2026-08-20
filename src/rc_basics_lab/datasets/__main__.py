"""``python -m rc_basics_lab.datasets`` の入口 (``make data-05``).

本体は ``datasets/cli.py`` にある。ここを分けているのは、``make data-05`` の
コマンド (``python -m rc_basics_lab.datasets``) を変えないためである。

以前は「``__init__`` が ``cli`` を再エクスポートしているので
``python -m rc_basics_lab.datasets.cli`` だと ``RuntimeWarning: found in
sys.modules ...`` が出る」ことが分けている理由だったが、D-72 で ``cli`` を
``__init__`` から外したため、その事情は無くなった (``cli`` を直接 ``-m`` で
呼んでも警告は出ない)。この入口自体は ``make data-05`` の互換のために残す。
"""

from __future__ import annotations

from rc_basics_lab.datasets.cli import main

raise SystemExit(main())
