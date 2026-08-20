"""``python -m rc_basics_lab.datasets`` の入口 (``make data-05``).

本体は ``datasets/cli.py`` にある。ここを分けているのは、
``python -m rc_basics_lab.datasets.cli`` だと ``datasets/__init__`` が先に
``cli`` を import してしまい ``RuntimeWarning: found in sys.modules ...`` が
出るため (``__init__`` からの再エクスポートは
``tests/test_public_api_reexport.py`` が全公開サブモジュールに要求している)。
"""

from __future__ import annotations

from rc_basics_lab.datasets.cli import main

raise SystemExit(main())
