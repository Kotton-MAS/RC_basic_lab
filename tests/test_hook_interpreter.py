"""フックが動く処理系が、このリポジトリのコードを解析できることを固定する (D-73)。

このリポジトリは PEP 695 の型エイリアス (``type X = ...``) を使っており、
Python 3.11 以前では **構文解析すらできない**。一方でキットのフックは
``PY=$(command -v python3)`` で処理系を解決するので、``PATH`` や
``PYENV_VERSION`` の設定次第で古い処理系を掴む。

この2つが噛み合うと「フックは起動するがソースを読めない」という状態になる。
`decisions.yaml` のスキーマ検証しかしていない間は無害だが、rationale の数値を
一次資料 (``src/`` と ``results/``) と機械照合する仕組みを足した瞬間に破綻する。
しかもその破綻は例外ではなく **無言の失敗** として出る。

そこで「設定が古い処理系を掴ませていないこと」を機械で固定する。
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SETTINGS = REPO_ROOT / ".claude" / "settings.json"
LOCAL_SETTINGS = REPO_ROOT / ".claude" / "settings.local.json"


def _modules_using_pep695() -> tuple[Path, ...]:
    """``type X = ...`` を含むモジュール (= 3.12 でしか解析できないもの)。"""
    found: list[Path] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.TypeAlias) for node in ast.walk(tree)):
            found.append(path)
    return tuple(found)


def _settings_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    env = loaded.get("env", {})
    return {str(key): str(value) for key, value in env.items()}


def _parses_sources(interpreter: str, sample: Path) -> subprocess.CompletedProcess[str]:
    """``interpreter`` が ``sample`` を構文解析できるか。"""
    return subprocess.run(
        [interpreter, "-c", f"import ast; ast.parse(open({str(sample)!r}).read())"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_the_repository_uses_syntax_that_python_310_cannot_parse() -> None:
    """このガードが守る前提そのものを固定する。

    PEP 695 を使うモジュールが1本も無くなったら、このファイルの他のテストは
    「何も守っていないのに緑」になる。そうなったことに気づけるようにする。
    """
    modules = _modules_using_pep695()
    assert modules, (
        "PEP 695 の型エイリアスを使うモジュールが見つかりません。"
        "リポジトリが 3.11 以前でも解析できるようになったなら、"
        "このファイルのガードは役目を終えています (削除を検討してください)。"
    )


def _required_python() -> tuple[int, int]:
    """``pyproject.toml`` の ``requires-python`` が要求する最小バージョン。"""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)', text)
    assert match is not None, "pyproject.toml から requires-python を読めません"
    return int(match.group(1)), int(match.group(2))


def test_project_settings_do_not_pin_an_interpreter_older_than_required() -> None:
    """`.claude/settings.json` が古い処理系を固定していないこと。

    以前は ``PYENV_VERSION: "3.10.0"`` が固定されており、フックが動く処理系は
    ``src/rc_basics_lab/diagnostics/ipc.py`` を SyntaxError でしか読めなかった。

    **バージョン文字列そのものを判定する。** 「いま ``python3`` を実行して
    解析できるか」で判定してはいけない —— ``uv run`` 配下では venv の 3.12 が
    pyenv の shims を覆い隠すので、固定が戻っていても緑のまま通ってしまう
    (この誤りは実際に一度書いて、変異注入で捕まえた)。
    """
    pinned = _settings_env(PROJECT_SETTINGS).get("PYENV_VERSION")
    if pinned is None:
        return
    match = re.match(r"(\d+)\.(\d+)", pinned)
    assert match is not None, (
        f"PYENV_VERSION={pinned!r} からバージョンを読めません。"
        "固定するなら requires-python 以上であることが読み取れる形式にしてください"
    )
    found = (int(match.group(1)), int(match.group(2)))
    required = _required_python()
    assert found >= required, (
        f".claude/settings.json が PYENV_VERSION={pinned!r} を固定していますが、"
        f"このリポジトリは Python {required[0]}.{required[1]} 以上の構文 "
        f"(PEP 695 の型エイリアス) を使っており、その処理系では解析できません。"
        "フックの処理系はマシン固有なので、固定ではなく "
        ".claude/settings.local.json の PATH で解決してください"
    )


def test_the_local_path_override_points_at_a_live_interpreter() -> None:
    """`.claude/settings.local.json` の PATH 先頭が生きていること。

    マシン固有の処理系解決はここに置いてある (gitignore 済み)。venv を作り
    直すと先頭要素が dangling になり、``command -v python3`` は静かに次の
    要素 (pyenv の shims) に落ちて、元の壊れ方に戻る。**それが起きたことに
    気づけない**のが最も困るので、ここで落とす。
    """
    path_value = _settings_env(LOCAL_SETTINGS).get("PATH")
    if path_value is None:
        pytest.skip(
            ".claude/settings.local.json に PATH の上書きが無い "
            "(このマシンではフックの処理系を別の方法で解決している)"
        )
    head = Path(path_value.split(":", 1)[0])
    interpreter = head / "python3"
    assert interpreter.exists(), (
        f"PATH の先頭 {head} に python3 がありません。"
        "venv を作り直した場合はここを更新してください "
        "(放置すると pyenv の shims に落ちてフックが古い処理系を掴みます)"
    )
    sample = _modules_using_pep695()[0]
    completed = _parses_sources(str(interpreter), sample)
    assert completed.returncode == 0, (
        f"{interpreter} は {sample.relative_to(REPO_ROOT)} を解析できません:\n"
        f"{completed.stderr.strip()}"
    )
