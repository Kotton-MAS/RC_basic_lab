"""成果物 (``results/``) が1バイトも変わっていないことを固定する (D-74)。

このリポジトリのガードは「壊れていないこと」を測るものばかりで、
「読めるか・削れるか」を測る軸が無い —— 8サイクルで src は単調増加し、
削除・統合を要求した findings は0件だった
(docs/process/agent-system-review-from-artifacts.md 1 節)。

整理を進めるには「振る舞いを変えずに削れたか」を言える道具が要る。
その判定は**成果物が1バイトも変わらないこと**であり、この検査がその道具である。
これまで守る側 (壊すな) にしか使われていなかったバイト不変検査を、
初めて攻める側 (削れ) の合否判定に使う。

成果物を意図的に変えたときは ``make artifacts-manifest`` で指紋を書き直す。
その差分はレビューで必ず目に入るので、理由を説明せずには通せない。
"""

from __future__ import annotations

import pytest
from _artifact_manifest import (
    MANIFEST_PATH,
    REPO_ROOT,
    artifact_paths,
    fingerprint,
    read_manifest,
)


def test_every_artifact_matches_its_committed_fingerprint() -> None:
    """``results/`` の全ファイルが指紋と一致すること。

    自動コミット (SubagentStop) が作業ツリーを丸ごと拾うため、成果物が
    変わってもコミットされて ``git status`` は綺麗なままになる。
    基準 ref からの差分を見るには基準 ref を覚えている必要があり、
    **気づかないまま通る経路**が残っていた。ここで落とす。

    **時間だけの差と、実質的な差を分けて報告する。** 23 枚の CSV のうち
    15 枚と JSON 5 枚が実行時間を含むので、再生成すると数値が同じでも
    バイト指紋は必ず動く。両方を同じ一覧に並べると、本物の変化が時間の
    ノイズに埋もれる (実測: 03 の capacity.csv は 118 行すべてが毎回変わる)。
    """
    expected = read_manifest()
    changed: list[str] = []
    timing_only: list[str] = []
    stale_content: list[str] = []
    for path in artifact_paths():
        relative, digest, size, content = fingerprint(path)
        if relative not in expected:
            continue  # 追加は別のテストが見る
        want_digest, want_size, want_content = expected[relative]
        line = f"{relative}: {want_digest[:12]}…/{want_size}B → {digest[:12]}…/{size}B"
        # **内容指紋を先に、バイト指紋と独立に見る** (D-141)。バイトが一致
        # したら content を見ない書き方だと、内容指紋の列が古くなっても
        # 誰も気づけない —— そして次に本物の退行が起きたとき、古い値との
        # 比較で「実行時の値だけが変わった」側に分類されて素通りする。
        if content != want_content:
            if (digest, size) == (want_digest, want_size):
                stale_content.append(
                    f"{relative}: {want_content[:12]}… → {content[:12]}…"
                )
            else:
                changed.append(line)
        elif (digest, size) != (want_digest, want_size):
            timing_only.append(line)

    report = ""
    if stale_content:
        report += (
            "**内容指紋の列が古くなっています** (バイトは一致。指紋の取り方を"
            "変えたか、列を手で書き換えたか):\n" + "\n".join(stale_content) + "\n\n"
        )
    if changed:
        report += "**実行時の値以外が変わった成果物**:\n" + "\n".join(changed) + "\n\n"
    if timing_only:
        report += (
            f"実行時の値だけが変わった成果物 ({len(timing_only)} 件。"
            "実行時間・時刻・commit を除くと同じなので、退行ではありません):\n"
            + "\n".join(timing_only)
            + "\n\n"
        )
    if changed:
        report += (
            "**PNG は footnote に commit を焼き込む** (FIG-6 / D-87) ので、"
            "HEAD が動いていれば必ず上の側に出る。画素から commit だけを抜く"
            "ことはできないため、再生成して一致するかは "
            "tests/test_golden.py が commit を固定して確かめている。\n\n"
        )
    assert not report, (
        report + "意図した変更なら `make artifacts-manifest` で指紋を書き直し、"
        "なぜ変わったかをコミットメッセージに書いてください。"
    )


def test_no_artifact_was_added_or_removed_without_updating_the_manifest() -> None:
    """成果物の増減も指紋の更新なしには通さない。

    「削れるか」の判定では、消えた図や CSV に気づけないことが最も痛い。
    """
    expected = set(read_manifest())
    found = {fingerprint(path)[0] for path in artifact_paths()}
    added = sorted(found - expected)
    removed = sorted(expected - found)
    assert not added and not removed, (
        f"成果物の構成が変わっています (追加: {added} / 消失: {removed})。\n"
        "意図した変更なら `make artifacts-manifest` で指紋を書き直してください。"
    )


def test_the_manifest_covers_the_results_directory() -> None:
    """指紋そのものが空振りしていないこと。

    ``results/`` を読み違えて0件の指紋を作ると、上の2つは「全一致」で
    緑になる。このリポジトリで実際に起きた『空虚なガード』(13件) と
    同じ形なので、前提を別に固定する。
    """
    expected = read_manifest()
    assert len(expected) >= 30, (
        f"指紋が {len(expected)} 件しかありません "
        f"({MANIFEST_PATH.relative_to(REPO_ROOT)} が空振りしている可能性があります)"
    )
    assert any(name.endswith(".png") for name in expected), "図の指紋がありません"
    assert any(name.endswith(".csv") for name in expected), "CSV の指紋がありません"
    assert any(name.endswith("meta.json") for name in expected), (
        "meta.json の指紋がありません"
    )


def test_a_stale_content_digest_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """**バイトが一致していても内容指紋の食い違いを見つける** (D-141)。

    以前はバイト指紋が一致した時点で ``content_sha256`` を見ずに次へ進んで
    いた。指紋の取り方を変えても列が古いまま緑で通り、そのあと本物の退行が
    起きたときに**古い値との比較で「実行時の値だけが変わった」側へ分類**
    されて素通りする。ここが空虚だと、分類そのものが信用できなくなる。
    """
    import test_artifact_invariance as module

    real = read_manifest()
    target = "results/03_capacity/meta.json"
    assert target in real, "対象の成果物がありません (この検査が空振りします)"
    digest, size, _ = real[target]
    poisoned = {**real, target: (digest, size, "0" * 64)}
    monkeypatch.setattr(module, "read_manifest", lambda: poisoned)

    with pytest.raises(AssertionError, match="内容指紋の列が古くなっています"):
        module.test_every_artifact_matches_its_committed_fingerprint()


def test_a_real_change_is_not_filed_under_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**数値が変わった成果物を「実行時の値だけ」側へ入れない** (D-141)。

    分類を間違えても行数は同じなので、報告は一見もっともらしいまま通る。
    「実行時の値だけが変わった」は**説明を要さない**という意味なので、
    そこへ誤って入った退行は誰にも読まれない。
    """
    import test_artifact_invariance as module

    real = read_manifest()
    target = "results/03_capacity/meta.json"
    assert target in real, "対象の成果物がありません (この検査が空振りします)"
    _, size, _ = real[target]
    poisoned = {**real, target: ("f" * 64, size + 1, "0" * 64)}
    monkeypatch.setattr(module, "read_manifest", lambda: poisoned)

    with pytest.raises(AssertionError, match="実行時の値以外が変わった成果物"):
        module.test_every_artifact_matches_its_committed_fingerprint()
