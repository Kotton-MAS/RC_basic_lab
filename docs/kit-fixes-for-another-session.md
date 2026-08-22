# kit 側の改修事項 — 別セッション向け

*作成: 2026-08-22 / 対象リポジトリ: `~/app/Claude_multi_agent`（`~/.claude/{agents,hooks,scripts,skills}` の実体）*
*起点: `docs/agent-behaviour-fixes.md` の提案 A〜D / `docs/agent-system-review-from-artifacts.md` 第3版*

## この文書の位置づけ

`rc-basics-lab` 側で**提案 A・B は実装済み**（D-97 / D-98）。
本稿に残すのは、**kit リポジトリ側でしか直せないもの**である。

このセッションでは kit を触っていない。理由は3つ:

1. このワークツリーは `rc-basics-lab` に隔離されており、kit を触ると
   このリポジトリの PR に載らない変更が別リポジトリに生まれる
2. kit は**全プロジェクトに効く**。1リポジトリ・5サイクルの実測だけを根拠に
   全体へ広げるのは根拠が薄い
3. kit には独自の `tests` と `install.sh` がある。そちらで検証してから
   再インストールするのが筋

## 前提: kit の構成（実測）

```
~/.claude/agents  → ~/app/Claude_multi_agent/agents
~/.claude/hooks   → ~/app/Claude_multi_agent/hooks
~/.claude/scripts → ~/app/Claude_multi_agent/scripts
~/.claude/skills  → ~/app/Claude_multi_agent/skills
```

すべてシンボリックリンク。実体は git 管理された独立リポジトリで、
`install.sh` / `Makefile` / `pyproject.toml` を持つ。

---

## K-1【最優先】`observed` が空でも `measured` を名乗れる

**対象**: `scripts/triage_findings.py`（kit）＋ `.claude/schemas/findings.schema.json`（project）

### 実測した非対称

`triage_findings.py` の evidence 検証にこう書かれている（実測、L82-85）:

```python
if method == "measured" and not repro:
    method = "inferred"                       # ← 降格する
    notes.append("method=measured だが repro_command が空のため inferred へ降格")
if method == "measured" and not observed:
    notes.append("method=measured だが observed が空 (実行結果を書くこと)")
                                              # ← note を足すだけ。降格しない
```

つまり:

| フィールド | 空のとき | 実行しないと書けるか |
|---|---|---|
| `repro_command` | **降格する** | **書ける**（コマンドは実行しなくても書ける） |
| `observed` | note のみ | **書けない**（出力は実行しないと分からない） |

**強制が効いているのは「実行しなくても書ける方」だけ**である。
実行の有無を実際に分けるのは `observed` の側なのに、そちらが素通りする。

### 実害

`docs/agent-behaviour-fixes.md` 型5 の実例:
reviewer が存在しない関数を指して「100k点で 26.7 秒」と報告し、実測は **0.428 秒**（62倍）。
`repro_command` が書かれていれば `measured` のまま通る現在の実装では、これは止まらない。

### 直し方

1. **kit**: `triage_findings.py` の `observed` 側も降格に変える（1行の非対称を消す）
2. **project**: `findings.schema.json` の `required` を条件付きにする

```json
"if":   { "properties": { "method": { "const": "measured" } } },
"then": { "required": ["method", "repro_command", "observed"] }
```

現在 `required` は `["method"]` だけで、`repro_command` / `observed` は
description に「measured なら必須」と書いてあるだけである（**散文による必須**）。

> ⚠️ **2つは必ず同時に入れること。** スキーマだけ厳しくしても
> `triage_findings.py` が降格しなければ空振りする ——
> それは `docs/agent-behaviour-fixes.md` が「空虚なガード」と呼んでいる形そのものである。

### 検証方法

kit 側に、`observed` が空の `measured` finding を食わせて `inferred` へ降格することを
確かめるテストを1本足す。変異注入（降格を note に戻す）で赤くなることまで確認する。

---

## K-2 完了手順から「成果物を生まない手順」を消す（提案 C の残り）

**対象**: `skills/pdca`（kit）と、各プロジェクトの `CLAUDE.md`

### 実測

3巡の測定で、完了手順4つのうち**結果が目に見えるものだけが実行された**。

| 手順 | 結果が目に見えるか | 実測 |
|---|---|---|
| `make ci` | 赤になる | 実行された |
| `reviewer-deletion` | 何も起きない | **飛んだ** |
| タグを打つ | ログに出る | 実行された |
| `git worktree prune` | 何も起きない | **飛んだ** |

### 現状

飛んだ2つは `rc-basics-lab` 側で**結果側の機械**に置き換えた:

- `reviewer-deletion` の実行漏れ → `tests/test_cycle_hygiene.py`（D-98）
- `git worktree prune` → 同（D-93）
- 走らせ漏れても重複そのものは止まる → `tests/test_duplicate_symbol_budget.py`（D-92）

**したがってプロジェクト側の `CLAUDE.md` からは手順2・4を消してよい状態になった。**
ただし今回は実装対象を A・B に絞ったため、文言はまだ消していない。

### kit 側でやること

同じ問題が**他プロジェクトにもある**なら、`skills/pdca` の完了手順にも同じ判断が要る。
ただし他プロジェクトには D-92 / D-93 相当の機械が無いので、
**手順を消す前に機械を配る**必要がある。順序が逆になると単に検査が消えるだけになる。

候補: `tests/test_cycle_hygiene.py` / `tests/test_duplicate_symbol_budget.py` を
kit のテンプレートとして持ち、`install.sh` が新規プロジェクトへ置く。

---

## K-3 reviewer の観点定義に「動作点」を要求する（提案 A の一般化）

**対象**: `agents/reviewer-architecture.md` ほか

`rc-basics-lab` では、文献の実測値を引くときに動作点を必須にした（D-97）。
これは作図層の話に見えるが、**一般には「先行研究の手法を再現した」と
「先行研究の設計を検証した」の取り違え**である。

実例（D-95）: 3-C は先行の「正則化なし」という手法だけを再現して
「先行の対照を足した」と書いたが、先行の動作点は `k ≈ n_train` で
こちらは `k/n_train <= 0.01` だった。**動作点が違えば批判の検証にならない。**

reviewer の観点に1行入れる価値がある。ただし**プロンプト層への追記**なので、
`CLAUDE.md` の「再発している事象への対策をプロンプト層に置かない」に照らすと
**1回目だから許される**という位置づけになる。2回目が出たら層を下げること。

---

## K-4 調査の過程で確認できた「問題ではなかったもの」

無駄な調査を繰り返さないための記録。

| 疑ったもの | 実測 | 結論 |
|---|---|---|
| `select_reviewers.py` が project 固有 reviewer を知らない | `--round 1` で `reviewer-deletion` を含めて出力する | **問題なし** |
| kit が git 管理外 | `~/app/Claude_multi_agent/.git` が存在 | **問題なし**（正しく管理されている） |
| `~/.claude` を直接編集する必要がある | 実体はシンボリックリンク先のリポジトリ | **kit リポジトリで直せる** |

---

## 優先順位

| # | 項目 | 層 | 理由 |
|---|---|---|---|
| 1 | **K-1** `observed` の非対称 | kit + project スキーマ | 62倍の誤報告を止める唯一の機械。**2つ同時に入れること** |
| 2 | **K-2** 完了手順の整理 | kit | 機械を配ってから手順を消す（順序を逆にしない） |
| 3 | **K-3** reviewer の観点 | kit（プロンプト層） | 1回目なので許容。2回目は層を下げる |

## 別セッションを立てるときの前提

- 作業ディレクトリは `~/app/Claude_multi_agent`（`~/.claude` ではない）
- kit を直したら再インストール（`install.sh` / `Makefile` を確認）
- **`rc-basics-lab` 側の `.claude/schemas/findings.schema.json` は project 管理**なので、
  K-1 の片割れはこのリポジトリの PR として出す必要がある
