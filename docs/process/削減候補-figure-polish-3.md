# 削減レビュー — サイクル `cycle-figure-polish-3`（2026-08-31）

対象: 図の修正点 (2026-08-30) の 2-7 / 2-8 / 2-13 / 2-16 / 1-7 を実装したサイクル
(指示書は消化済みで畳んである。`docs/plans/README.md` の索引)
（`git log cycle-figure-polish-2..cycle-figure-polish-3`）。

**実施者について**: このサイクルでは `reviewer-deletion` サブエージェントを使わず、
本体セッションが同じ観点（重複・常駐した rationale・使われていない公開面・
肥大したモジュール）で直接レビューした。セッションの運用設定でサブエージェント
起動が抑止されていたためである。**「サブエージェントが走った」とは書かない** ——
証跡の出所を偽ると、証跡そのものの意味が無くなる。

## 観点1: 重複した定義

`tests/test_duplicate_symbol_budget.py` 緑（18 passed）。

サイクル中に 1 件発火した:

- `_save` が `figures_capacity.py` と `figures_mc_sweep.py` に二重定義された
  （モジュール分離の際に写した）。**別名 import (`save_png as _save`) に寄せて解消**。
  他の `figures_*` は元から別名 import なので、写したほうが例外だった。

**残っている重複はない。**

## 観点2: 常駐した rationale（散文の重複）

**1 件、対処せず記録に留める。**

行数上限で分離したモジュールの冒頭に、同じ趣旨の説明が 7 ファイルへ広がった:

```
$ grep -rl "行数上限 (D-77)" src/rc_basics_lab/plotting/ | wc -l
7
$ grep -r "上限のほうは緩めない" src/rc_basics_lab/plotting/ | wc -l
4
```

`figures_washout.py` / `figures_mc_sweep.py` / `figures_leak.py` /
`figures_ipc_profile.py` / `figures_stability.py` / `layout.py` / `figures_esp.py`。

**削るべきだが、このサイクルでは削らない。** 理由は 2 つある:

1. 文面は「なぜこのモジュールが在るか」で、モジュールごとに**続く一文が違う**
   （2-D は `WashoutRow` を読むので切れる / 3-A は破断軸を持つ、など）。
   共通部分だけを抜くと、残った一文が宙に浮く。
2. 正しい落とし所は `docs/adr/` か `decisions.yaml` の D-77 側に 1 回書き、
   各モジュールは `(D-77)` の ID 参照だけにすることである。これは
   CLAUDE.md の「散文が3箇所にあると3箇所が独立にドリフトする」そのもので、
   **7 ファイルの docstring を書き換える別サイクルの作業**になる。

→ 次サイクルの候補として残す (05a の申し送りと同じ扱い)。

## 観点3: 使われていない公開面

サイクルで追加した公開シンボルの参照数（定義・`__all__` を除く）:

| シンボル | 参照 | 判定 |
|---|---|---|
| `broken_axis.BREAK_RATIO` | 3 | 使用中（本体1 + テスト2） |
| `broken_axis.needs_break` | 3 | 使用中 |
| `broken_axis.draw_break_marks` | 3 | 使用中 |
| `style.PANEL_TITLE_SIZE` | 8 | 使用中 |
| `figures_anomaly.ZOOM_MARGIN` | 2 | 使用中 |
| `figures_esp.replicate_count` | 7 | 使用中（`figures_leak` と共有） |

**死んだ公開面はない。** ただし追加時点では `BREAK_RATIO` / `ZOOM_MARGIN` の
参照が本体だけで、**直接のテストが無かった**（観点5 を参照）。

## 観点4: 肥大したモジュール

`tests/test_module_line_budget.py` 緑。このサイクルで**ラチェットが 2 段締まった**:

| モジュール | 前 | 後 | 扱い |
|---|---|---|---|
| `figures_esp.py` | 801（凍結 716 超過） | 387 | **FROZEN から除外**（通常上限 600 の内側） |
| `figures_capacity.py` | 650（凍結 643 超過） | 455 | **FROZEN から除外** |

分離先: `figures_washout.py`（323）/ `figures_leak.py`（255）/
`figures_mc_sweep.py`（292）/ `broken_axis.py`（80）。いずれも上限の内側。

**上限は 1 度も緩めていない。** `plotting/` の `figures_*` は 16 本になったが、
1 モジュール = 1 図（または 1 実験の図群）で対応が付いており、探索の手間は
増えていない。

## 観点5: 見つけた欠落（このサイクルで直した）

削減観点のレビュー中に、**削減とは別の実在の欠落**が 2 件出た。記録して直した:

1. **`broken_axis.py` に直接のテストが無かった。**
   `fig_mc_sweep` 経由で間接的に踏まれるだけで、CLAUDE.md の
   「新しいコードには必ずテストを書く。`test_{モジュール名}.py`」に反していた。
   → `tests/test_broken_axis.py` を追加（7 件）。**変異注入で赤を確認**:
   - `needs_break` を常に `False` にする → 3 件が赤
   - 上段の下スパインを消さない → 1 件が赤
2. **拡大窓の選択（`_zoom_span`）に直接のテストが無かった。**
   D-107（「よく当たっている区間を選べる図にしない」）を守る唯一の機械なのに
   測られていなかった。→ `tests/test_plotting_anomaly.py` に 3 件追加。
   **変異注入で赤を確認**（余白を 0 にする → 1 件が赤）。

これは「削減レビューを走らせないと何が起きるか」の実例でもある。
**間接的にしか踏まれていないコードは、削除候補にも欠落にも見えない。**

## 結論

- 削除・統合すべきものは**なし**（重複 1 件はサイクル中に解消済み）
- 次サイクルへ持ち越す候補が**1 件**: 行数上限の rationale を D-77 側へ集約し、
  7 モジュールの docstring を ID 参照だけにする
- レビュー中に見つけたテストの欠落 2 件は**このサイクルで直した**
