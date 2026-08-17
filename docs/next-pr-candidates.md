# 次のPR候補 — rc-basics-01 サイクル1 の MEDIUM / INFO findings

*出典: `.claude/tmp/findings/round-1/triage.json`（round 1、全7 reviewer）*
*このサイクルでは BLOCKER 0件 / HIGH 2件のみを修正した。以下19件は意図的にスコープ外とした。*

重要度順ではなく、**着手する価値の高い順**に並べてある。02 に着手する前に潰しておきたいものを上に置いた。

---

## A. 02 の着手前に片付けたいもの

### F-1-007 [INFO / architecture] 要件書02 と D-01 が同じ対象に別の署名を規定している
`docs/要件_rc-basics-02.md:39` — 要件_02 の設計判断1 は ESP 判定を
`f(state_sequence_a, state_sequence_b) -> ESPResult` の純関数と規定している。
一方 D-01 は第2軌道を `ctx.companion_states` に逃がす `f(X, u, y, *, ctx)` の1形に固定している。

**両文書が並存したまま 02 に着手すると、実装者が要件書に従って別署名を切り、D-01 が壊れる。**
重要度は INFO だが、**波及の大きさでは実質最優先**。実装変更は不要で、要件書02 の設計判断1 を
D-01 の共通署名に合わせて書き換えるだけで済む。

### F-1-004 [MEDIUM / architecture] `main.py` の docstring が拡張方法を誤って宣言している
`main.py:30` — `EXPERIMENTS` の docstring が「実験番号 → 設定 YAML。サイクル 02〜05 はここに1行足す」
と書いているが、実際には 02 の YAML は `ExperimentConfig` に無いキーを持つため D-09 で `ConfigError` になり、
`pipeline.run_and_report` も 01 固有の4成果物にハードコードされている。

**02 の実装者が「1行足すだけ」を信じて着手し、CLI 層の再設計を後から強いられる。**
最低限 docstring を実態に合わせる。パイプラインの一般化は別タスク。

### F-1-006 [INFO / architecture] D-01 の文言が診断固有パラメータを `DiagnosticContext` に押し込む読み方を誘発する
`src/rc_basics_lab/diagnostics/base.py:68` — 既に `dt`(04専用) / `seed`(03専用) /
`companion_states`(02専用) が入っており、05 まで進むと全診断の設定の union になる。
`Diagnostic` は `__call__` を持つ Protocol なので、パラメータ付き診断は frozen dataclass の
`__call__` として書けば署名を変えずに固有パラメータを構築時に渡せる。

D-01 の rule に「`DiagnosticContext` が持つのはデータの素性のみ。診断固有パラメータは
パラメータ化した callable の構築時に渡す」を1文追加する提案。**署名は変わらないので波及なし。**

### F-1-018 [MEDIUM / test] D-01 の guard_test が実行時には空振りしうる
`tests/test_diagnostics_base.py:37` — `test_dummy_diagnostic_conforms_to_protocol` は
呼び出しが成功することしか見ておらず、`ctx` のキーワード専用マーカーを外すという
**実際の D-01 違反を加えても pytest は通る**（実測）。ガードは `mypy strict` に依存しているが、
Stop フックの絞り込み実行では mypy が走るとは限らない。

**D-01 は後続4サイクル全部が乗る土台の決定なので、ガードが型検査任せなのは弱い。**

---

## B. 実験結果の信頼性に関わるもの

### F-1-005 [MEDIUM / architecture] 集計値が生成物として存在せず README の表が手書き
`results/comparison.csv:1` — 要件書の出力仕様は「NRMSE と、シード複数本の平均±標準偏差」だが、
実装の CSV は長形式30行のみで mean/std 列が無く、`meta.json` にも集計値が無い。
集計値が存在するのは図の中と `README.md` の**手書きの表**だけ。

承認済み仕様（`docs/plans/rc-basics-01.md`）が「長形式 + 集計は下流」と決めているので実装は仕様どおりだが、
**「記事の数値とリポジトリの実測値を機械的に一致させる」という連載の規律がこの1点だけ手作業に依存している。**
実験を回し直したとき、README の表だけが古い値のまま残っても何も落ちない。

修正案: `aggregate_nrmse` を experiment 側へ移し（F-1-003 と同時）、集計結果を
`meta.json` の summary キーか `results/comparison_summary.csv` として書き出す。

### F-1-002 [MEDIUM / architecture] `bias_column=0` の既定値がリーキーな抽象化
`src/rc_basics_lab/readout/ridge.py:98` — `penalty_matrix` / `fit_ridge` / `select_alpha` の
`bias_column: int | None = 0` が「先頭列は必ずバイアス」を暗黙の前提にしている。
`build_design_matrix` は `bias=False` の設計行列を作れる（`state_space.py:116` が使用）。
その行列を既定値のまま `fit_ridge` に渡すと **`u` の lag0 列が黙って無罰則になり、D-03 の意図と逆の縮小**が起きる。

現状この経路は無いので実害は未発生。ただし 02〜05 で読み出しを足すたび、
**渡し忘れが例外ではなく「少しだけ違う係数」として静かに通る。**

修正案: `DesignMatrix` に `bias_column` プロパティを持たせ、`bias_column` の既定値 0 を外して
キーワード必須にし、渡し忘れを型で落とす。

---

## C. 性能（現状は実害なし、将来のため）

### F-1-010 [MEDIUM / performance] `select_alpha` が Gram 行列を alpha 格子の本数だけ再計算
`src/rc_basics_lab/readout/ridge.py:150` — Gram 行列 `ΦᵀΦ` は alpha に依存しないのに、
alpha ごとに `fit_ridge` を呼んで毎回再計算している。

**実測**: マイクロベンチマーク（T=3900, F=201）で 13回の `fit_ridge` 0.00777秒 に対し、
Gram を1回だけ計算して solve のみ13回で 0.00224秒 = **3.47倍**。
cProfile では `fit_ridge` 1200 calls / tottime 0.098秒（実験全体 0.976秒 の約10%）。

**現状 300秒制約に対して2桁の余裕があるので急がないが、02〜05 で N や格子が拡大すると
`F²` に比例して無駄が増える。格子拡大前に直す価値がある。**

### F-1-011 [INFO / performance] `ESN.run` の逐次ループが単一関数として最大の寄与
`src/rc_basics_lab/reservoir/esn.py:258` — `_update`(0.256秒) + `_input_drive`(0.091秒) で
実験全体の約35.6%。**ただし `step` とのビット一致を保証する意図的な設計**（自走実験のため）であり、
再帰依存で本質的に逐次。現時点で変更不要。03 で状態行列が拡大する場合の材料として記録。

### F-1-012 [INFO / performance] Mackey-Glass の RK4 積分が2番目のコスト
`src/rc_basics_lab/tasks/mackey_glass.py:78` — tottime 0.194秒（全体の約31%）。
5レプリケートは独立なのでレプリケート方向のバッチ化余地はある。`n_replicates` が大きく増える場合のみ検討。

---

## D. 可読性・構造

### F-1-014 [MEDIUM / style] `plot_comparison` が87行 / CC=11
`src/rc_basics_lab/plotting/figures.py:125` — CLAUDE.md の目安50行を超過し、
radon 実測でリポジトリ内で最も長い。1つの for ループに「誤差棒の計算」「基準線の描画」
「注記のアノテーション」「軸範囲・ラベル設定」の4責務が同居。
`_plot_task_panel` への切り出しで `plot_state_space` ともども50行台に収まる。

### F-1-015 [MEDIUM / style] `build_design_matrix` の CC=15（リポジトリ最高）
`src/rc_basics_lab/readout/design.py:100` — 「3ベースライン共通の唯一の入口」という設計意図は妥当だが、
形状検証と特徴ブロックの組み立てが同じ関数内で分岐している。
検証部分を `_validate_inputs` に分離しても**単一入口は保たれる**（D-05・受け入れ条件1 に抵触しない）。

### F-1-003 [MEDIUM / architecture] 集計ロジックが表示層にある
`src/rc_basics_lab/plotting/figures.py:92` — `aggregate_nrmse` が `plotting/` にあり、
戻り値型が private な `_Aggregate`。**matplotlib を import しないと集計できない。**
F-1-005 と同時に experiment 側へ移すのが自然。

### F-1-019 [MEDIUM / test] エラーケースを1テストに束ねている
`tests/test_reservoir.py:145,159` — `test_invalid_config_raises` / `test_shape_errors` が
5件・4件の独立したエラーケースを束ねており、同じPR内の他ファイルの `parametrize` 慣習と不整合。
1件目で失敗すると残りが検証されない。

### F-1-016 [INFO / style] `select_alpha` が6引数
`src/rc_basics_lab/readout/ridge.py:123` — CLAUDE.md の目安5個をわずかに超える。
`(phi, y)` のペアを表す小さな型を導入すれば4引数に減る。優先度低。

---

## E. その他（対応不要と判断されたもの）

### F-1-009 [INFO / architecture] `plan_replicate` の二重実行
`src/rc_basics_lab/experiment/state_space.py:112` — `collect_state_space` が `plan_replicate` を
呼び直すため、レプリケート0のタスク生成・ESN 構築・設計行列構築が2度走る。
図と CSV が同じ軌道を見ていることは `make_rng` の決定性への暗黙依存（現状は D-06 により決定的）。

### F-1-008 [INFO / architecture] `setup_style()` が rcParams をプロセス全体に破壊的更新
`src/rc_basics_lab/plotting/style.py:84` — ライブラリ関数がグローバルな第三者設定を書き換え、復元しない。
`matplotlib.rc_context` の with ブロックで包むのが素直。優先度低。
*（triage が「数値主張を含むが実測されていない」と指摘した項目）*

### F-1-013 [INFO / security] `git` を PATH 経由で解決
`src/rc_basics_lab/meta.py:36` — CWE-426 相当だが、攻撃者が既に PATH を書ける前提が必要。
`cwd` 固定・`timeout=5`・`check=False`・`shell=True` 不使用で扱いは適切。**現状維持でよい。**

### F-1-020 / F-1-021 [INFO / uv] 確認事項の記録（対応不要）
- wheel に `py.typed` が含まれること、余分なファイルが入らないことを実測確認済み
- `pythonpath = ["."]` は `tests/test_main.py` がルートの `main.py` と `conftest` を import するため**今も必要**

---

## 着手順の提案

1. **F-1-007**（要件書02 と D-01 の矛盾）— 02 着手前に必須。ドキュメント1箇所
2. **F-1-004**（`main.py` の docstring）— 02 着手前。docstring 1箇所
3. **F-1-018**（D-01 guard の強化）+ **F-1-006**（D-01 の文言追加）— セットで
4. **F-1-003 + F-1-005**（集計を experiment 側へ移し生成物にする）— セットで
5. **F-1-010**（Gram 行列のキャッシュ）— 格子・N を拡大する前に
6. **F-1-002**（`bias_column` の必須化）
7. 残り（style / test の構造改善）
