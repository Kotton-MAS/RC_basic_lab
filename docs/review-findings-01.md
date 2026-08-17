# レビュー findings 記録 — rc-basics-01 サイクル1

**この文書は削除しないこと。** `src/` と `tests/` の docstring / コメントが、この文書に載っている
finding ID（`F-1-xxx` / `F-2-xxx`）を「なぜこうなっているか」の根拠として参照している。
`tests/test_finding_id_references_resolve.py` が、コード中に出現する全 ID がこの文書に実在することを
機械的に検証している。参照が解決できなくなった時点でテストが赤くなる。

**新しく finding ID を docstring に書いたら、このファイルにも追記すること。** 逆にこの文書からエントリを
消す前に、`grep -rnE "F-[0-9]+-[0-9]{3}" src/ tests/` でコード中に参照が残っていないか確認すること。

## この文書の性格

「次の PR 候補」ではなく、rc-basics-01 サイクル1（round 1 の実装レビュー + round 2 の追検証）で
出た findings の **対応記録**。BLOCKER/HIGH は各 round 内で修正済み、MEDIUM/INFO は個別に
対応 / 見送り（理由つき） / ユーザー判断待ちを記録している。02 着手前に過去の判断を再確認したいときは
ここを見る。

---

## round 1（`.claude/tmp/findings/round-1/triage.json`、reviewer 7名、BLOCKER 0 / HIGH 2 / MEDIUM 9 / INFO 10）

### BLOCKER/HIGH（round 1 内で修正済み）

- **F-1-001** [HIGH / architecture] `diagnostics/` → `reservoir/` の推移的依存を防ぐ guard test が、
  直接 import しか見ておらず `config.py` 経由の間接依存を検出できなかった。
  → 別プロセスで `sys.modules` を検査する guard test を追加し、決定 `D-12` として記録
  （`tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir`）。
  round 2 でさらに `pkgutil.iter_modules` による全サブモジュール検査へ強化（→ F-2-001）。
- **F-1-017** [HIGH / test] `_coerce_scalar` の bool→int/float 暗黙変換拒否に対する回帰テストが1件も無かった。
  → `tests/test_config.py::test_coerce_scalar_rejects_loose_conversions`（14ケースの parametrize）を追加。

### MEDIUM/INFO — 対応済み

- **F-1-002** [MEDIUM / architecture] `bias_column` の既定値 `0` が「先頭列は必ずバイアス」を暗黙の前提にし、
  `bias=False` の設計行列を渡すと無罰則列が静かに変わりうる。
  → `DesignMatrix.bias_column` プロパティを追加し、`penalty_matrix` / `fit_ridge` / `select_alpha` の
  `bias_column` をキーワード必須にして既定値を外した（`src/rc_basics_lab/readout/ridge.py`,
  `src/rc_basics_lab/readout/design.py`）。渡し忘れは `TypeError` になることを
  `tests/test_ridge.py::test_bias_column_is_keyword_required` で固定。
- **F-1-003** [MEDIUM / architecture] 集計ロジック (`aggregate_nrmse`) が表示層 (`plotting/figures.py`) に
  あり、matplotlib を import しないと集計できなかった。
  → `experiment/summary.py` へ移し、戻り値型を公開 dataclass `Aggregate` にした。
- **F-1-004** [MEDIUM / architecture] `main.py` の `EXPERIMENTS` docstring が「02〜05 はここに1行足すだけ」
  と誤った拡張方法を宣言していた。
  → docstring を実態（設定クラスとパイプラインの組を足す必要があり、YAML 追加だけでは動かない）に更新
  （`main.py`）。
- **F-1-005** [MEDIUM / architecture] 集計値（平均±標準偏差）が生成物として存在せず、README の表が手書きで
  実測値との一致が機械的に保証されていなかった。
  → `experiment/report.py::write_comparison_summary_csv` で `results/comparison_summary.csv` を書き出し、
  README の表がこの生成物と一致することを `tests/test_readme_summary.py` で固定。
- **F-1-007** [INFO / architecture] `docs/要件_rc-basics-02.md` の設計判断1 が D-01 と異なる ESP 判定の署名
  (`f(state_sequence_a, state_sequence_b) -> ESPResult`) を規定しており、02 着手時に D-01 と衝突する
  おそれがあった。
  → 要件_02 の設計判断1 を D-01 の共通署名に合わせて書き換え済み（実装変更なし、ドキュメントのみ）。
- **F-1-008** [INFO / architecture] `setup_style()` が `matplotlib.rcParams` をプロセス全体へ破壊的に
  更新し、復元していなかった。
  → `rc_context` ベースへ変更し、`setup_style()` は `StyleContext` の生成のみに責務を絞った
  （`src/rc_basics_lab/plotting/style.py`, `figures.py`）。
- **F-1-009** [INFO / architecture] `collect_state_space` が `plan_replicate` を呼び直し、
  レプリケート0の計算（タスク生成・ESN 構築・設計行列構築）が2度走っていた。
  → `run_and_report` がレプリケート0の `ReplicatePlan` を1回だけ作り、`run_experiment` と
  `collect_state_space` の両方へ明示的に渡す形にした（`experiment/pipeline.py`, `runner.py`,
  `state_space.py`）。
- **F-1-010** [MEDIUM / performance] `select_alpha` が alpha 格子の本数だけ Gram 行列
  (`ΦᵀΦ`, O(T·F²)) を再計算していた（実測 3.47倍の無駄）。
  → Gram 行列を1回だけ計算して `solve` のみを繰り返す `fit_ridge_from_gram` を追加し、`select_alpha` から
  使うよう変更（`src/rc_basics_lab/readout/ridge.py`）。数学的に `fit_ridge` と同一であることを
  `tests/test_ridge.py::test_fit_ridge_from_gram_matches_fit_ridge` で固定。
- **F-1-014** [MEDIUM / style] `plot_comparison` が87行 / CC=11 で目安を超過し、4責務が同居していた。
  → `_plot_task_panel` へ切り出し、50行台に収めた（`src/rc_basics_lab/plotting/figures.py`）。
- **F-1-015** [MEDIUM / style] `build_design_matrix` の CC=15（リポジトリ最高）で、形状検証と特徴ブロック
  組み立てが同居していた。
  → 形状検証を `_validate_inputs` に分離した（単一入口は維持、`src/rc_basics_lab/readout/design.py`）。
- **F-1-018** [MEDIUM / test] D-01 の guard test が `mypy strict` 頼みで、`ctx` の keyword-only 制約違反を
  pytest 単体では検出できなかった（実測で確認済み）。
  → `inspect.signature` で契約を実行時に検査する
  `tests/test_diagnostics_base.py::test_all_diagnostics_conform_to_d01_signature_contract` を追加。
- **F-1-019** [MEDIUM / test] `test_invalid_config_raises` / `test_shape_errors` が複数のエラーケースを
  1テストに束ねており、最初の失敗で残りが検証されなかった。
  → 他ファイルの慣習に合わせ `@pytest.mark.parametrize` へ分解（`tests/test_reservoir.py`）。

### MEDIUM/INFO — 見送り（理由つき）

- **F-1-011** [INFO / performance] `ESN.run` の逐次ループ（`_update` + `_input_drive`）が実験全体の
  約36%を占める最大コスト。`step` とのビット一致を保証する意図的な設計であり、再帰依存で本質的に逐次。
  現状 300秒制約に対し2桁の余裕がある。→ **見送り**。N/T が大きく伸びる場合の材料として記録のみ。
- **F-1-012** [INFO / performance] Mackey-Glass の RK4 積分（約31%）。5レプリケートは独立でバッチ化余地は
  あるが、300秒制約に対し無害。→ **見送り**。`n_replicates` が大きく増える場合のみ検討。
- **F-1-013** [INFO / security] `git` を PATH 経由で解決（CWE-426相当）。`cwd` 固定・`timeout=5`・
  `check=False`・`shell=True` 不使用で扱いは適切、reviewer-security 自身が現状維持を推奨。
  → **見送り**（現状維持でよい）。
- **F-1-016** [INFO / style] `select_alpha` が6引数（目安5個をわずかに超過）。`(phi, y)` ペアを表す型を
  導入すれば4引数に減るが、「`phi` と `y` を並べて渡す」という既存の呼び出し規約の一貫性を崩す。
  → **見送り**。優先度低、現状維持で許容範囲。
- **F-1-020** [INFO / uv] wheel に `py.typed` が含まれること、余分なファイルが入らないことを実測確認済み。
  → 対応不要（確認事項の記録）。
- **F-1-021** [INFO / uv] `pythonpath = ["."]` は `tests/test_main.py` がルートの `main.py` / `conftest`
  を import するため今も必要。
  → 対応不要（確認事項の記録）。
- **F-1-006** [INFO / architecture] D-01 の rule 本文が「拡張は `DiagnosticContext` への既定値つき
  フィールド追加のみ」とだけ書いており、診断固有のパラメータまで `ctx` に押し込む読み方を誘発していた。
  05 まで進むと `DiagnosticContext` が全診断の設定の union になり、要件_02 設計判断3（ESP判定の閾値と窓は
  設定可能）・要件_03（IPC のサロゲート本数・最大遅延/次数）のパラメータをどこに置くかが決まらない問題が
  あった。→ ユーザー承認のうえ対応済み。D-01 の `rule` に「`DiagnosticContext` が持つのはデータの素性
  (`washout`/`dt`/`seed`/`companion_states`) のみとし、診断固有パラメータはパラメータ化した callable
  (frozen dataclass の `__call__`) の構築時に渡す」を追記し（`.claude/decisions.yaml`）、
  `diagnostics/base.py` の docstring に使い分けの例を追記、guard test に
  `tests/test_diagnostics_base.py::test_parameterized_callable_conforms_to_d01_signature_contract`
  を追加（frozen dataclass の `__call__` も D-01 の署名契約を満たすことを実行時に固定）。
  `diagnostics/state_space.py` の `state_pca` を dataclass 化する対応自体は、02 で実際に必要性が
  出てから行う判断とし、今回はスコープ外（ルールと guard の整備のみ）。

### MEDIUM/INFO — ユーザー判断待ち

- **受け入れ条件4 の記事での扱い**（F-1-009 issue 内 (4)）: 「リザバー状態 > 遅延埋め込み入力」という
  当初期待が実測で不成立（数値は `docs/design.md` §7.2・`meta.json` に記録済み）。実装上の対応は不要だが、
  記事本文でこの結果をどう扱うか（正直に書く / 追加実験で条件を変える 等）は編集判断であり、
  **ユーザーの判断が要る**。
  → **保留**。

---

## round 2（`.claude/tmp/findings/round-2/triage.json`、reviewer 5名、BLOCKER 0 / HIGH 0 / MEDIUM 9 / INFO 4。
fixer-input は空 = round 2 に BLOCKER/HIGH は無かった）

round 2 は round 1 の HIGH 修正（F-1-001, F-1-017 の対応）に対する追検証。

### 対応済み

- **F-2-001** [MEDIUM / architecture] 推移閉包 guard test の probe が `diagnostics` パッケージ本体のみを
  import しており、`__init__.py` が再エクスポートしていないサブモジュールが検査対象から漏れる偽陰性が
  実測で確認された。
  → probe を `pkgutil.iter_modules` で全サブモジュールを個別 import する形に拡張
  （`tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir`）。
- **F-2-002** [MEDIUM / architecture] 「`diagnostics/` は `reservoir` に依存しない」という不変条件が
  `.claude/decisions.yaml` に登録されていなかった。
  → `D-12` として追加（guard_test は F-2-001 と同じテスト）。
- **F-2-003** [MEDIUM / architecture] `diagnostics/base.py` の docstring が禁止方向のみ書いており、
  許可される方向（`config.py` が diagnostics の設定 dataclass を import する向き）が明示されていなかった。
  → 1文追加（`src/rc_basics_lab/diagnostics/base.py`）。`conventions.md` にも reservoir と同様の例外注記を
  追加。
- **F-2-004** [INFO / architecture] probe とテスト本体の契約が stdout 全体のカンマ split で、
  被検査コードの stdout 汚染に弱かった。
  → 最終行マーカー付き JSON (`LEAKED=...`) 方式に変更。
- **F-2-005 / F-2-006** [MEDIUM / docs] README.md / docs/design.md のテスト件数記述が実測と乖離していた
  （当時 200件 → 実測215件）。
  → 当時の実測値に更新済み。**注記**: その後さらにテストが追加され、本文書作成時点の実測は 229 件
  （今回の (2) 追加テストで +1）。件数はテスト追加のたびにドリフトしうる既知の性質であり、都度
  `uv run pytest -q` で確認するのが確実。件数表記の運用（具体数を書き続けるか、生成に切り替えるか）は
  別途検討の余地がある。
- **F-2-007 / F-2-010** [MEDIUM / docs, style] `tests/test_config.py` の docstring が
  `.claude/decisions.yaml` にも本文書にも存在しない一時的な ID `F-1-017` を引用しており、
  `.claude/tmp/` の掃除後に参照が宙に浮く構造だった。
  → `F-1-017` への参照を削除し、docstring 本文だけで理由が読める形に書き換えた
  （`tests/test_config.py::test_coerce_scalar_rejects_loose_conversions`）。
  **この finding が今回のタスクの直接の起点**（同じクラスの欠陥が2回目のため、今回はテスト層
  ((2) の新規テスト) で機械的に塞ぐ）。
- **F-2-011** [INFO / style] 別プロセスで実行する probe コードが文字列連結で読みにくかった。
  → `textwrap.dedent` + triple-quote に変更（`tests/test_diagnostics_base.py`）。

### 見送り（理由つき）

- **F-2-008** [INFO / docs] `base.py` の docstring 追記は文体・具体性とも良好という評価。
  → 対応不要（指摘なし、記録のみ）。
- **F-2-013** [INFO / test] `_coerce_scalar` の `target is bool` accept分岐は、`ExperimentConfig` に
  bool 型フィールドが1つも無いため現状到達不能（カバレッジ実測 94%、Missing 131-133）。
  → 対応不要（現状は正しい判断）。将来 bool フィールドを追加するタイミングで
  `test_coerce_scalar_rejects_loose_conversions` に「bool→bool は許可される」ケースを足すことを検討。

### 次の PR 候補（今回スコープ外、MEDIUM）

- **F-2-009** [MEDIUM / style] 新規 parametrize が `pytest.param(..., id=...)` 形式で、既存3箇所の
  `ids=[...]` 別引数形式と異なる（inferred, 実測なし）。既存に揃えるか、新形式を標準として
  `conventions.md` に明記するかは実装判断。
- **F-2-012** [MEDIUM / test] `test_diagnostics_package_does_not_import_reservoir`（AST版）が
  `diagnostics/__init__.py` に未登録のファイルを見逃す可能性がある（`test_diagnostics_package_does_not_import_reservoir`
  の対象は `package_dir.glob("*.py")` でファイル網羅だが、登録漏れそのものを検出する assert は無い）。
  02 で新規診断モジュールを追加する際に、`__init__.py` への配線漏れを検出する仕組みを足すことを検討。

---

## 着手順の提案（残っているものだけ）

1. **受け入れ条件4 の記事での扱い** — ユーザー判断待ち。02 着手前に決めておくと後戻りがない。
   （F-1-006 は decisions.yaml へ反映済み、対応済みに移動）
2. **F-2-012**（`__init__.py` 登録漏れの検出）— 02 で診断モジュールを追加する直前に。
3. **F-2-009**（parametrize の `ids` 形式統一）— 優先度低。
