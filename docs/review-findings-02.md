# レビュー findings 記録 — rc-basics-02 サイクル2a

**この文書はコードから参照されているので削除しないこと。** `src/` と `tests/` の docstring / コメントが、
この文書に載っている finding ID（`F-02-<round>-xxx` 形式）を「なぜこうなっているか」の根拠として参照している。
`tests/test_finding_id_references_resolve.py` が、コード中に出現する全 ID が接頭辞に対応する記録文書
（`F-02-*` はこの文書、接頭辞にサイクル番号を持たないレガシー形式 `F-<round>-xxx` は
`docs/review-findings-01.md`）に実在することを機械的に検証している。参照が解決できなくなった時点で
テストが赤くなる。

**新しく finding ID を docstring に書いたら、このファイルにも追記すること。**

## この文書の性格

サイクル1 (`docs/review-findings-01.md`) と同じ形式の「ID → 対応内容」の記録。ID の名前空間を
サイクルごとに区切るため（`F-1-004` のようなレガシー形式はサイクルをまたいで再利用されると衝突する。
実例: サイクル2a の `config.py` が書いた「F-1-004」は `review-findings-01.md` の F-1-004（`main.py` の
`EXPERIMENTS` docstring の件）と無関係な内容を指してしまっていた）、サイクル2以降の finding ID は
`F-02-<round>-<seq>` の形式で振る。今回は充実した記述は次サイクル (T5) に譲り、ID と1行要約の対応表のみ。

## サイクル 2a round 1 (`F-02-1-xxx`)

| ID | 概要 |
|---|---|
| F-02-1-002 | D-01 契約テストの追加引数境界チェック (D-15 guard) を、`_CONFIG_TYPES` という静的タプルから `pkgutil` 自動列挙 (`_iter_diagnostic_callables`) へ移設。新診断が黙って検査対象から外れる問題を解消。 |
| F-02-1-005 | 実験層の未着手を検出する時限装置の信管を、モジュール名1個 (`find_spec("rc_basics_lab.experiment.esp")`) から `experiment/` 配下のモジュール集合の変化 (`KNOWN_EXPERIMENT_MODULES` との突き合わせ) へ拡張。 |
| F-02-1-015 | D-01 契約テスト (`test_all_diagnostics_conform_to_d01_signature_contract`) の必須 assert が `u` を用意しない共通 `ctx` を使い回しており、`u` 依存の新診断で `suppress` に頼る回避策が最も安くなる構造だった。`MINIMAL_VALID_INPUT` レジストリ (診断 qualname → 最小有効入力) と、完全性を強制する `test_minimal_valid_input_registry_covers_all_diagnostics` を導入。 |
| F-02-1-016 | `conditional_lyapunov` 内の `scale = float(np.sqrt(n_units))` を検査するテストが存在しなかった。曲率を持つ `_decoupled_quadratic_system` を新設し、`test_lyapunov_per_step_is_independent_of_n_units` (N=8 vs 2000) を追加。 |
| F-02-1-017 | 伝播器が状態と異なる形状の配列を返すケースが未検査だった。`test_propagator_returning_wrong_shape_raises` を追加。 |
| F-02-1-020 | `test_public_api_reexport.py` の `PACKAGE_NAMES` が手書きの固定タプルで、実際のトップレベルパッケージ集合との完全性チェックが無かった。`test_package_names_matches_automatic_enumeration` を追加し `pkgutil.iter_modules` の自動列挙と突き合わせる。 |

## サイクル 2a round 2 (`F-02-2-xxx`)

| ID | 概要 |
|---|---|
| F-02-2-001 | F-02-1-015 の「二重化」(`MINIMAL_VALID_INPUT` レジストリ + 完全性テスト) が実際には防御になっていなかった。完全性テストはキー集合の一致しか見ておらず、不十分なファクトリ (例: `u` 依存の診断に `_minimal_input_no_extras` を誤って割り当てる) を検出できなかった (reviewer-test / オーケストレータが実測: 不十分なファクトリでも完全性テストが通ることを確認)。加えて、契約テストの必須 assert が `suppress(ValueError)` を5回含む同一関数内にあり、「6個目の suppress」を足す誘惑が強い構造だった。→ (1) 必須 assert を `suppress` を一切書かない独立関数 `test_minimal_valid_input_actually_produces_a_result` へ切り出し、(2) `test_minimal_valid_input_registry_covers_all_diagnostics` で各ファクトリを実際に呼び出して `DiagnosticResult` が返ることまで検証するよう拡張。2手攻撃 (不十分なファクトリの登録 + 必須 assert の suppress) を安全な変異注入で実測し、完全性テストが独立に落ちることを確認済み。 |

## ID 名前空間の衝突について (F-02-1-002/004/005/015/016/017/020 の由来)

サイクル 2a の実装時、`docs/review-findings-01.md`（サイクル1の記録）が既に使い切っていた
`F-1-002` / `F-1-004` / `F-1-005` / `F-1-015` / `F-1-016` / `F-1-017` / `F-1-020` を、サイクル2a の
docstring が「今回の finding」のつもりで再利用してしまっていた（`test_finding_id_references_resolve.py`
が ID の実在しか検査しておらず、指している内容の一致までは検査していなかったため、衝突したまま
テストは緑だった）。round 2 で `F-02-<round>-<seq>` へ改名し、本文書を新設して解決した。
