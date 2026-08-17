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
| F-02-1-004 | `EspDecayConfig` / `TimescaleSweepConfig` / `EspMapConfig` が `input_scale` / `n_units` / `density` / `n_replicates` を重複定義していたのを `ReservoirSweepConfig`（`Esp02Config.reservoir` 直下の1インスタンス）に集約。セクションごとに `n_units` 等が食い違う事故を構造的に禁止。 |
| F-02-1-005 | 実験層の未着手を検出する時限装置の信管を、モジュール名1個 (`find_spec("rc_basics_lab.experiment.esp")`) から `experiment/` 配下のモジュール集合の変化 (`KNOWN_EXPERIMENT_MODULES` との突き合わせ) へ拡張。 |
| F-02-1-015 | D-01 契約テスト (`test_all_diagnostics_conform_to_d01_signature_contract`) の必須 assert が `u` を用意しない共通 `ctx` を使い回しており、`u` 依存の新診断で `suppress` に頼る回避策が最も安くなる構造だった。`MINIMAL_VALID_INPUT` レジストリ (診断 qualname → 最小有効入力) と、完全性を強制する `test_minimal_valid_input_registry_covers_all_diagnostics` を導入。 |
| F-02-1-016 | `conditional_lyapunov` 内の `scale = float(np.sqrt(n_units))` を検査するテストが存在しなかった。曲率を持つ `_decoupled_quadratic_system` を新設し、`test_lyapunov_per_step_is_independent_of_n_units` (N=8 vs 2000) を追加。 |
| F-02-1-017 | 伝播器が状態と異なる形状の配列を返すケースが未検査だった。`test_propagator_returning_wrong_shape_raises` を追加。 |
| F-02-1-020 | `test_public_api_reexport.py` の `PACKAGE_NAMES` が手書きの固定タプルで、実際のトップレベルパッケージ集合との完全性チェックが無かった。`test_package_names_matches_automatic_enumeration` を追加し `pkgutil.iter_modules` の自動列挙と突き合わせる。 |

## サイクル 2a round 2 (`F-02-2-xxx`)

| ID | 概要 |
|---|---|
| F-02-2-001 | F-02-1-015 の「二重化」(`MINIMAL_VALID_INPUT` レジストリ + 完全性テスト) が実際には防御になっていなかった。完全性テストはキー集合の一致しか見ておらず、不十分なファクトリ (例: `u` 依存の診断に `_minimal_input_no_extras` を誤って割り当てる) を検出できなかった (reviewer-test / オーケストレータが実測: 不十分なファクトリでも完全性テストが通ることを確認)。加えて、契約テストの必須 assert が `suppress(ValueError)` を5回含む同一関数内にあり、「6個目の suppress」を足す誘惑が強い構造だった。→ (1) 必須 assert を `suppress` を一切書かない独立関数 `test_minimal_valid_input_actually_produces_a_result` へ切り出し、(2) `test_minimal_valid_input_registry_covers_all_diagnostics` で各ファクトリを実際に呼び出して `DiagnosticResult` が返ることまで検証するよう拡張。2手攻撃 (不十分なファクトリの登録 + 必須 assert の suppress) を安全な変異注入で実測し、完全性テストが独立に落ちることを確認済み。 |
| F-02-2-002 | `_iter_diagnostic_callables()` (`tests/test_diagnostics_base.py`) が `inspect.isfunction` 限定のため、D-01 が明示的に許す第2形 (パラメータ化した frozen dataclass の `__call__` インスタンス、例: 03 の `Ipc(n_surrogates=...)`) を構造的に列挙できなかった (reviewer-architecture / オーケストレータが実測: `_ParameterizedDummyDiagnostic(threshold=0.9)` は `isfunction` が `False` で列挙 0 件)。この穴により、第2形の診断が `diagnostics/` に現れた瞬間、`MINIMAL_VALID_INPUT` 登録の強制・D-15 guard・D-01 契約テストの3つが同時に静かに無効化される (D-01 の推奨に従うとその契約を守る guard が効かなくなる倒錯)。→ 列挙述語を「関数、または `__call__` の戻り値アノテーションが `DiagnosticResult` である public callable インスタンス (クラス自体は除外)」へ拡張。安全な変異注入 (`diagnostics/_tmp_second_form.py` を一時追加 → 検証 → 即削除、`git diff` で差分ゼロを確認) で、(1) 未登録なら `test_minimal_valid_input_registry_covers_all_diagnostics` が落ちる、(2) 追加引数を keyword-only 違反にすると `test_extra_diagnostic_parameters_are_keyword_only_and_do_not_overlap_ctx` が落ちる、(3) `ctx` の keyword-only マーカーを外すと `test_all_diagnostics_conform_to_d01_signature_contract` が落ちる、の3つを実測で確認。既存5診断 (すべて第1形) は引き続き列挙され、`test_diagnostic_enumeration_finds_all_known_diagnostics` は変更不要だった。 |
| F-02-2-003 | `docs/plans/rc-basics-02.md` の T2 実装メモ5 が「各実験セクションは `ESNConfig` を内包せず、必要な値だけを平らに持つ (`leak_rate`/`input_scale`/`n_units`/`density`)」のままで、round 2 で `input_scale`/`n_units`/`density`/`n_replicates` が `Esp02Config.reservoir` (`ReservoirSweepConfig`) に1本集約された後の形と食い違っていた (T3 の実装者はこの節を読む前提で YAML キーの位置を誤認しうる)。→ 実装メモ5 を「セクション固有の掃引軸だけをセクションに残し、`input_scale`/`n_units`/`density`/`n_replicates` は `reservoir` に1本」へ書き換え、集約した理由 (セクション間で N が食い違う事故を構造的に禁じ、§8 Q3 で承認された「N=200 を連載通して固定」をコードで保証する) を明記。T3 節の `EspRow` 生成の説明にも `input_scale`/`n_units`/`density` が `reservoir` 由来である旨を追記。 |
| F-02-2-004 | `tests/test_config_wiring_esp.py` の時限装置に2つの穴があった。(1) 粒度: 「新規モジュールが1本でも増えた かつ pending が1件でも残る」という条件のため、T3 が `experiment/esp.py` を作った時点で T4 (`washout`) 担当の pending まで同時に赤くなり、T3 完了〜T4 着手の間テストが緑にならなかった。(2) 解除経路: `KNOWN_EXPERIMENT_MODULES` が手書きスナップショットで実集合との突き合わせが無く、発火時に1語足せば黙って解除できた。→ (1) `WiringCase` に構造化フィールド `task` を追加し (`wiring.py`、01 側は既定値 `None` で無影響)、`TASK_STAGE_MODULES` (`T3` → `{esp, esp_pipeline}`、`T4` → `{washout}`) で段階とモジュールの対応を固定、`test_pending_cases_disappear_once_the_experiment_layer_exists` を段階ごとに判定するよう書き換え。(2) `KNOWN_EXPERIMENT_MODULES` を01時点のスナップショットとして凍結する旨を明記し、`TASK_STAGE_MODULES` のいずれかがこの集合に含まれた時点で対応する段階の pending が空であることを要求する `test_known_experiment_modules_cannot_be_widened_while_pending_remains` を新設。安全な変異注入 (`experiment/esp.py` を一時追加 → 検証 → 即削除、`git diff` で差分ゼロを確認) で、T3 の pending だけが赤くなり T4 の pending は緑のままであることを実測で確認。 |

## ID 名前空間の衝突について (F-02-1-002/004/005/015/016/017/020 の由来)

サイクル 2a の実装時、`docs/review-findings-01.md`（サイクル1の記録）が既に使い切っていた
`F-1-002` / `F-1-004` / `F-1-005` / `F-1-015` / `F-1-016` / `F-1-017` / `F-1-020` を、サイクル2a の
docstring が「今回の finding」のつもりで再利用してしまっていた（`test_finding_id_references_resolve.py`
が ID の実在しか検査しておらず、指している内容の一致までは検査していなかったため、衝突したまま
テストは緑だった）。round 2 で `F-02-<round>-<seq>` へ改名し、本文書を新設して解決した。
