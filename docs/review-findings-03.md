# レビュー findings 記録 — rc-basics-03 サイクル3a (MC / IPC)

**この文書はコードから参照されているので削除しないこと。** `src/` と `tests/` の docstring / コメントが、
この文書に載っている finding ID（`F-03-<round>-xxx` 形式）を「なぜこうなっているか」の根拠として参照している。
`tests/test_finding_id_references_resolve.py` が、コード中に出現する全 ID が接頭辞に対応する記録文書に
実在することを機械的に検証している。参照が解決できなくなった時点でテストが赤くなる。

**新しく finding ID を docstring に書いたら、このファイルにも追記すること。**

## この文書の性格

`docs/review-findings-01.md` / `docs/review-findings-02.md` と同じ形式の「ID → 対応内容」の記録。
サイクル3a (T1: MC / T2: IPC) の round 1 レビューで出た24件 (BLOCKER 2 / HIGH 4 / MEDIUM 11 / INFO 7) を
ID と対応内容の一覧として記録する。

## サイクル 3a round 1 (`F-03-1-xxx`)

| ID | severity | 概要 |
|---|---|---|
| F-03-1-001 | HIGH | D-24 の窓計算 (`t0 - delay` の切り出し) が MC (2箇所) / IPC (1箇所) に複製され、MC 側だけ値レベルの guard が無かった (窓を1ステップずらしても MC の22テストが全て緑のまま通ることを実測)。`CapacityProblem.lagged(series, delay)` を共有カーネルに追加し、`memory_capacity.py::_iter_delay_chunks` / サロゲート基底の切り出し、`ipc.py::_target_column` の3箇所をこれに置き換えた。カーネル側の値レベル guard (`test_capacity_problem_lagged_matches_expected_offset`) に加え、`_iter_delay_chunks` (MC が実際に呼ぶ関数) 自体の出力値を固定する `test_iter_delay_chunks_matches_expected_offset` も追加した —— `lagged` 単体の正しさだけでは、呼び出し側が `lagged` に渡す `delay` を取り違えるミスまでは検出できないため (`_iter_delay_chunks` を丸ごと差し替える変異で実際に確認: `lagged` 自体は無傷のまま既存22テストが全て緑のまま通った)。 |
| F-03-1-002 | MEDIUM | `IpcConfig` の `basis`/`input_distribution` の語彙 (`LEGENDRE`/`HERMITE`/`UNIFORM`/`NORMAL`/`SUPPORTED_BASIS_PAIRS`) が `ipc.__all__` に無く、`threshold_mode` の語彙とだけ公開方針が割れていた。3b の判断に送る (3b の config 配線着手時に対応)。 |
| F-03-1-003 | MEDIUM | 仕様書 (T1 実装時に決めたこと5) は「chi2 は T2 で IPC の共有カーネルに足す」としていたが、実装は `ipc.py` の private 関数のままだった。`_chi2_threshold` を `_capacity.py` へ `chi2_threshold` として移設し、`ipc.py` から import する形に揃えた (コード側を仕様に合わせた)。 |
| F-03-1-004 | MEDIUM | 共有カーネルの境界が「線形代数」で切られており、MC/IPC が実際に複製している足場 (`_input_series`・`_validate_config` の共通検証・チャンク生成の型) がカーネル外にあった。今回は (a) `iter_column_chunks` (利用者ゼロだった公開関数) を BLOCKER 修正 (F-03-1-012) の副作用として削除、(b) `_input_series` を `input_series` として `_capacity.py` へ集約 (F-03-1-020 として実施) の2点のみ対応。`_validate_config` の共通フィールド検証とチャンク足場の統一は 3b に送る (カーネル粒度の見直しは影響範囲が大きく、3a のスコープを超えるため)。 |
| F-03-1-005 | MEDIUM | IPC の `scalars` キー集合 (`ipc_threshold_degree{d}`) が `max_delay_by_degree` の本数に依存し、既存5診断の「キー集合は固定」という慣習から外れる。3b の capacity.csv 設計時に判断する (次数依存の量を `arrays['ipc_thresholds']` へ寄せるか、固定キーに落とすかは 3b の行 dataclass 設計と合わせて決める必要があるため)。 |
| F-03-1-006 | MEDIUM | `orthonormal_basis(u_lagged, degree, distribution=UNIFORM, *, basis=LEGENDRE)` は D-28 が対で意味を持つと宣言する値の片方だけが位置引数で、自然な呼び方 `orthonormal_basis(u, 2, NORMAL)` が ValueError になる。3b に送る (呼び出し側 src 2 箇所 + tests 6 箇所の一括変更になり、3a のスコープ (findings 対応) を超える構造変更のため)。 |
| F-03-1-007 | INFO | `diagnostics.memory_capacity` / `diagnostics.ipc` はモジュール名と公開関数名が同名で、属性アクセス経由のモジュール参照が関数に解決される。対応不要 (改名は仕様が指定する名前のため提案しない)。 |
| F-03-1-008 | INFO | D-23 の依存境界・D-01・D-13 が実測で守られていることの確認結果。対応不要。 |
| F-03-1-009 | HIGH | `docs/plans/rc-basics-03.md` が 3a のスコープに明記する `docs/design.md` §11.1 (容量測定の定義と正規化) が未着手だった。§11.1 を追記した (t0 の決め方 / Gram 共有カーネル / D-24 の単一基準点 / D-25 の固定 alpha / D-28 の正規直交化)。既存 §1〜§10 と同じ「実測値は一次資料 (コード) と機械照合する」書き振りに揃え、既定値表は `tests/test_design_doc.py` の既存パラメータ化テストがそのまま照合する。 |
| F-03-1-010 | MEDIUM | `docs/plans/rc-basics-03.md:425` の T2 節導入文だけ「(500 tests 緑)」とテスト件数を書いており、T1 節・既存の規律 (件数を書かない) と食い違っていた。削除して T1 節と同じ文言に揃えた。 |
| F-03-1-011 | INFO | D-23 rationale の「旧 guard は改名済みで比較対象が HEAD に残っていない」という記録。対応不要 (今後は比較対象のコミットを明記する運用にする)。 |
| F-03-1-012 | BLOCKER | `surrogate_threshold` が `(n_samples, n_base * n_surrogates)` を一括確保しており、IPC 既定 (T=1e6, N=50) で peak RSS 6.0〜6.5GB (予算4GB を50〜60%超過)。シャッフル列を `chunk_size` 列ずつ生成する `_iter_surrogate_chunks` に置き換え、生成順序を chunk_size に依存しない形(単調増加index)に固定して `test_chunk_size_does_not_change_results` を壊さずに一括確保をやめた。あわせて F-03-1-014 の `bounded_chunk_size` も適用 (詳細はそちらを参照)。 |
| F-03-1-013 | BLOCKER | `CapacityProblem.from_states` が `phi = concatenate((ones, X))` で `X` と同じ大きさのコピーをもう1枚作っており、MC 本番 (T=1e6, N=200) で peak RSS 5.0〜6.9GB。バイアス列は定数列なので Gram を `[[T_eff, sum(X,0)], [sum(X,0).T, X.T@X]]` にブロック分解し、`Phi` を一度も実体化しない設計に変更 (`CapacityProblem.x` が `X` のビューを持つのみ)。`capacity_of_targets` の `rhs` も `[sum(Z,0); X.T@Z]` に変更し、`X` に触れるのは `X.T@Z` の1回だけに保った。`test_capacity_of_targets_touches_phi_exactly_once` は `problem.phi` が無くなったため `test_capacity_of_targets_touches_x_exactly_once` (「`X` に触れるのは1回」を検査) へ書き換え、guard の強度を維持した。 |
| F-03-1-014 | MEDIUM | `chunk_size` 既定値 256 は T=1e6 では1チャンク 2.05GB になり大きすぎた。F-03-1-012/013 の BLOCKER 修正 (Phi 実体化の除去・サロゲートのチャンク生成) だけでは不十分だった: 実測で IPC 6.25GB / MC 未計測のまま残ることを発見したため、`_capacity.bounded_chunk_size(configured, n_samples)` を追加し、`IpcConfig.chunk_size` / `MemoryCapacityConfig.chunk_size` の**既定値そのもの**は変えずに、実際に使うチャンク列数を「1チャンクが128MiBを超えない」という閉形式の上限で T_eff に応じて下げる形にした (結果は変わらない性能パラメータのまま、D-26)。チャンク境界で一時的に2チャンク分が同時生存する generator の構造まで含めて実測で調整した。 |
| F-03-1-015 | INFO | D-26 の「solve 回数は目標数に比例しない」が実測で成立していることの確認結果。対応不要。 |
| F-03-1-016 | MEDIUM | `max_targets` が `ipc_heatmap` の確保サイズ (`n_degrees x max(max_delay_by_degree)`) を縛っておらず、目標数 20万で検査を通過したまま heatmap が 1.59GB になりうる (CWE-789)。`_validate_config` に「heatmap のセル数が `max_targets` を超えたら ValueError」という確保前・閉形式の検査を追加した。 |
| F-03-1-017 | INFO | D-23 プローブが実行コードを文字列置換 (`.replace`) で組み立てている (現状は注入不成立だが将来のリスク)。対応推奨に従い、`sys.argv` 経由 (`subprocess.run([..., json.dumps(...)])` + プローブ側 `json.loads(sys.argv[1])`) に変更した。 |
| F-03-1-018 | HIGH | `ipc()` 本体が133行 (既存最長 `conditional_lyapunov` の116行を超過)。集約ループを `_aggregate_by_cell`、scalars 組み立てを `_build_scalars`、params 組み立てを `_build_params` へ切り出し、110行に短縮した。 |
| F-03-1-019 | MEDIUM | `TargetSpec = tuple[tuple[int, int], ...]` が旧来の代入形式で、リポジトリの慣習 (`type` 文) から外れていた。`type TargetSpec = tuple[tuple[int, int], ...]` に書き換えた。 |
| F-03-1-020 | MEDIUM | `_input_series` が `ipc.py` / `memory_capacity.py` にほぼ同一の実装で複製されていた。`_capacity.py` に `input_series(u, *, diagnostic)` として集約し、両モジュールから呼ぶ形にした。 |
| F-03-1-021 | INFO | `test_diagnostics_ipc.py` の2つの parametrize ブロックに `ids=` が無い。対応不要 (今回のスコープ外、3b で他の parametrize と合わせて整理する候補)。 |
| F-03-1-022 | INFO | `_degree_thresholds` / `surrogate_threshold` の引数が目安の5個を超える。対応不要 (キーワード専用なので実害は小さいという reviewer の評価どおり、優先度は低いため見送り)。 |
| F-03-1-023 | HIGH | `test_mc_total_does_not_exceed_n_units` に下限チェックが無く、容量を0.02倍に潰しても全アサーションが通ることを実測で確認 (IPC の `saturation_ratio >= 0.5` に相当する対策が MC 側に無かった)。`mc_ratio >= 0.2` の下限チェックを追加した (実測ベースライン rho=0.5/0.9/0.99 で 0.376/0.620/0.653)。 |
| F-03-1-024 | MEDIUM | `test_mc_profile_lengthens_with_spectral_radius` の docstring「比の最小は1.75」は5シードだけの観測で、62シードの実測 (最小1.559、閾値1.5に対し約4%の余裕) より楽観的だった。docstring を実測値に訂正した (テスト自体は62シード中失敗なしのため閾値は変更せず)。 |

## BLOCKER 修正の設計判断について

F-03-1-012 / F-03-1-013 は D-26 (「(T,K) を実体化しない」原則) から外れていた2箇所を、原則に揃える形で
修正した。`CapacityProblem` が `phi` (設計行列) ではなく `x` (状態のビュー) と `gram` (ブロック分解した
Gram) だけを持つようになったことに伴い、`capacity_of_targets` が `Phi` に一度も触れない形へ変わっている。

**実測した条件下では bit-for-bit 一致したが、アルゴリズム的に保証されているわけではない
(F-03-2-007)。** 旧実装 (`Phi = concatenate((ones, X))` を実体化し `Phi.T @ Phi` を1回の BLAS
呼び出しで計算) と新実装 (`CapacityProblem.from_states` のブロック分解: `gram_xx = X.T @ X` と
バイアス行 `column_sums = np.sum(X, axis=0)` を別々に計算して `gram` に埋める) を独立な標準正規乱数
`X ~ N(0, I)` (T=4000 と T=1,000,000、いずれも N=200) で比較すると `np.array_equal(gram_old,
gram_new)` は両方とも `True` だった (再現: `/tmp` の使い捨てスクリプトで
`gram_old = np.concatenate((ones, X), axis=1); gram_old = gram_old.T @ gram_old` と
`CapacityProblem.from_states(X, t0=10).gram` を突き合わせ)。ただしこれは「ブロック分解は常に
数値的に安全」を意味しない —— 分解に使う部分演算 `ones.T @ X` (BLAS matmul 経由) と
`np.sum(X, axis=0)` は一般には加算順序が異なり、独立な標準正規乱数でも最大絶対誤差が
`4.18e-12` (T=50,000) から `7.37e-11` (T=999,600) までTに応じて増加することを実測した
(`np.abs((np.ones((1, T)) @ X) - np.sum(X, axis=0, keepdims=True)).max()`、`seed=0`・`N=200`
の標準正規乱数。performance reviewer は別データ・別条件で最大 `1.66e-10` を報告しており、
オーダーは一致する)。乱数シード・生成条件・numpy/BLAS のバージョンに依存する目安値であり、
別条件で再実行すると桁は一致してもこの数値そのものは再現しない
(F-03-3-012: 同じ `seed=0`・`N=200` で再実行すると `5.12e-12` / `1.01e-10` になり、
オーダーは一致するが厳密には一致しない)。今回の
`from_states` 経路の入力 (低ランク信号+微小ノイズが典型) ではたまたま一致しただけで、データ形状・
numpy/BLAS バージョン・プラットフォームが変われば丸め誤差レベルでの乖離がありうる。

## サイクル 3a round 2 (`F-03-2-xxx`)

round 1 の BLOCKER 修正 (F-03-1-012/013) の副産物として fixer が指示に無いまま追加した
`bounded_chunk_size` を、4つの reviewer が独立に突いた回 (BLOCKER 1 / HIGH 2 / MEDIUM 10 / INFO 7)。

| ID | severity | 概要 |
|---|---|---|
| F-03-2-001 | HIGH | `bounded_chunk_size` の実効値が `params['chunk_size']` に反映されず (設定値をそのまま記録)、D-26 guard の閉形式 (`ceil(K/chunk_size)`) が本番規模 (T=200000) で崩れていた (期待2回、実際4回)。適用も `ipc.py` 2箇所・`memory_capacity.py` 1箇所に手書きで複製されていた。`CapacityProblem.effective_chunk_size(configured)` を追加して3箇所の複製を解消し、`params['chunk_size_effective']` を追加、D-26 の guard (`test_gram_solve_count_does_not_scale_with_target_count`) にキャップが実際に発動する規模のケースを足して閉形式を `ceil(K/effective_chunk_size)` に更新した。 |
| F-03-2-002 | MEDIUM | `_MAX_CHUNK_BYTES=128MiB` と「設定値は変えず実効値だけ内部で縛る」判断が decisions.yaml にも design.md にも plan doc にも記録されていなかった。`.claude/decisions.yaml` に決定 (round3 で D-33 へ改番、F-03-3-001) を追加し、`docs/design.md` §11 の既定値表に `_MAX_CHUNK_BYTES` / `max_degrees` の行を足した。 |
| F-03-2-003 | MEDIUM | `CapacityProblem.x` が呼び出し側 `X` のビューであるため、`from_states` 後に `X` を書き換えると `gram` と desync し、例外も警告もなく誤った容量が返る (実測: 容量 1.2553668e+08)。`CapacityProblem` の docstring (クラス docstring と `x`/`gram` の Attributes) に「`from_states` 後に元の `X` を書き換えてはならない」契約を明記した。 |
| F-03-2-004 | MEDIUM | `ipc.py` の `__all__` に `LEGENDRE`/`HERMITE`/`UNIFORM`/`NORMAL`/`SUPPORTED_BASIS_PAIRS` (既に import 済み) が無く、`threshold_mode` 側の語彙とだけ公開方針が割れていた。3b を待たず `__all__` に5シンボルを追加した (F-03-1-002 の3b送りを round 2 で前倒しで解消)。 |
| F-03-2-005 | INFO | `CapacityProblem.lagged` が窓計算専用でテストがダミー状態行列を構築している件。reviewer 自身が「対応不要、3b で `RowAlignment` 切り出しを検討」と結論。対応不要。 |
| F-03-2-006 | INFO | F-03-1-001 (窓計算の複製解消) が実効であることの確認結果 (変異注入3種で検出、MC/IPC の基準点一致を確認)。対応不要。 |
| F-03-2-007 | MEDIUM | `docs/review-findings-03.md` (旧版) が浮動小数の丸め差分の根拠を「fixer の最終報告 (会話ログ)」としており、リポジトリ内の成果物ではなく再現不能だった。実測値 (`np.array_equal` による bit-for-bit 一致の確認、`ones.T@X` vs `np.sum` の丸め差分) を本文に直接書き、再現コマンドも残した。あわせて「丸めはわずかに異なりうる」という記述が実測 (bit-for-bit 一致) より弱かった点も「実測した条件下では一致したが、アルゴリズム的に保証されているわけではない」に訂正した (F-03-2-012 も同時に解消)。 |
| F-03-2-008 | INFO | `bounded_chunk_size` だけ Args/Returns セクションが無かった。追加した (安価な修正のため対応)。 |
| F-03-2-009 | MEDIUM | `chunk_size` を大きく明示指定しても無条件に切り下げられ、`params` にも切り下げ後の値が記録されないため利用者が気づけなかった。F-03-2-001 の `chunk_size_effective` で解決し、`bounded_chunk_size` の docstring に「大きい方向の意図は保護されない」旨を明記した。 |
| F-03-2-010 | INFO | BLOCKER 2件 (F-03-1-012/013) の解消を reviewer が独立に実測確認した結果 (IPC 0.93GB / MC 2.11GB)。対応不要。 |
| F-03-2-011 | INFO | 実行時間の悪化なし (むしろ約7%高速) の確認結果。対応不要。 |
| F-03-2-012 | INFO | ブロック分解後の bit-for-bit 一致と、`ones.T@X` vs `np.sum` の丸め差分 (最大 1.66e-10) の実測。F-03-2-007 の文書修正に統合して対応した。 |
| F-03-2-013 | MEDIUM | CWE-789。`max_targets` / heatmap のセル数検査は次数の本数 (`len(max_delay_by_degree)`)、延いては `psi_table` (次数 x 系列長) の確保サイズを縛っていなかった (実測: n_degrees=1400, T=200000 で psi_table 単独 peak RSS 2.69GB)。`IpcConfig.max_degrees` (既定20) を追加し、`_validate_config` で確保前に検査するようにした。 |
| F-03-2-014 | MEDIUM | CWE-400。`count_targets` の閉形式が `max_variables` の上限を持たず、大きい値では多倍長整数の組合せ計算がハングしうる (実測: D=V=4000 で 373.73s)。`_validate_config` に `max_variables` の独立な安全上限 (20) を追加し、`count_targets` にも `n_vars > max_delay` の早期打ち切りを追加した。 |
| F-03-2-015 | MEDIUM | CWE-789。BLOCKER 修正でサロゲート列の生成はチャンク化されたが、その入力である代表目標行列 `base` は対象外で `chunk_size=1` を指定しても効かなかった (実測: K=400, T=1e6 で base 単独 peak RSS 3.23GB)。`_degree_thresholds` で `picked` (代表目標の index) 自体も `chunk_size` と同じ予算で分割し、`base` を一括確保しない形にした。 |
| F-03-2-016 | MEDIUM | `capacity_of_targets()` の docstring 本文が phi→x のブロック分解 (F-03-1-013) 前の実装を説明したまま残っていた。モジュール先頭 docstring やコード内コメントと同じ書き振りに揃えた。 |
| F-03-2-017 | INFO | `_aggregate_by_cell` 等4関数が引数6個で目安の5個を超える件。reviewer 自身が「急いで直す必要はない、3b で検討」と結論。対応不要。 |
| F-03-2-018 | BLOCKER | `bounded_chunk_size` を no-op に差し替えても既存58テストが1件も落ちない空虚な安全機構だった (テスト規模では切り詰め分岐が一度も真にならないため)。直接の単体テスト (configured<budget で無変更 / configured>budget で切り詰め / n_samples<=0 の防御分岐 / 下限 max(1,...)) を追加し、`test_chunk_size_does_not_change_results` にキャップが実際に発動するケースを1件追加した。in-process で `bounded_chunk_size` を no-op 化すると5テストが失敗することを確認済み。 |
| F-03-2-019 | HIGH | `mc_ratio >= 0.2` は rho=0.5 では約48〜50%喪失で検出できるが rho=0.9/0.99 では約65〜70%喪失まで通過し、IPC の `saturation_ratio >= 0.5` (約30%喪失で検出) より明確に緩かった。rho ごとの個別下限 (`_MC_RATIO_LOWER_BOUND = {0.5: 0.25, 0.9: 0.4, 0.99: 0.4}`) に変更し、実測で破断点が33%/36%/39%喪失になることを確認した (IPC と同水準の30〜45%レンジ)。 |
| F-03-2-020 | MEDIUM | `CapacityProblem.lagged` の `values.ndim != 1` 分岐がテストで一度も実行されず coverage で Missing だった。`test_capacity_problem_lagged_rejects_multi_dimensional_series` を追加した。 |

## サイクル 3a round 3 (`F-03-3-xxx`)

round 2 の修正 (D-29 の追加、`picked` のブロック化、`max_degrees` /
`_MAX_VARIABLES_FOR_COUNT` の新設) 自体が新たな問題を生んだ回
(BLOCKER 1 / HIGH 3 / MEDIUM 8 / INFO 12)。BLOCKER はオーケストレータ
(fixer を起動した側) の指示ミスによる ID 衝突で、fixer の実装ミスではない。

| ID | severity | 概要 |
|---|---|---|
| F-03-3-001 | BLOCKER | 新設 D-29 (chunk_size のキャップ) の ID が、ユーザー承認済みの `docs/plans/rc-basics-03.md` line 9 が 3b 用に予約した D-29〜D-32 (line 271 に D-29 = NARMA10 が既に記録済み) と衝突していた。オーケストレータが予約を確認せず番号を指定した指示ミス。この決定を **D-33** に改番し、`design.md` / `ipc.py` / `_capacity.py` のコメント / `review-findings-03.md` の D-29 参照を D-33 へ置換した。3b 用の D-29〜D-32 は plan doc のまま動かしていない。`check_decisions.py` が OK になることを確認した。 |
| F-03-3-002 | HIGH | F-03-2-015 (`picked` のブロック化) の副作用で、IPC が `surrogate_threshold` の第1戻り値を `_` で捨て `float(np.quantile(...))` を自前に計算しており、D-27 の「サロゲートは同じ関数に流す」前提が壊れていた (実測: カーネル側の閾値を NaN にしても IPC の出力は不変、MC は nan に伝播)。`_capacity.py` に `surrogate_capacities(problem, base_blocks, alpha, ...)` を切り出し、`surrogate_threshold` をその上に定義し直した。`base_blocks` は Iterable of block になり、`ipc.py` は `picked` のブロックを生成する `_picked_target_blocks` を渡すだけになった (`np.quantile` は `ipc.py` から消えた)。MC は `[base]` の1要素 Iterable を渡すだけで既存の呼び方を維持した。修正後、カーネル側の閾値を汚染すると IPC も MC も影響を受けることを実測で確認した。 |
| F-03-3-003 | HIGH | `max_degrees=20` と `_MAX_VARIABLES_FOR_COUNT=20` の根拠が decisions.yaml にも plan doc にも無く、`design.md:761` は存在しない決定 (D-29 の rule は chunk_size のキャップのみ) を参照していた。`_MAX_VARIABLES_FOR_COUNT` の docstring/テストが根拠にした実測ケース (`max_delay_by_degree=(1,)*4000`) は、次数の本数 (4000) が既定 `max_degrees=20` の検査に先に捕まり到達不能だった。判断を **D-34** として guard_test 付きで記録し、`design.md` の ID を D-34 に正した。`_MAX_VARIABLES_FOR_COUNT` の docstring を、`_MAX_DEGREES=32` (F-03-3-019 で追加) の下では `count_targets` が `max_variables` の値によらず実測上つねに 1ms 未満に収まること (最悪ケース: 次数32・遅延10^300・`max_variables=32` でも 0.3ms) を明記し、到達不能な D=4000 のケースを根拠から外した。維持理由は (a) 目標1本の変数本数として意味のある値域の明示、(b) `_MAX_DEGREES` が将来引き上げられた場合の多重防御、の2点とした。 |
| F-03-3-004 | MEDIUM | D-29 (→D-33) の rule (iii)「params に設定値と実効値の両方を記録する」を守るテストが0件で、`chunk_size_effective` を params から削除しても落ちるテストが無かった (実測: 削除しても 522 passed で変化なし)。`test_params_record_configured_and_effective_chunk_size_when_capped` を追加し、D-33 の guard_test をこちらへ差し替えた。 |
| F-03-3-005 | MEDIUM | 確保・列挙を縛る上限が `max_targets` (設定フィールド、単位の違う2量を同じ値で縛る) / `max_degrees` (設定フィールド、絶対上限あり) / `_MAX_VARIABLES_FOR_COUNT` (モジュール定数) の3形に割れており、線引きが正本に無かった。`design.md` §11 に「引き上げ可能な実験パラメータ」と「引き上げ不可の安全装置」の線引きを1文追加し、`max_degrees` の docstring に `_MAX_DEGREES` への相互参照を足した。`max_targets` の単位の食い違いの解消は3bに送った。 |
| F-03-3-006 | MEDIUM | `problem.x` が書き込み可能なビューのままで、`problem.x[...] = ...` と書いても素通りし `gram` と無言で desync する経路が残っていた (src 内に代入は0件で安全に閉じられる)。`CapacityProblem.from_states` で保持するビューを `writeable=False` にし、`test_capacity_problem_x_is_read_only` を追加した。元の `X` 自身への書き込みを塞ぐのは3bの受け入れ条件に送った。 |
| F-03-3-009 | MEDIUM | `.claude/decisions.yaml` の D-26 の rule が `ceil(K/chunk_size)` のまま (effective_chunk_size 導入前) で、直後の D-33 (旧D-29) の rationale が「D-26 の閉形式を `ceil(K/effective_chunk_size)` に更新した」と書いており正本内で自己矛盾していた。D-26 の rule を `ceil(K/chunk_size_effective)` に更新した。 |
| F-03-3-010 | MEDIUM | `docs/plans/rc-basics-03.md` に `bounded_chunk_size` / `effective_chunk_size` / `chunk_size_effective` / `max_degrees` / `picked` のブロック分割への言及が0件で、line 136 も旧閉形式のままだった。「T2 実装時に決めたこと」に項目17・18を追加し (D-33/D-34 の経緯を記録)、line 136 の閉形式を `ceil(K/chunk_size_effective)` に更新した。 |
| F-03-3-018 | MEDIUM | CWE-400。`count_targets` / `enumerate_targets` は `ipc.__all__` の公開関数だが `_validate_config` を呼ばないため、`max_delay` を大きくすると D=1600 で 107.89s ハングする (round2 の PoC 形は消えたが形状特化の修正だった)。`_validate_config` から組合せ計算量の検査だけを `_validate_combinatorial_bounds` として切り出し、`count_targets` の先頭で呼ぶようにした (`max_targets`/heatmap の検査は含めず、`count_targets` を目標数の下見に使う既存の呼び出し側の意味を変えないようにした)。修正後、病的な cfg を `count_targets` に直接渡しても1秒未満で `ValueError` になることを実測した。 |
| F-03-3-019 | MEDIUM | CWE-789。`max_degrees` が上限なしの設定フィールドで、`max_degrees=400` の1行変更だけで psi_table 膨張が再現した (実測 peak RSS 0.095GB→0.721GB)。同一クラスの脆弱性で `max_variables` (上書き不能なモジュール定数) と防御強度が非対称だった。`_MAX_DEGREES=32` のモジュール定数を追加し `_validate_config` で検査するようにした。修正後、`max_degrees` を大きくしても psi_table が膨らまないことを実測で確認した。 |
| F-03-3-023 | MEDIUM | D-26 のソルブ回数テスト (`_expected_solve_count`) が本番の `bounded_chunk_size` を直接呼んで期待値を計算しており、`bounded_chunk_size` 自体にバグがあると両辺が同時に動いて素通りする (実測: バイト予算計算を4倍過剰に厳しくする変異を注入してもこのテスト単体では1 passed)。docstring に「capping の正しさはこのテストの担保範囲外で `bounded_chunk_size` の直接単体テストが担う」旨を明記した。 |
| F-03-3-022 | HIGH | `max_variables` の新設上限 (`_MAX_VARIABLES_FOR_COUNT=20`) の invalid_configs は下限違反 (0) と上限違反 (21) しかカバーせず、成功すべき境界値 20 のテストが無かった (実測: off-by-one 変異で40テスト全通過)。`test_max_variables_boundary_value_20_is_accepted` を追加した。 |
| F-03-3-007 | INFO | `picked` のブロック幅が `chunk_size` (実効値) を流用しており、代表目標行列の確保幅と1回の solve に畳む列数という別の軸が隠れて結合している。対応不要 (3b で `surrogate_capacities` 切り出しと同時にやると追加コストがほぼ無いため、3bへ送る)。 |
| F-03-3-008 | INFO | `__all__` への5シンボル追加で3bの config層がbasis語彙を公開モジュール経由で取れるようになったことの確認結果 (悪化なし)。対応不要。 |
| F-03-3-011 | INFO | D-29 (→D-33) rationale の「no-op 化で58テストが1件も落ちない」が round2 開始時点のスイートに対する主張で、read-only agent の権限では裏取りできなかった件。対応不要 (次に検証する reviewer が git worktree の使える権限で確認するとよい、との reviewer 自身の結論)。 |
| F-03-3-012 | INFO | `docs/review-findings-03.md` の丸め誤差の数値 (4.18e-12 / 7.37e-11) にシードと N の指定が無く再現できなかった (実測: seed=0・N=200 では 5.12e-12 / 1.01e-10)。本文にシード・N の明記と「目安値であり再実行では厳密には再現しない」旨を追記した。 |
| F-03-3-013 | INFO | `picked` ブロック分割は既定 `n_surrogate_targets=4` が実効 chunk_size を常に下回るため実運用では1ブロックしか実行されず、round2 で懸念したサロゲート経路の悪化は測定上発生していないことの確認結果。対応不要。 |
| F-03-3-014 | INFO | round2 の BLOCKER 修正由来の4GB予算が round3 のリファクタで崩れていないことの確認結果 (IPC 0.906GB / MC 2.09GB)。対応不要。 |
| F-03-3-015 | INFO | 3b (T3) の代表条件 (3-A/3-B/3-B') を個別実測し、性能予算に収まることを確認した結果 (合計約161秒)。対応不要。 |
| F-03-3-016 | INFO | `effective_chunk_size` の呼び出し回数がホットパスに比例しないことの確認結果。対応不要。 |
| F-03-3-017 | INFO | `count_targets` の早期打ち切りが既定設定の計算コストを変えないことの確認結果 (601、変化なし)。対応不要。 |
| F-03-3-020 | INFO | `n_surrogates` × `n_surrogate_targets` が他のどの上限とも突き合わされていないが、増え方が厳密に線形で OOM 経路にはならないことの確認結果。急ぎの対応は不要 (3b で閉形式の検査を1本足すと確保・計算軸の規律が揃う、との reviewer の提案は3bへ送る)。 |
| F-03-3-021 | INFO | `enumerate_targets` の `max_targets` 超過メッセージが多倍長整数 (`total`) を f-string で直接文字列化しており、Python 3.11+ の int→str 変換桁数制限 (既定4300桁) を超えると別の `ValueError` に化けて運用者に意図したメッセージが届かない (実測: D=1600 で 'Exceeds the limit (4300 digits)...')。安価な修正のため対応: `_format_target_count` を追加し、`bit_length` から桁数を見積もって上限に近ければ概算の指数表記 (`~1eN`) に落とすようにした (`float(total)` も `OverflowError` を起こしうるため使わない)。 |
| F-03-3-024 | INFO | `_MC_RATIO_LOWER_BOUND` が固定シード (seed=7) のベースラインでのみ調整されており、他シードではマージンが約25%まで縮む (フレーキーではない) ことの確認結果。対応不要 (reviewer 自身が「将来シードを変える場合は破断点を再測定するコメントを残すとよい」と結論、コメント追加は3bへ送ってよい低優先度)。 |
