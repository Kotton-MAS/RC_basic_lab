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
| F-03-1-001 | HIGH | D-24 の窓計算 (`t0 - delay` の切り出し) が MC (2箇所) / IPC (1箇所) に複製され、MC 側だけ値レベルの guard が無かった (窓を1ステップずらしても MC の22テストが全て緑のまま通ることを実測)。`CapacityProblem.lagged(series, delay)` を共有カーネルに追加し、`memory_capacity.py::_iter_delay_chunks` / サロゲート基底の切り出し、`ipc.py::_target_column` の3箇所をこれに置き換えた。カーネル側に値レベルの guard (`test_capacity_problem_lagged_matches_expected_offset`、`tests/test_diagnostics_memory_capacity.py`) を新設し、MC・IPC 両方の複製ぶんの穴を1本で閉じた。 |
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
浮動小数の丸めは変更前後でわずかに異なりうる (`Phi.T @ Phi` を1回の BLAS 呼び出しで計算する経路と、
ブロックごとに `sum` / `X.T @ X` で計算する経路は数学的に同一だが、加算順序が異なりうるため)。実測した
差分は、fixer の最終報告 (会話ログ) を参照。
