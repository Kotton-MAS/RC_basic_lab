# レビュー findings 記録 — rc-basics-lab サイクル3b-1 (03: 記憶容量 T3 の仕上げ)

**この文書はコードから参照されているので削除しないこと。** `src/` と `tests/` の docstring / コメントが、
この文書に載っている finding ID を「なぜこうなっているか」の根拠として参照している。
`tests/test_finding_id_references_resolve.py` が、コード中に出現する全 ID が接頭辞に対応する記録文書に
実在することを機械的に検証している。参照が解決できなくなった時点でテストが赤くなる。

**新しく finding ID を docstring に書いたら、このファイルにも追記すること。**

## この文書の性格

`docs/review-findings-01.md` / `-02.md` / `-03.md` と同じ形式の「ID → 対応内容」の記録。
サイクル 3b-1 (docs/plans/rc-basics-03b.md) の round 1 レビューで出た25件
(BLOCKER 0 / HIGH 4 / MEDIUM 16 / INFO 5) を ID と対応内容の一覧として記録する。

## ID 採番規則について (`3b1` セグメントの扱い)

`tests/test_finding_id_references_resolve.py` の ID 解決規則は、サイクル修飾形式
`F-<cycle>-<round>-<seq>` の `<cycle>` セグメントを元々 `[0-9]+` (数字のみ) で
定義していた。今回の一時集約フォルダ (`.claude/tmp/findings/3b1-round-1/`) が
振った ID の `<cycle>` セグメントは `3b1` で、数字だけの正規表現にはマッチしない。

**判断: ID は `F-3b1-1-NNN` のまま (fixer-input.json / triage.json との対応を保つ
ため変更しない) にし、テスト側の規則を拡張した。** 変更点は2つ:

1. `_QUALIFIED_ID_PATTERN` の `<cycle>` セグメントを `[0-9]+` から `[0-9A-Za-z]+`
   へ広げ、英数字混在のサイクルラベル (`3b1`) を受理できるようにした
   (数字のみだった既存サイクル (`01`/`02`/`03`) の解決には影響しない)。
2. サイクルラベルから記録文書名への変換に `_CYCLE_LABEL_TO_DOC_SUFFIX` という
   別名テーブルを追加し、`3b1` を `03b` (この文書のファイル名の由来。
   `docs/plans/rc-basics-03b.md` と同じ命名規則) に変換してから
   `review-findings-<suffix>.md` を組み立てるようにした。

この別名テーブルが要るのは、一時集約フォルダの命名 (レビュー実行時に機械的に
決まる、ラウンドを跨いだ再実行のたびに変わりうる) と、記録文書の永続的な命名
規則 (01/02/03 に続けて 03b) が異なるため。記録文書を `review-findings-3b1.md`
にリネームして一時フォルダの命名に揃える選択肢もあったが、01/02/03 の連番規則
から外れるため採らなかった。

## サイクル 3b-1 round 1 (`F-3b1-1-xxx`)

| ID | severity | 概要 |
|---|---|---|
| F-3b1-1-001 | HIGH | D-36 の rule『ESN.run には常に rng を渡す』が `esp.py` の2箇所のうち `simulate_reference_trajectory` 側にしか実装されておらず、`simulate_condition` の比較軌道ループ (02 専用) は rng 無しで呼ばれたままだった (`ReferenceTrajectory` が rng を外へ出さないリーキーな抽象化のため)。`ReferenceTrajectory` に `rng` フィールドを1本足し、比較軌道ループを `reference.esn.run(reference.drive, x0=x0, rng=reference.rng)` に直して『常に』を文字通り満たした。`state_noise=0` では乱数を1個も引かないため 02 の成果物はバイト不変 (既存 guard_test で確認済み)。`tests/test_experiment_esp.py::test_simulate_condition_always_passes_rng_to_every_esn_run` を追加し、`ESN.run` を monkeypatch して参照軌道1回+比較軌道 n_pairs 回のすべてが `rng is not None` であることを直接固定した (この行を revert すると同テストが赤くなることを実測済み)。D-36 の rationale にこの修正の経緯を追記した。 |
| F-3b1-1-002 | MEDIUM | `CapacityOutcome.ipc_by_degree` が書き込み専用の死んだ payload だった (全 outcome を -999 で埋めても成果物はバイト一致することを実測)。`CapacityOutcome` から `ipc_by_degree` フィールドを削除し、`n_degrees` の算出は `evaluate_capacity_condition` 内のローカル変数のまま行う形にした。`CapacityOutcome` / `profile_rows` の docstring に、フィールドとして運ばない理由 (図の消費側は `CapacityRow.ipc_linear`/`ipc_nonlinear` と `mc_profile`/`ipc_heatmap` しか読まない) を明記した。 |
| F-3b1-1-003 | MEDIUM | D-38 の rule『規準は n_targets_kept と同一にする』と `CapacityProfileRow` docstring の『しきい値を超えたことと行が在ることが1対1に対応する』が、実装が満たしていない不変条件 (行数 == n_targets_kept) を主張していた (実測: 本番成果物117行すべてで行数 != n_targets_kept、例: 81セル vs 297目標)。実装が実際に満たしている不変条件は『capacity 列の総和が ipc_total/mc_total と一致する』こと (実測: 117行すべてで一致、誤差0件)。D-38 の rule/rationale と `CapacityProfileRow` / `profile_rows` の docstring をこの実測に合わせて訂正し、guard_test (`test_profile_csv_columns_are_static_and_cells_are_positive`) に総和一致の検算 (MC/IPC 両方、`math.isclose`) を追加した。 |
| F-3b1-1-004 | MEDIUM | `evaluate_capacity_condition` が軌道生成/read-only化/2診断呼び出し/行組み立ての4つを1関数に閉じており、外部生成の `X` (3-C は 01 の `run_task` が作る) から `CapacityRow` を作る接ぎ目が無い。3b-2 の T4 が確実に踏む。今回は割らず、`docs/plans/rc-basics-03b.md` §3.3 の「3a から持ち越した設計課題」に項目8として追加し、T4 冒頭で `measure_capacity(states, u, *, ctx, mc_cfg, ipc_cfg)` + `capacity_row_from(mc, ipc, *, ...)` の2段に割る別タスクとして切り出すよう明記した。 |
| F-3b1-1-005 | MEDIUM | `ConservationConfig` の2つの上書きフィールド (`max_delay_by_degree` / `n_replicates`) が別々の継承規則 (無条件上書き / `None` のときだけ継承) なのに docstring は『同じ片方向の上書き』とだけ書いており、規則が同一であるかのように読めた。`n_replicates` の docstring を書き直し、規則が異なること (打ち切りは 3-B' の定義そのものなので常に効く必要がある、レプリケート数は予算超過時の縮退のノブなので既定では 3-A/3-B と同じ信頼度を保ちたい) とその理由を明記した。 |
| F-3b1-1-006 | MEDIUM | `reservoir_config_for` が `ReservoirSweepConfig.n_replicates` に横断共有値 (`config.reservoir.n_replicates`) を無条件で詰めており、3-B' で `conservation.n_replicates` が上書きされている場合に実効値と食い違った設定オブジェクトが境界を越えていた (実測: 実効1 に対し3が渡る)。`drive_config_for` が同じ規律で `n_pairs` を外しているのに非対称だった。`n_replicates_for(config, condition.experiment)` の実効値を渡すよう修正し、`tests/test_capacity_pipeline.py::test_reservoir_config_for_carries_the_effective_replicate_count` で固定した (修正前に戻す変異で赤くなることを実測済み)。 |
| F-3b1-1-007 | MEDIUM | `representative_leak_rate(rows, key: str)` が `getattr(row, key)` で列名を文字列指定しており、mypy が `getattr` の戻り値を `Any` とみなすため列名のタイプミスが型検査を素通りしていた。引数を `value_of: Callable[[CapacityRow], float]` に変え、呼び出し側 (`plot_mc_sweep` / `plot_ipc_profile`) を `lambda row: row.mc_total` / `lambda row: row.ipc_total` に書き換えた。存在しない列名を直接属性アクセスで書くと mypy が `attr-defined` で落とすことを実測で確認 (`row.nonexistent_column` を試すと `error: "CapacityRow" has no attribute "nonexistent_column"`)。`getattr` 経由の呼び出しが残っても実行時に `AttributeError` になることを `tests/test_plotting_capacity.py::test_representative_leak_rate_rejects_a_nonexistent_column_via_getattr` で固定した。 |
| F-3b1-1-008 | MEDIUM | `experiments/03_capacity/config.yaml:153` のコメントが『3-C の ESN は N=50 (D-39)』と書いた直後で `n_units: 200` (`esn_mackey_glass` も 200) になっており、承認済みの決定 (D-39、仕様 §8) と正反対の値がコメント付きでコミットされていた。3-C は 3b-2 の T4 が配線するため今は不活性だが、T4 がこの YAML をそのまま読むと N=200 で回ってしまう。コメントを『D-39 は N=50 と決まっているが、この n_units はまだ合わせていない (T4 が配線するまで既定のまま)。T4 で 50 に変更すること』という値と矛盾しない表現に書き換えた (値そのものは 04/3b-2 の T4 が変更する)。 |
| F-3b1-1-009 | INFO | `rc_basics_lab.plotting` を最初に import すると循環 import (`plotting.figures` -> `experiment.runner` -> `experiment/__init__` -> `pipeline` -> `plotting.figures`) で `ImportError` になる既存負債 (base-ref でも再現、3b-1 は同じ形の辺を2本増やしただけで新しい循環を作っていない)。対応不要 (今回のスコープ外)。`docs/plans/rc-basics-03b.md` §10「スコープ外として触らなかったもの」に、04 冒頭の `config.py` package 化と同じ回で扱う別タスク候補として明記した。 |
| F-3b1-1-010 | HIGH | README.md 冒頭が『本リポジトリはサイクル01と02の範囲を実装している』と述べたままで、`results/03_capacity/` に CSV3本・PNG4枚・meta.json が既にコミットされ `main.py --experiment 03` / `make figures-03` / `make saturation-03` が使えるにもかかわらず、03 への言及が0件だった。範囲宣言を03まで含む形に訂正し、`## 実験03: 記憶容量 (MC / IPC)` セクション (make コマンド・成果物一覧の表) を追加、リポジトリ構成の一覧にも `experiments/03_capacity/` / `results/03_capacity/` を追記した。詳しい数値考察は 3b-2 (T5) の担当として明記した。 |
| F-3b1-1-011 | MEDIUM | `make help` の `figures-03` の一行説明が `(2 CSV + meta)` で図4枚の言及が抜けており、`figures-01`/`figures-02` が図の枚数を明記しているのと非対称だった。`(2 CSV + 4 figures + meta)` に揃えた。 |
| F-3b1-1-012 | MEDIUM | `docs/design.md` §11.1 末尾が『数値そのものは3bの capacity.csv 生成時に一次資料と機械照合する』という将来形のままで、実際には `capacity.csv` / `capacity_profile.csv` / 図4枚 / `meta.json` が本 diff で既に生成・コミット済みだった。§11.1 末尾に一時的な注記 (結果は 3b-1 で生成済み、数値考察は 3b-2 (T5) で追記予定であることと、その分担が `docs/plans/rc-basics-03b.md` の T3 メモに明記されていること) を追加した。 |
| F-3b1-1-013 | INFO | `fig_mc_sweep.png` 左パネルの上限線 y=N=200 に対し実測データの最大値は 37.84 (18.9%) で、対数軸により線と点群の間に大きな空白が生まれる (対数軸を選んだ理由自体は plan doc に明記済みで設計判断としては妥当)。対応不要 (今回のスコープ外)。3b-2 で design.md に図の解説を追記する際に『上限線はスケールの参照であり、本番設定では N の18.9%までしか到達しない』という一文を加えるよう申し送る (design.md §11 はまだ 3b-2 の担当のため、今回は追記しない)。 |
| F-3b1-1-014 | MEDIUM | 設計文書 (`docs/plans/rc-basics-03b.md`) の実測表が『328.60s / 371.55s』の差43秒を『3-B' の IPC だけの機械側のばらつき』と説明していたが、round1 reviewer が独立に測った3標本目 (368.73s、コミット済み `meta.json` も 370.42s) の区間別内訳は 3-A/3-B を含む**全区間に一様に約+14%**高く、「3-B' だけの特異なばらつき」という説明は当たっていなかった。実測表を観測レンジ (325〜371s、3標本中2標本が360s超) で書き直し、328.60s は最良ケースであって典型値ではないことと、900秒予算への安全マージンは最悪観測値ベースで判断すべきこと、3b-2 で条件を追加する際はこの系統的な+14%変動を見積りに織り込むことを明記した。 |
| F-3b1-1-015 | MEDIUM | 状態生成が60秒予算の58.5〜65.6% (round1 reviewer実測では60.9%) を117条件だけで消費しており、残り3〜4割という余白がplan docに記録されていなかった。実測表の該当行に使用率と、残りの余白が3b-2 (T4) / 04 で条件数・n_steps が増えたときの参考になる旨を追記した。 |
| F-3b1-1-016 | INFO | 実装者報告『03が追加した pytest は2.13秒』に対し round1 reviewer の実測は 5.34〜5.59s (90テスト) だった。対応不要 —— `docs/plans/rc-basics-03b.md` / `docs/design.md` のいずれにも『2.13秒』という記述自体が見当たらず (grep で0件)、訂正対象の一次資料を特定できなかった。3b-2 で design.md §11.5 にテスト実行時間を転記する際は、測定コマンドと対象ファイル集合を併記するよう申し送る。 |
| F-3b1-1-017 | MEDIUM (security, CWE-789/CWE-400) | 3a で入れた診断側の確保上限 (D-34) は IPC の目標列挙・psi_table・chunk の軸しか縛っておらず、03 で新設された実験層の確保軸 (`n_units` / `n_steps`) には閉形式の上限も絶対上限も存在しなかった。設定 YAML の1行変更 (`conservation.n_units_grid: [100000]` 等) で数十GBの確保に到達しうる (実測: N=100000 なら重み行列だけで約80GB)。D-34 と同じ規律で `experiment/capacity.py` に `_MAX_UNITS=5000` / `_MAX_STATE_ELEMENTS=200_000_000` を定数で置き、`evaluate_capacity_condition` が `simulate_reference_trajectory` を呼ぶ前に `_validate_condition_bounds` で検査して `ValueError` にするようにした (`CapacityCondition` 1個の検査で 3-A/3-B/3-B'/length_sweep の4経路すべてが守られる)。`tests/test_experiment_capacity.py` に2本のテストを追加し、`simulate_reference_trajectory` を monkeypatch して確保より前に落ちる (呼ばれない) ことを直接固定した。実測: N=100,000 は 0.003ms で `ValueError` になり、実際の確保は一切発生しない。 |
| F-3b1-1-018 | INFO | `results/03_capacity/meta.json` に求められた4項目 (認証情報等) の混入は0件で、残るのは開発機のフィンガープリント情報 (platform / cjk_font / timestamp) のみ。01/02 の meta.json と同一の既存パターンで、今回の diff が新たに悪化させた点は無い。対応不要。 |
| F-3b1-1-019 | INFO | `main.py` の `--out` 経由のパストラバーサル経路は無い (書き出すファイル名はモジュール定数のみで、設定値がパス要素に混ざる箇所は無い) ことの確認結果。対応不要。 |
| F-3b1-1-020 | MEDIUM | `config.py` が非空615行 (round1 修正後の行数。修正前は非空607行) に達し、仕様 (`docs/plans/rc-basics-03.md`) が明文化した着手条件『非空600行を超えたら04冒頭で独立タスクとして package 化する』に到達した。コード変更は不要 (計画どおり04冒頭で着手)。`docs/plans/rc-basics-03b.md` §3.2 に到達した事実 (実測値・計測コマンド) を追記し、04 着手時に再計測せずに済むようにした。 |
| F-3b1-1-021 | MEDIUM | `evaluate_capacity_condition` が本文92実コード行で、diff 中最長の関数 (02の対応関数 `evaluate_condition` は63行)。対応不要 (ブロッキング水準ではないという reviewer の評価どおり)。次回リファクタリング候補として `_build_capacity_row` 等への分割を検討する (F-3b1-1-004 の分割タスクと合わせて 3b-2/04 で扱うのが自然)。 |
| F-3b1-1-022 | MEDIUM | `tests/test_experiment_capacity.py:199` の `# noqa: B009` に理由コメントが無く、同じサイクルの `# noqa: E501` (行内に理由を明記) と非対称だった。`module` が `ModuleType` 型なので mypy が `module.ipc` を解決できず `getattr` が要る、という理由コメントを追加した。 |
| F-3b1-1-023 | MEDIUM | `tests/test_config_wiring_capacity.py` の配線テストのヘルパー群 (`fingerprint` / `_changed_leaves` 等) が `tests/test_config_wiring_esp.py` とほぼ同型で264行重複していた (01↔02間の109行の重複という既存慣習の延長で、新規パターンではない)。対応不要 (今回統合すると 04 冒頭の `config.py` package 化と競合しうるため)。reviewer の推奨どおり、04 で `config.py` package 化と合わせて `tests/wiring.py` へ共通化する候補として送る (今寄せるなら 01↔02 間の重複も同時に解消しないと一貫性が崩れるため、単独では着手しない)。 |
| F-3b1-1-024 | HIGH | 受け入れ条件1 のうち guard_test `test_mc_effective_delay_increases_with_rho` は『ρ とともに伸びる』側だけを検査し、『N を上限に振る舞う』側は実験レベルで一度も検査されていなかった (図には y=N の参照線が描かれるが、それを裏付ける定量的な下限が無かった。実測: mc_ratio 最小 0.053 / 最大 0.189、ρ 別平均は 0.066→0.088→0.118→0.128→0.139→0.117)。3a の IPC 側の同種の欠落 (F-03-1-023) と同型。guard_test に (iv) ピーク ρ (=1.0) でのリーク率別平均 `mc_ratio` が最低ライン (`MC_RATIO_LOWER_BOUND=0.10`、実測0.139に対し約30%マージン) を上回ることと、`mc_total <= n_units*1.02` の trivial 上限を追加した。破断点を実測: 下限側は mc_ratio を約88.6%まで縮小 (11.4%減) すると検出、上限側は mc_total を約5.39倍に拡大すると検出 (詳細は本文回答を参照)。 |
| F-3b1-1-025 | HIGH | 『本番成果物の古い設定取り残し防止機構』である `test_production_config_matches_the_committed_results` が実験ごとの**行数**しか見ておらず、格子の**値**が変わっても検出しなかった (実測: `rho_grid` の末尾を 1.1→1.2 に差し替えた stale 設定を注入しても行数だけを見ているテストは通過する)。`meta.json['config']` と現在の `load_config_as(...)` を `dataclasses.asdict` (タプル/リストの型差は正規化) で突き合わせる `test_production_config_matches_the_committed_meta_json` を追加した。修正後、同じ変異 (rho_grid の値を差し替え) を注入すると新テストが実際に落ちることを実測で確認した (行数一致テストは相変わらず通過するため、両テストが揃って初めて値ドリフトが閉じることも確認済み)。 |

## サイクル 3b-1 round 2 (`F-3b1-2-xxx`)

round 2 は `.claude/tmp/findings/3b1-round-2/reviewer-{test,architecture,docs}.json` の3件 (test/architecture/docs)。
architecture と docs は round 2 到達時点で PASS (締めてよい) 判定、test は「HIGH 1件を直せば締めてよい」判定。
BLOCKER 0 / HIGH 1 / MEDIUM 3 / INFO 5 の計10件で、3b-1 を締める最終ラウンド。

| ID | severity | 概要 |
|---|---|---|
| F-3b1-2-001 | HIGH | `tests/test_capacity_pipeline.py` の `MC_RATIO_LOWER_BOUND` docstring が『本番実測 (rho=1.0 のリーク率別平均) は 0.139 で約30%のマージン』と書いていたが、この 0.139 は rho=1.0 の全 leak_rate・全 replicate を束ねた単純平均 (grand mean) で、assert が実際に評価する統計量 (`min(peak_ratio_means)`、leak_rate ごとに平均してから最小値を取る) ではなかった。実測すると `min(peak_ratio_means)` = 0.1129 (leak=0.3) でマージンは約13%、`IPC側(27%)やMC単体guard(33〜39%)と同水準』という結論は誤りだった。docstring の根拠数値を実際に assert が使う統計量 (0.1129) に置き換え、マージンを『約13%』と正しく記載し、複数シードでの再実測結果 (min(leak別平均) 0.1126〜0.1189、個別 replicate 最小 0.1057) を1行加えた。assert のロジック自体は変更していない (`min(peak_ratio_means) >= 0.10` のまま)。 |
| F-3b1-2-002 | MEDIUM | round2 で追加した `test_production_config_matches_the_committed_meta_json` (F-3b1-1-025) が `Capacity03Config` 全体を `meta.json['config']` と突き合わせるため、`figures-03` の成果物が一切依存しない `narma` セクション (3-C、3b-2 の T4 担当で現在は不活性) の編集でも赤くなった (実測: narma の `n_units` を 200→50 に変えるだけで不一致になり、T4 が最初に踏む F-3b1-1-008 の指示そのものと衝突する)。`tests/test_config_wiring_capacity.py` の `PENDING_SECTIONS` (narma を『まだ配線されていない節』として明示的に扱う定数) を import し、比較前に `current`/`committed` 双方から `PENDING_SECTIONS` に含まれるキーを落とすよう修正した。narma だけを編集しても不一致にならないこと、`rho_grid` の値を変えると相変わらず落ちること (本来の目的を損なっていないこと) の両方を実測で確認した。 |
| F-3b1-2-003 | MEDIUM | `config.yaml` の D-39 警告コメント (旧153-156行) が `esn_delay_parity` の直前にしか無く、同じ narma セクションの `esn_mackey_glass` (旧145行、同じく `n_units: 200`) には何も付いていなかった。コメント本文自体が『T4 がどちらのセクションを読むか配線するまでは不活性』と両セクションが候補であることを認めているのに、指示は直後の1つしか指していなかった。警告コメントを両 ESN セクション (`esn_mackey_glass` / `esn_delay_parity`) の前にまとめて1つ置く形に書き直し、『以下の ESN セクションはどちらも D-39 (N=50) に未追従』と明記した。値の変更は従来どおり T4 に残した。 |
| F-3b1-2-004 | MEDIUM | D-38 (訂正後) の rule『成果物から保存則を検算する不変条件は、capacity 列の総和が ipc_total/mc_total と一致すること』の検算が `tiny_config` を回す guard_test 内の2箇所にしか無く、本番成果物 (`results/03_capacity/`) を見るテストは正値性と条件キーの包含しか見ていなかった (round1 で指摘された『行数 == n_targets_kept』という本番で成立しない不変条件の再発と同型 — 記録が本番成果物について述べているのに検査が無い構図)。`test_production_profile_rows_are_positive_and_reference_the_same_conditions` に、条件キー (experiment/replicate/rho/leak_rate/n_units/state_noise) ごとに diagnostic 別の capacity 列の総和を取り `capacity.csv` の `mc_total`/`ipc_total` と `math.isclose` で突き合わせる検査を追加した (両 CSV は既読み込みで追加コストはほぼ0)。実測: 117行すべてで誤差0件のため追加時点で緑。`capacity_profile.csv` の1セルを +1000 する変異を注入すると実際に落ちることを確認 (空虚でないことの実測)。 |
| F-3b1-2-005 | INFO (reviewer-test) | 実験層の確保上限ガード (`_MAX_UNITS`/`_MAX_STATE_ELEMENTS`) に境界値ちょうどのテストが無く、`>` が `>=` に書き換えられても検出できない。対応不要 (今回のスコープ外、次PR候補)。 |
| F-3b1-2-006 | INFO (reviewer-architecture) | D-36 は `src/` 内の `ESN.run` 呼び出し3箇所すべてで文字通り満たされたが、`esp.py` の `esn_propagator` が `esn.step` を rng なしで呼んでおり、04 で 02 経路に `state_noise` を入れると D-36 が防いだのと同じ `ValueError` が復活しうる (`conditional_lyapunov` の伝播器は摂動成長率を測るため決定的である必要があり、単に rng を渡せばよい話ではない)。コード変更・コメント追加とも見送った (esp.py への変更はスコープ外)。04 冒頭の別タスク候補として本報告に記録し、`docs/plans/rc-basics-03b.md` 側の申し送りは 3b-2/04 の担当に委ねる。 |
| F-3b1-2-007 | INFO (reviewer-architecture) | 比較軌道が `state_noise=0` の現状では乱数を1個も引かないため 02 の軌道はバイト不変だが、`state_noise>0` になると比較軌道は『初期状態もノイズ実現値も違う』軌道になり D-14 の3ストリーム分離に4本目の未制御な変動が混ざりうる。コード変更・コメント追加とも見送った (esp.py への変更はスコープ外)。04 でノイズを入れる担当への申し送り事項として本報告に記録した。 |
| F-3b1-2-008 | INFO (reviewer-docs) | `.claude/decisions.yaml` の D-38 rationale と plan doc の『合計 1.78 MB』が実測合計 1.74 MB (1,824,677 bytes) と約2.3%ずれる (今回の diff で変更された行ではなく既存の劣化)。対応不要 (今回のスコープ外、次PR候補)。 |
| F-3b1-2-009 | INFO (reviewer-docs) | `docs/plans/rc-basics-03b.md` の性能実測4標本 (324.57/326.71/368.73/370.42) が ~325s 側2件・~370s 側2件の2山に分かれる可能性があるが、現在の文書は『全区間に一様な系統的変動』という連続分布的な説明のみ。実務上の結論 (最悪観測値ベースで予算判断) は変わらないため対応不要 (今回のスコープ外、3b-2 でサンプルを追加する際に候補として書き添えることを推奨)。 |
| F-3b1-2-010 | INFO (reviewer-docs) | README『## 実験03』表の CSV 行数 (117行/21,636行) は成果物と一致するが、01/02/03 いずれも CSV 行数は `tests/test_readme_summary.py` の機械照合対象になっていない。03 固有の新規ギャップではなく既存の慣習の範囲内のため対応不要 (reviewer 自身も対応不要と判定)。 |

## サイクル 3b-2 round 1 (`F-3b2-1-xxx`)

サイクル3を締める最終ラウンド。**BLOCKER 0 / HIGH 2 / MEDIUM 4 / INFO 5**。
reviewer-architecture は「サイクル3は締めてよい」と判定済み。HIGH 2件・MEDIUM 4件を修正した。
**運用上の注記**: このラウンドは `.claude/tmp/findings/3b2-round-1/` の `fixer-input.json` /
`triage.json` が生成されないまま fixer に渡された (ディレクトリが空)。HIGH 2件・MEDIUM 4件は
オーケストレータの指示文からそのまま作業した (通常の運用はファイル経由で findings を渡し、
プロンプトへの転記を禁じているが、今回は転記元のファイルが存在しなかったための例外対応)。
INFO 5件のうち D-35 の変異試験再現 (下記 F-3b2-1-005) 以外の4件は指示文に内容が無く、
出典ファイルも存在しなかったため対応していない (捏造を避けるため記録もしていない)。

| ID | severity | 概要 |
|---|---|---|
| F-3b2-1-001 | HIGH (security) + MEDIUM x2 (architecture) | 3-C (`run_narma10`) としきい値法比較 (`run_threshold_comparison`) が `experiment/capacity.py` の `_validate_condition_bounds` (確保より前の上限検査、D-34/F-3b1-1-017) を1回も呼ばずに素通りしていた。`CapacityCondition` を組み立てる経路が当初の4本 (3-A/3-B/3-B'/length_sweep) から、しきい値法比較を含む5本目に増えていたのに検査は個別追加式のままだった (実測: `Narma10Config(length=10**12)` が受理され、`length=1e8 x n_units=50` で状態行列だけ5e9要素=40GBに到達しうる状態だった)。`simulate_condition_trajectory` (`_validate_condition_bounds` を内包して軌道を作る) と `capacity_context` (D-37 の `DiagnosticContext` を1個作る、3モジュールに分散していた複製の解消) を `capacity.py` に切り出し、`evaluate_capacity_condition` / `run_threshold_comparison` / `run_narma10` の3か所を差し替えた。あわせて `tasks/narma.py` の `_validate` に `length` 単体の絶対上限 (`_MAX_LENGTH=2e8`) と `length * base.esn_mackey_glass.n_units` の絶対上限 (`_MAX_STATE_ELEMENTS=2e8`) を追加し、`CapacityCondition` を持たない3-Cを課題層単体で塞いだ。テストを4本追加 (`test_oversized_ipc_sweep_n_units_is_rejected_before_any_allocation` / `test_validate_condition_bounds_is_actually_called` / `test_oversized_narma10_length_is_rejected_before_any_allocation` / `test_narma10_length_boundary_plus_one_over_n_units_product_is_rejected`)。実測: 修正前のコードに相当する変異 (`run_threshold_comparison` が `_validate_condition_bounds` を経由しない9引数呼び出しに戻す) を注入すると新テスト2本が実際に落ちることを確認済み。`capacity.py:157` の `_validate_condition_bounds` docstring の「4経路すべてが守られる」という古い記述も実態 (5経路 + 3-Cは別経路) に訂正した。 |
| F-3b2-1-002 | HIGH (test) | D-29 の guard_test (`test_matches_reference_recurrence`) が先頭5ステップ (`y[10]..y[14]`) しか照合しておらず、窓を `sum_{i=0}^{9}` (10項) から `sum_{i=0}^{10}` (11項) に広げる変異を検出できなかった (`y[0..9]` が0初期化のため、11項目 (`y[t-10]`) が非0の値を拾い始めるのは `t - 10 >= 10` すなわち `y[21]` を計算する時点からで、それより手前は偶然一致する)。`REFERENCE_INPUT` を15ステップから22ステップに延ばし、独立実装 (`_hand_computed_window_reference`、Fraction・別ループ、`narma10_series` のコードは参照しない) で `y[21]` まで計算する `test_matches_reference_recurrence_through_extended_window` を追加した。あわせて `test_shifted_index_would_not_match` と対になる `test_wider_window_would_not_match` を追加し、「11項の変異は `y[21]` で初めて食い違う (それより手前は10項の正しい実装と偶然一致する)」ことをセルフテストとして実測で固定した。実測: `narma10_series` の窓を `y[max(t-NARMA10_ORDER,0):t+1]` (11項) に書き換える変異を注入すると `test_matches_reference_recurrence_through_extended_window` が実際に落ち (`0.5448609208382754 != 0.5398125327224791`)、既存の `test_matches_reference_recurrence` (5ステップのみ) は変わらず緑のままであることを確認した (=延長前の guard_test はこの変異を検出できなかったことの直接証拠)。 |
| F-3b2-1-003 | MEDIUM (docs) | `docs/design.md` §11.5 の3表 (3-A の `mc_effective_delay` リーク率×ρ表、3-B の `ipc_total`/`ipc_linear`/`ipc_nonlinear` リーク率×ρ表 12行、3-B' の `ipc_total`（`saturation_ratio`）N×`state_noise`表 9セル) が、§11.2/§11.5 の他の表 (しきい値比較・3-C成績・3-C容量・実行時間・成果物サイズ) と違い `capacity.csv` と機械照合されていなかった (groupby 照合0件)。値は現在正確だが、再生成しても表が古いままだと赤くならない状態だった。既存の `_table_after` + `_assert_cell_matches` パターンで3本のテストを追加 (`test_mc_sweep_delay_table_matches_the_capacity_csv` / `test_ipc_sweep_table_matches_the_capacity_csv` / `test_conservation_table_matches_the_capacity_csv`、`capacity.csv` を rho/leak_rate または n_units/state_noise で groupby してレプリケート平均を再計算し照合)。in-process monkeypatch (`test_design_doc._text` を差し替え、ファイルは一切編集していない) で表の1セルを書き換える変異を注入し、新テストが実際に落ちることを実測で確認した。 |
| F-3b2-1-004 | MEDIUM (architecture) | `capacity_pipeline.SectionTiming.wall_time_s` の docstring は「条件の合計 (状態生成+MC+IPC + 行の組み立て)」と宣言していたが、3-C だけ `_narma_timing` が `narma.wall_time_s` (`run_task`、3手法×全レプリケートを含む3-C全体) に差し替えており、`CapacityRow.wall_time_s` (常に容量測定のみ) と同じ列名で意味が3-Cだけ食い違っていた (実測: 3-C の残差 0.149秒 = 行の45%、他の実験は0.002〜0.004秒の丸め)。フィールド追加 (`wall_time_scope` 等) ではなく **docstring の訂正**を選んだ (`SectionTiming.wall_time_s` のクラス docstring と `capacity_row_from` の Args に3-Cの例外を明記)。フィールドを足すと `meta.json` のキーが増え、成果物の再生成と §11.5 実行時間表の機械照合を同時に更新する必要があり、コード変更なしで塞げる問題にコストが見合わないと判断した。`results/03_capacity/` はロジック・出力キーとも変更していないためバイト不変。 |
| F-3b2-1-005 | INFO (docs) | D-35 の rationale が主張する変異試験結果 (`capacity.py` の `states.flags.writeable = False` を消すと `tests/test_experiment_capacity.py` は「2 failed / 13 passed」になる) を reviewer が読み取り専用権限で確認できなかった件。fixer が in-process monkeypatch (`rc_basics_lab.experiment.capacity.measure_capacity` を read-only 化の1行を欠いた版に差し替え、ファイルは一切編集していない) で再現し、`test_states_are_read_only_before_capacity_problem` と `test_externally_built_states_can_produce_a_capacity_row` の2件が実際に落ち、それ以外 (`test_mc_and_ipc_share_the_same_state_matrix` 等) は通過することを確認した。rationale の記載どおりであることを確認済み (コード変更なし)。 |
