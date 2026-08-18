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
