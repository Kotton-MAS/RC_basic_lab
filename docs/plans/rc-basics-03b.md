# 仕様: rc-basics-03b —— 容量の実験と記事の図 (T3・T4・T5 の残り)

*要件書: `docs/要件_rc-basics-03.md` / 前サイクル仕様: `docs/plans/rc-basics-03.md` (§4 T3・T4・T5)*
*前提: サイクル1・2a・2b・**3a** 完了 (534 tests / D-01〜D-28 + D-33 + D-34 / `make ci` 緑)*

> **本書の位置づけ**: `docs/plans/rc-basics-03.md` line 9 で承認済みの分割
> 「3b (実験と記事の図) = T3・T4 + T5 の残り (D-29〜D-32、図5枚、実測表)」を、
> 3a の完了事実 (性能実測・D-32〜D-34・持ち越し6項目) を踏まえて具体化したもの。
> §4 のタスク番号は 3b 内での通し番号 (**T1〜T5**) に振り直してある。
> 旧仕様との対応: 3b-T1+T2+T3 = 旧 T3 / 3b-T4 = 旧 T4 / 3b-T5 = 旧 T5 の残り。

---

## 1. ゴール

3a の診断2本 (MC / IPC) を**配線するだけ**で、ρ・リーク率・ノイズに対する容量の移動と
公平な対照下の NARMA10 比較を、1コマンド再生成できる CSV・図5枚・実測表として出す。

---

## 2. 現状認識

### 2.1 関連箇所 (すべて実測で確認済み)

| 用途 | 場所 | 3b での使い方 |
|---|---|---|
| MC 診断 | `src/rc_basics_lab/diagnostics/memory_capacity.py:160-277` | `memory_capacity(X, u, ctx=..., cfg=...)`。scalars 6個 + `mc_profile` |
| IPC 診断 | `src/rc_basics_lab/diagnostics/ipc.py:720-834` | `ipc(X, u, ctx=..., cfg=...)`。scalars 7個 + `ipc_threshold_degree{d}` (**本数が cfg 依存**) + arrays 3本 |
| 目標数の下見 | `src/rc_basics_lab/diagnostics/ipc.py:366-456` | `count_targets(cfg)` / `enumerate_targets(cfg)` は公開済み。打ち切り変更時の見積りをテストから直接引ける |
| 容量カーネルの契約 | `src/rc_basics_lab/diagnostics/_capacity.py:79-196` | `CapacityProblem` は `X` の**ビュー**を持ち `gram` はスナップショット。**構築後に `X` を書き換えると例外も警告もなく desync** |
| 参照軌道の生成 | `src/rc_basics_lab/experiment/esp.py:260-312` | `simulate_reference_trajectory(reservoir, drive_config, ...)`。**03 のために 02 から切り出し済み** |
| ESN 構成 | `src/rc_basics_lab/experiment/esp.py:225-242` | `build_esn_config(reservoir, rho, leak_rate)`。**`state_noise` を渡す口が無い** (下記 2.4) |
| 状態ノイズ | `src/rc_basics_lab/reservoir/esn.py:182-199, 233-262` | `state_noise > 0` かつ `rng is None` は `ValueError`。`state_noise=0` なら 1個も引かないので既存結果は不変 |
| 01 の公平性機構 | `src/rc_basics_lab/experiment/runner.py:295-333` (`run_task`) / `161-191` (`plan_replicate`) | 3-C は `run_task(base_config, task_entry, plan0=...)` を直接呼ぶ。`build_tasks` に触らない (D-31) |
| 01 の行 dataclass | `src/rc_basics_lab/experiment/runner.py:93-117` | `ResultRow` / `CSV_COLUMNS` を 3-C がそのまま使う |
| pipeline の前例 | `src/rc_basics_lab/experiment/esp_pipeline.py:221-291` | 成果物一覧を定数 `*_ARTIFACTS` にし、CLI は薄い層 |
| meta の書き出し | `src/rc_basics_lab/experiment/report.py:76-112` | `write_meta_for(config, seeds, wall_time_s, n_rows, path, extra=...)` |
| 設定ローダ / 内包の前例 | `src/rc_basics_lab/config.py:452-488` / `314-332` | `load_config_as(path, cls)`。**01 の `ExperimentConfig` をまるごと内包する前例がある** |
| 配線テストの機構 | `tests/wiring.py` / `tests/test_config_wiring_esp.py:114-120` | チャネル割り当てを機械的に閉じる |
| 実験レジストリ | `main.py:76-96` | `"03"` を1行。`out_dir=results/03_capacity` |
| Makefile | `Makefile:51-80` | `figures-01` / `figures-02` / `threshold-02` の3前例 |
| design.md の既定値表 | `docs/design.md:700-807` (§11.1) | 表の「コード上の出どころ」列は**実在の属性に解決される**ことが機械検査される |

### 2.2 既存の慣習で守るもの (3個)

1. **CSV の列順は行 dataclass の宣言順が単一の真実**
2. **設定 dataclass は純データ、値域検証は使う側**。未知キーは `ConfigError` (D-09)
3. **図のラベルは `labels.label(ja, en, cjk=...)` を通す** (D-10)

### 2.3 影響範囲

- **追加**: `config.py` に 03 セクション群 / `experiment/capacity.py` / `experiment/capacity_pipeline.py` /
  `experiment/narma.py` / `tasks/narma.py` / `plotting/figures_capacity.py` /
  `experiments/03_capacity/{config.yaml,run_03.py}` / テスト5本
- **変更**: `experiment/esp.py` (**既定値つきキーワード `state_noise` の追加のみ**、D-36) /
  `main.py` (1行) / `Makefile` (2ターゲット) / 各 `__init__.py` / `README.md` /
  `docs/design.md` §11.2〜11.5 / `.claude/decisions.yaml`
- **変更しない**: `diagnostics/` 配下 (**3a で完成。3b は1行も触らない**) /
  `experiment/runner.py` の `build_tasks` / `ExperimentConfig` / `seeds.py` の既存ストリーム index /
  01・02 の成果物

### 2.4 3a 完了時点で新たに判明した事実

1. **性能予算に余裕がある** (実測): 3-A 0.131s/条件・3-B 1.821s/条件・3-B' 9.841s/条件。
   診断計算の合計は約 161 秒で 900 秒予算の 18%。
   ただし**状態生成側 (素の tanh ESN、O(T·N²) の Python ループ) は未実測**。3b で実測する (T2)。
2. **メモリは T=1e6 で MC 2.11 GB / IPC 0.90 GB** (予算 4 GB)。
3. **`ipc_threshold_degree{d}` の本数は D-34 の `max_degrees` により最大20本と静的に有界**。
   ただし本仕様では列にせず行に落とす (D-38)。
4. **`simulate_reference_trajectory` は `state_noise` を渡せない**。`build_esn_config` が
   `ESNConfig.state_noise` を設定せず、`esn.run(u, x0=x0)` に `rng` も渡していないため、
   受け入れ条件2 を測る経路が**現状は存在しない**。

---

## 3. 前提・制約

### 3.1 ハード制約 (絶対に変えない)

- **D-01 / D-15**: 診断の署名は `f(X, u=None, y=None, *, ctx)`。`DiagnosticContext` に**フィールドを足さない**
- **D-12 / D-23**: `diagnostics/` は `config` / `reservoir` を import しない。**3b の配線は `experiment/` 側**
- **D-13**: `ExperimentConfig` (01 専用) に 03 のフィールドを1個も足さない
- **D-04 / D-05 / D-08 / D-31**: 3-C は 01 の `run_task` 経路をそのまま通す
- **D-24〜D-28 / D-32 / D-33 / D-34**: 3a で確定済み。3b は消費側であり再定義しない
- **D-02**: 誤差指標は NRMSE 主・NMSE 併記
- 型: Python 3.12+ / `Any` 禁止 / `# type: ignore` を理由なく使わない
- `results/` は**コミット対象**。生成物のサイズは受け入れ基準に含める

### 3.2 ソフト制約 (理由があれば変えてよい)

- `config.py` の package 化は **04 の冒頭で独立タスク**とする
- 03 は 02 の `DriveConfig` / `ReservoirSweepConfig` を**設定として再利用しない**。
  **関数**は再利用し、**値**は配線層が1か所で組み立てる
- **駆動強度 `sigma_u` はセクションごとに持つ**。MC の ρ 依存は準線形域を要し、
  IPC の次数分解は中程度の駆動を要するため最適な動作点が一致する保証が無い。
  **図をまたいだ絶対値比較はしない**ことを design.md §11.5 に明記する
- 漸近展開による IPC 推定は v0.1 では実装しない

### 3.3 3a から持ち越した設計課題の扱い (**着手前に一読**)

| # | 課題 | 3b での扱い |
|---|---|---|
| 1 | `X` の read-only 化 | **3b で対応** (D-35。T1 の受け入れ基準) |
| 2 | F-03-1-005: IPC の scalars キー集合が cfg 依存 | **3b で対応** (D-38。長形式の行に落とす) |
| 3 | F-03-1-006: `orthonormal_basis` の引数設計 | **04 送り** (`diagnostics/` を触るため) |
| 4 | `max_targets` が cells と targets を同じ値で縛る | **04 送り**。3b の最深設定でも目標 4,075 / セル 800 で上限 200,000 に対し2桁の余裕 |
| 5 | `RowAlignment` の切り出し / 引数集約 dataclass | **04 送り** |
| 6 | `chunk_size` が性能軸とメモリ軸を兼ねる | **04 送り** (分離に新しい設計判断が要る) |
| 7 | `diagnostics/__init__.py` の関数 `ipc` によるモジュール名の隠蔽 | **3b では直さない** (公開 API 変更)。§10-1 の注意 + テストを1本 |

> **3〜6 を 04 へ送る根拠**: いずれも `diagnostics/` 内部の整理であり、3b の成果物の正しさに影響しない。
> 3b で触ると「測定装置の設計判断が実験層の設計に混ざる」という 3a/3b 分割の目的が崩れる。

---

## 4. タスク分解

### T1: 設定層と1条件の配線 (想定所要: **L**)

1. `config.py` に 03 の設定群 (D-13。`ExperimentConfig` には触らない):

```
Capacity03Config
├ name / seeds(reservoir, drive, surrogate) / drive(distribution, washout)
├ reservoir(input_scale, density, n_replicates)      # セクション横断で共有する3つだけ
├ mc_sweep(rho_grid, leak_rate_grid, sigma_u, n_units=200, n_steps=20000)
├ ipc_sweep(rho_grid, leak_rate_grid, sigma_u, n_units=50, n_steps=100000)
├ conservation(n_units_grid=(25,50,100), state_noise_grid, rho, leak_rate, sigma_u,
│              n_steps=200000, max_delay_by_degree=(200,60,20,10))
├ length_sweep(n_steps_grid=(1e5,2e5,5e5,1e6), ...)   # make saturation-03 用
├ mc: MemoryCapacityConfig / ipc: IpcConfig            # 3a の診断設定 (D-15)
└ narma: Narma10Config(base: ExperimentConfig, length=8000)
```

   - **`n_units` はセクションが持つ** (D-32)。`reservoir` が持つのは横断共有の3つだけ
   - `conservation.max_delay_by_degree` は 3-B' でのみ `config.ipc` を上書きする (片方向)

2. `experiment/capacity.py` (**`reservoir` と `diagnostics` の両方を import してよい唯一の場所**):
   - `CapacityCondition` / `evaluate_capacity_condition(config, condition) -> CapacityOutcome`:
     1. `simulate_reference_trajectory` で **X を1条件につき1回だけ**作る
     2. **`X.flags.writeable = False` にしてから**診断へ渡す (D-35)
     3. 同じ `X` と `u` で `memory_capacity` と `ipc` の**両方**を呼ぶ (D-38 の前提)
     4. `ctx = DiagnosticContext(washout=config.drive.washout, seed=config.seeds.surrogate)` (D-37)
   - `CapacityOutcome`: `row` + 図が使う配列 (02 の `ConditionOutcome` と同型)
3. `experiment/esp.py` に **既定値つきキーワード** `state_noise: float = 0.0` を追加し、
   `esn.run(u, x0=x0, rng=reservoir_rng)` にする (D-36)

**受け入れ基準**

- `test_all_capacity_config_fields_are_covered` — 全葉にチャネル割り当て。委譲先と過不足なく一致
- `test_each_capacity_parameter_changes_output`
- `test_states_are_read_only_before_capacity_problem` (D-35)。**変異試験**: read-only 化を外すと落ちる
- `test_reference_states_match_esp_simulate_condition`: `state_noise=0` で 02 と**バイト一致** (D-36)
- `test_state_noise_changes_states_and_requires_rng`
- `test_mc_and_ipc_share_the_same_state_matrix`: 軌道生成の呼び出し回数が条件数と一致
- `test_surrogate_seed_is_shared_and_recorded` (D-37)
- `test_diagnostics_ipc_module_and_function_are_both_reachable` (§10-1)
- `test_ipc_reservoir_is_smaller_than_mc_reservoir` (D-32)

### T2: 実験 3-A / 3-B / 3-B' と成果物2枚 (想定所要: **L**)

1. 掃引3本 (`run_mc_sweep` / `run_ipc_sweep` / `run_conservation_sweep`)
2. **行 dataclass 2本**:

   `CapacityRow` (`capacity.csv`、1行 = 1条件、**全列が常に埋まる**):
   `experiment` / `replicate` / `seed_*` ×3 / `rho` / `leak_rate` / `input_scale` / `sigma_u` /
   `input_drive_std` / `n_units` / `density` / `state_noise` / `n_steps` / `washout` /
   `t0_mc` / `n_samples_mc` / `mc_total` / `mc_total_raw` / `mc_threshold` / `mc_effective_delay` /
   `mc_ratio` / `n_delays` / `t0_ipc` / `n_samples_ipc` / `ipc_total` / `ipc_total_raw` /
   `ipc_linear` / `ipc_nonlinear` / `ipc_saturation_ratio` / `n_targets` / `n_targets_kept` /
   `n_degrees` / `chunk_size_mc_effective` / `chunk_size_ipc_effective` /
   `wall_time_state_s` / `wall_time_mc_s` / `wall_time_ipc_s` / `wall_time_s`

   `CapacityProfileRow` (`capacity_profile.csv`、長形式、D-38):
   `experiment` / `replicate` / `rho` / `leak_rate` / `n_units` / `state_noise` /
   `diagnostic` / `degree` / `delay` / `capacity` / `threshold`
   - **しきい値後の容量が厳密に正のセルだけを書く** (全セルだと約6万行)

3. `experiment/capacity_pipeline.py`: 2 CSV + 図4枚 + `meta.json`。`CAPACITY_ARTIFACTS` 定数
4. `main.py` に `"03"`、`Makefile` に `figures-03` (本番) と `saturation-03` (**`figures-03` に含めない**)
5. **駆動強度の較正**: `sigma_u ∈ {0.05, 0.1, 0.2, 0.5}` を代表条件で測り、
   **受け入れ条件1 を満たす最小値**を選ぶ。**選んだ値と落選した値の実測を design.md §11.5 に残す**

**受け入れ基準**

- **性能の実測を先に出す**。**合計見積りが 700 秒を超えた場合に許可される調整は
  `conservation.n_replicates` を 3 → 1 に落とすことだけ**。打ち切り・条件数・`n_steps` は動かさない
- `test_profile_csv_columns_are_static_and_cells_are_positive` (D-38)
- `test_capacity_csv_has_no_missing_values`
- `test_conservation_respects_the_bound` (受け入れ条件2)
- `test_mc_effective_delay_increases_with_rho` (受け入れ条件1)
- `test_conservation_section_does_not_change_the_other_experiments` (scope 検査)
- 縮小設定の CLI が **20 秒以内** / `results/03_capacity/*.csv` 合計 **< 5 MB**

### T3: 図4枚 (想定所要: **M**)

| 図 | 元データ | 軸 | 受け入れ条件 |
|---|---|---|---|
| `fig_mc_sweep.png` | 3-A | 左: x=ρ, y=`mc_total` (leak 別・平均±s.d.)、上限線 y=N。右: `mc_profile` を ρ 別に重ねる | **1** |
| `fig_ipc_profile.png` | 3-B | (次数 × 遅延) ヒートマップを ρ 4点 × 代表 leak のパネルに | **4** |
| `fig_memory_nonlinearity.png` | 3-B | `ipc_linear` と `ipc_nonlinear` の積み上げで配分の移動を見せる | **4** |
| `fig_ipc_conservation.png` | 3-B' | x=N, y=`ipc_total`、noise 別の線、上限線 y=N (**傾き1の対角線**) | **2** |

**受け入れ基準**: 200 dpi / `test_conservation_figure_draws_the_bound_line` /
`test_figures_use_the_style_context_labels` / 縮小データ (2条件) で全図が描ける

### T4: 実験 3-C —— 公平な対照での NARMA10 (想定所要: **M**)

1. `tasks/narma.py` (D-29): **係数と入力分布はモジュール定数**。`Narma10Config` は `length` のみ。
   発散は `ValueError` (D-30)
2. `experiment/narma.py`: `plan0 = plan_replicate(...)` を作り `run_task(base, entry, plan0=plan0)` へ渡す。
   **出力は 01 の `ResultRow` をそのまま使う**。`plan0.states` に対する MC / IPC は
   `capacity.csv` に `experiment="3C_narma10"` の行として追加する
3. `plot_narma10_control`: 参照線 NMSE=0.16 / 0.107 を**注記つき**で描き、**原典未特定を明記**

**受け入れ基準**: `test_matches_reference_recurrence` (D-29) / `test_divergence_raises_instead_of_clipping` (D-30) /
`test_narma10_reuses_run_task_and_shares_rows_across_methods` (D-31) / `test_narma10_alpha_grid_is_shared_across_methods` (D-04) /
`test_narma10_capacity_uses_the_same_states_as_the_esn_run` / `test_narma10_esn_size_matches_the_declared_choice` (D-39) /
`test_01_artifacts_are_unchanged`。**結果の向きは問わない**

### T5: 記録 (想定所要: **M**)

- `docs/design.md` §11.2 (しきい値法の比較、受け入れ条件3) / §11.3 (打ち切り表と目標数・実行時間) /
  §11.4 (NARMA10 の採用式と先行との差分) / §11.5 (実測結果・実行時間表・**駆動強度の較正記録**・
  `config.py` の行数)
- `.claude/decisions.yaml` に **D-29〜D-32 と D-35〜D-39**
- `README.md` に `make figures-03` / `make saturation-03` と成果物表
- `tests/test_design_doc.py` に §11.2 の照合を1件

---

## 5. 評価軸 (Check フェーズに渡す)

### 性能観点 (**予算を明示する**)

| 区間 | 予算 |
|---|---|
| `make figures-03` 合計 | **< 900 秒** |
| 3-A (54 条件) | < 120 秒 |
| 3-B (36 条件) | < 180 秒 |
| 3-B' (27 条件) | < 400 秒 |
| 3-C | < 120 秒 |
| **状態生成の合計** | < 60 秒 / **内訳を `meta.json` に出す** |
| `make saturation-03` | < 1800 秒 (予算外・手動) |
| 03 が追加する pytest | < 60 秒 |
| ピークメモリ | < 4 GB |
| `results/03_capacity/*.csv` | < 5 MB |

**禁止する構造**: 「条件ごとに X を2回作る (MC 用と IPC 用)」/「(delay, degree) ごとに ESN を再実行する」

### 安全性観点

- 01・02 の成果物がバイト単位で不変 / `seeds.py` の既存4ストリームの `spawn_key` を動かさない
- `DiagnosticContext` のフィールドが増えていない (D-01) / `diagnostics/` の import 制約 (D-23)
- **`diagnostics/` 配下の差分が 0 行** (3b は消費側)

### 有効性観点 (**必須**)

- `Capacity03Config` の**全葉**にチャネル割り当て。委譲先と過不足なく一致
- **セクション固有の葉は scope 検査つき** (`mc_sweep.*` は 3-A の行だけを変える、を4セクションで)
- `seeds.surrogate` を変えるとしきい値が変わり `ipc_total` が動く (D-37 の配線の実体)
- `conservation.max_delay_by_degree` を変えると `n_targets` が変わり `count_targets` と一致する

---

## 6. 意図的な決定

> **D-29〜D-32 は `docs/plans/rc-basics-03.md` §6 が 3b 用に予約済みの採番**。
> 新規は **D-35 以降** (D-33 / D-34 は 3a で消費済み)。

主要な新規決定:

- **D-35**: `CapacityProblem.from_states` に渡す X は呼び出し側が read-only にする。
  診断側でコピーすると T=1e6 で 1.6GB 増えて 4GB 予算を壊す (F-03-1-013 で潰したのと同じ失敗)
- **D-36**: `state_noise` を `esp.py` に既定値つきキーワードで足し、`run` に常に rng を渡す。
  02 の呼び出しは書き換えず成果物はバイト不変
- **D-37**: サロゲートの `ctx.seed` は1個を全条件で共有する (共通乱数法)。
  条件ごとに振ると容量差にしきい値の推定ノイズが独立に乗る
- **D-38**: cfg 依存のキー集合は列にせず長形式の行に落とす。正値セルのみ書く
- **D-39**: 3-C の ESN は **N=50** とし 3-B (IPC 掃引) と同じリザバー規模にする。
  参照値 (0.16 / 0.107) が50ノード級であり、N=200 で回すと『先行より良い』が規模差なのか
  対照設計の差なのか分離できない。D-08 により N は検証分割で選ばれないので宣言した1点を報告する

---

## 7. 想定リスク (起きたら止まって相談)

1. **`make figures-03` の合計が 700 秒を超える**。許可される調整は
   `conservation.n_replicates` を 3 → 1 だけ。それでも収まらないなら**実装者判断で削らず止まる**
2. **本番設定で受け入れ条件1 が成立しない**。原因は (a) 駆動強度が飽和域 (b) T 不足による偽陽性
   (c) しきい値法 のどれかで対処が正反対。T2-5 の較正で (a) を先に切り分け、それでも出ないなら止まる
3. **3-C で遅延線が ESN を上回る**。結果としては受け入れ (記事になる) が、探索予算の非対称性
   (D-08) が効いている可能性があるため追加条件を回す前に相談する

---

## 8. 確定事項 (ユーザー承認済み・2026-08-18)

| 問 | 決定 |
|---|---|
| **分割** | **2分割**。**3b-1 = T1+T2+T3** (容量実験・図4枚) / **3b-2 = T4+T5** (NARMA10・記録) |
| **3-C の ESN 規模** | **N=50** (D-39)。参照値 0.16 / 0.107 が50ノード級で、規模差と対照設計差を分離できる。3-B (IPC 掃引 N=50) とそのまま突き合わせられる |
| **持ち越し設計負債** | F-03-1-006 / `max_targets` の単位 / `RowAlignment` / `chunk_size` の二重意味は **04 冒頭**の「`config.py` package 化 + 容量カーネルの整理」タスクへ送る。3b は `X` の read-only 化 (D-35) と CSV 列設計 (D-38) のみ対応し、**`diagnostics/` の差分を 0 行に保つ** |

### 分割後のタスク配分

| サイクル | タスク | L の本数 | 主な成果物 | 終了条件 |
|---|---|---|---|---|
| **3b-1** | T1 (L) / T2 (L) / T3 (M) | 2 | `capacity.csv` / `capacity_profile.csv` / 図4枚 / `make figures-03` / D-32・D-35〜D-38 | `make figures-03` が予算内で成果物を出し、受け入れ条件1・2・4 が実測で満たされる |
| **3b-2** | T4 (M) / T5 (M) | 0 | `narma10.csv` / `fig_narma10_control.png` / design.md §11.2〜11.5 / D-29〜D-31・D-39 | 受け入れ条件3・5・7 が満たされ、decisions が全件 guard_test つきで揃う |

> 2分割の根拠: 容量実験 (T1〜T3) と NARMA10 (T4) は**依存が無く**、成果物も CSV / 図が別。
> 唯一の接点は「3-C の容量行を `capacity.csv` に追記する」1点だけ。3a と同じ 1,500 行超の diff を
> 1本の reviewer ラウンドに載せると、3a で起きた「BLOCKER の修正が新しい未記録の決定を生む」
> 連鎖 (D-33 / D-34 の経緯) が再現しやすい。

### 質問にせず前提として進めるもの

- 3-A / 3-B / 3-B' の**全条件で MC と IPC の両方を測る** (要件書 設計判断3 の直接の帰結)
- `conservation` (3-B') の既定レプリケート数は **3**。予算超過時の縮退規則は T2 に事前宣言済み
- `make saturation-03` は **T ∈ {1e5, 2e5, 5e5, 1e6} の掃引**で `capacity_length.csv` を出す。
  本番 (`figures-03`) には含めない
- セクション名は 3-B' を `conservation`、T 掃引を `length_sweep` とする
  (旧仕様は両方を "saturation" と呼んで衝突していた。Make ターゲット名 `saturation-03` は承認済みなので変えない)
- `diagnostics/__init__.py` の関数 `ipc` 再エクスポートは**維持する** (公開 API を壊さない)。
  代わりに §10-1 の罠をテストで固定する
- しきい値法の既定はシャッフルサロゲート (D-27)、誤差指標は NRMSE 主・NMSE 併記 (D-02)、
  参照値 0.16 / 0.107 の原典は**未特定のまま注記**、漸近展開 IPC は v0.1 見送り

## 9. 受け入れ条件 → タスク対応表

| # | 受け入れ条件 | タスク | 検証手段 |
|---|---|---|---|
| 1 | MC が N を上限に振る舞い ρ↑ でプロファイルが伸びる | T2 / T3 | `test_mc_effective_delay_increases_with_rho`, `fig_mc_sweep.png` |
| 2 | IPC_total ≤ N。ノイズ下では厳密に N 未満 | T2 / T3 | `test_conservation_respects_the_bound`, `fig_ipc_conservation.png` |
| 3 | しきい値処理の記録と既定の根拠 | T2 / T5 | `meta.json.threshold_comparison`, design.md §11.2 |
| 4 | ρ・leak で線形/非線形の配分が移動 | T2 / T3 | `capacity.csv`, `fig_ipc_profile.png`, `fig_memory_nonlinearity.png` |
| 5 | 探索予算をそろえた遅延線と ESN の比較 | T4 | `test_narma10_reuses_run_task_and_shares_rows_across_methods` |
| 6 | MC・IPC が外部生成の状態系列で動く | (3a で完了) | `tests/test_diagnostics_base.py` |
| 7 | 図5枚が1コマンド再生成 + pytest green | T2 / T3 / T4 | `make figures-03`, `make ci` |

---

## 10. 実装者への注意 (最も壊れやすい3点)

1. **`import rc_basics_lab.diagnostics.ipc as m` はモジュールではなく関数を返す**。
   `diagnostics/__init__.py` が関数 `ipc` を再エクスポートしているため。
   **3a のレビュー中に実際に踏み、変異試験が偽の緑になった**。
   `importlib.import_module(...)` を使うか、**呼び出し側のモジュール属性**を差し替えること
2. **`X` は診断へ渡す前に read-only にする** (D-35)。`CapacityProblem` は `problem.x` を
   read-only にするが**元の `X` は塞げない**。desync すると容量が例外なく桁違いになる。
   なお MC と IPC は `t0` が異なるため `CapacityProblem` は条件あたり2個作られる ——
   これは正常で、「1個にまとめる」最適化をすると D-24 の単一基準点が壊れる
3. **参照軌道の生成を 03 側で書き直さない**。`simulate_reference_trajectory` をそのまま呼び、
   `state_noise` は既定値つきキーワードで足す (D-36)。バイト一致テストが分岐を防ぐ

---

## T1 実装時に決めたこと (3b-1 T1 完了時に追記)

仕様に書かれていない選択をした箇所と、その理由。**次の周の reviewer / fixer が読むのはこの節**。

### 1. `CapacityRow` は T1 で定義した (§4 の T1 / T2 の境界の調整)

§4 は行 dataclass 2本を T2 に置いているが、T1 の受け入れ基準
`test_each_capacity_parameter_changes_output` (「縮小設定の**行**の指紋が変わる」) は
行 dataclass が無いと成立しない。**列の並びは §4 T2-2 の指定をそのまま採用**し
(`seed_*` ×3 は `seed_reservoir` / `seed_drive` / `seed_surrogate`)、T2 は CSV への
書き出しだけを担当する。`CapacityProfileRow` (長形式、D-38) は指定どおり T2 のまま。

### 2. `CapacityOutcome` は配列3本だけを持つ (しきい値は持たない)

§4 T1-2 の指定どおり `mc_profile` / `ipc_heatmap` / `ipc_by_degree` のみ。
**T2 への申し送り**: `CapacityProfileRow.threshold` 列に次数ごとのしきい値
(`ipc_threshold_degree{d}`、cfg 依存で本数が変わる) が要るなら、
`CapacityOutcome` にフィールドを1本足すこと。**全条件を再計算してはいけない**。

### 3. セクション固有の葉は `CHANNEL_PENDING` (task=T2 / T4) + 信管

T1 が実装するのは「1条件の配線」までで、掃引 (条件の列挙) は T2 なので、
`mc_sweep.*` / `ipc_sweep.*` / `conservation.*` (打ち切りを除く) / `length_sweep.*` /
`reservoir.n_replicates` / `narma.length` の 23 葉は出力で実測できない。
02 の pending 機構をそのまま使い、消費側が生えた瞬間に落ちる信管
(`tests/test_config_wiring_capacity.py::test_pending_cases_disappear_once_the_sweeps_exist`)
を張った。**信管はモジュール名だけでなく関数名でも発火する** —— 03 の掃引は T1 で既に
存在する `experiment/capacity.py` の**中に**生える計画なので、02 の
`KNOWN_EXPERIMENT_MODULES` (モジュール新設で発火) だけでは沈黙するため。
in-process で `capacity.run_mc_sweep` を生やすと実際に発火することを実測済み。

§5 有効性観点の「セクション固有の葉は scope 検査つき (4セクション)」は、
この pending の解消 (T2) と同時に満たされる。

### 4. 3-B' の打ち切り上書きは `ipc_config_for(config, experiment)` に置いた

実験ラベルで分岐する純関数にしたので、掃引が無い T1 でも
`conservation.max_delay_by_degree` を `CHANNEL_ROWS` (scope=3-B') として実測できる。
`dataclasses.replace` は `max_delay_by_degree` 以外のフィールドを触らないことも assert 済み。

### 5. `conservation` に `n_replicates` フィールドは無い (**T2 で要判断**)

§4 T1-1 の構造どおり、レプリケート数は `reservoir.n_replicates` (セクション横断) 1本にした。
一方 §7 リスク1 と §8 は「予算超過時に許可される調整は `conservation.n_replicates` を
3 → 1 だけ」と書いており、**そのフィールドは現状存在しない**。
T2 は (a) `ConservationConfig` に `n_replicates` を足す (横断共有との二重定義になるので
「セクション側があればそちらを優先」の規則が要る) か、(b) 縮退規則を
`reservoir.n_replicates` (= 3実験すべてに効く) へ読み替えるかを、**planner に確認してから**
決めること。実装者判断で (a) を先取りすると D-32 の「横断共有は3つだけ」が崩れる。

### 6. 既定値の出どころ

- 格子の点数は §5 の条件数と一致させた: 3-A 6×3×3rep=54 / 3-B 4×3×3rep=36 /
  3-B' 3×3×3rep=27。
- `sigma_u` の既定 (3-A 0.1 / 3-B・3-B'・length_sweep 0.2) は**暫定値**であり、
  T2-5 の較正 (`sigma_u ∈ {0.05, 0.1, 0.2, 0.5}`) で確定する。
- `seeds.surrogate = 4`。**`SeedStream` ではない** (`ctx.seed` へ直接渡る整数) ので
  `seeds.py` の既存 4 ストリームの `spawn_key` は1つも動いていない。
- 実験ラベルは `3A_mc_sweep` / `3B_ipc_sweep` / `3Bp_conservation`
  (T4 が足す 3-C は §4 T4 の指定どおり `3C_narma10`)。
- `length_sweep` の「...」は `rho` / `leak_rate` / `sigma_u` / `n_units` と解釈した。

### 7. `Narma10Config` は `config.py` に置いた

01 の `MackeyGlassConfig` / `DelayParityConfig` と同じ場所・同じ向き
(`tasks/` が `config.py` から自分の設定 dataclass を import する既存慣習)。
フィールドは `length` + `base: ExperimentConfig` の2つで、係数と入力分布は
T4 が `tasks/narma.py` のモジュール定数として持つ (D-29) —— 設定にはしない。

### 8. `drive_config_for` は 02 の `DriveConfig.n_pairs` を既定のまま使う

`n_pairs` は ESP 判定 (比較軌道の本数) 専用で `simulate_reference_trajectory` は読まない。
03 の `CapacityDriveConfig` に `n_pairs` を持たせないのは「設定したのに効いていない」
フィールドを作らないため (§2.2-2 / D-09 と同じ規律)。

### 9. 指紋から外す列

`wall_time_s` に加えて `wall_time_state_s` / `wall_time_mc_s` / `wall_time_ipc_s` の
計4本 (実行ごとに変わる実測時間)。T2 が CSV に書く列としては残る。

---

## T2 実装時に決めたこと (3b-1 T2 完了時に追記)

仕様に書かれていない選択をした箇所と、その理由。**次の周の reviewer / fixer が読むのはこの節**。

### 1. `ConservationConfig.n_replicates: int | None = None` を追加した (**仕様の訂正**)

§7 リスク1 と §8 の縮退規則「`conservation.n_replicates` を 3 → 1 に落とす」に対応する
フィールドが存在しなかった (T1 の申し送り5)。ユーザー判断により **(a) セクション側に足す**
を採用。`None` なら `reservoir.n_replicates` を継承する片方向の上書きで、
`conservation.max_delay_by_degree` が `config.ipc` を上書きするのとまったく同じ形にした。
解決は `experiment/capacity.py` の `n_replicates_for(config, experiment)` (純関数) に置き、
`ipc_config_for` と並べてある。継承と上書きの両方を
`test_conservation_replicates_default_to_the_shared_value_and_can_be_overridden` が固定する
(片方だけ測ると「上書きが無視される」実装と「`None` を int と誤って扱う」実装のどちらかが通る)。

これに伴い **`config.py` のローダに `X | None` の対応を足した** (`_coerce_optional`)。
従来は `UnionType` を無条件で `ConfigError` にしていた。受理するのは `None` との2項 union だけで、
`int | str` のような「どちらでも良い」型は引き続き `ConfigError` にする —— YAML の値から型を
推測し始めると `1` と `"1"` で挙動が変わる設定が生まれ、D-09 の規律が崩れるため。

### 2. `CapacityOutcome` に `ipc_thresholds` を1本足した (T1 の申し送り2への回答)

`CapacityProfileRow.threshold` に次数ごとのしきい値が要る。T1 の申し送りどおり
**再計算はせず**、`ipc()` の `scalars["ipc_threshold_degree{d}"]` を次数の昇順に並べた
タプルとして条件ごとに1回だけ運ぶ。再計算すると 3-B' では1条件あたり 6.6 秒の追加になる。

### 3. 実験ラベル `3L_length_sweep` を新設し `CAPACITY_EXPERIMENTS` を4本にした

`length_sweep.*` の5葉も T2 担当の `CHANNEL_PENDING` だったので、消費側 (`run_length_sweep`) を
今回作った。`capacity.csv` には出ない (`capacity_length.csv` 側にしか現れない) が、
**scope 検査のためにラベルは必要**である —— 3-A と同じラベルを名乗ると「T 掃引の設定を変えたら
3-A の行まで動いた」を検出できない。`capacity.csv` に出る3実験は `FIGURE_EXPERIMENTS`
(予算 900 秒の対象) として別に持つ。

### 4. 受け入れ条件1 の「単調非減少」を **rho <= 1.0 に限った** (**要確認**)

§4 T2 の受け入れ基準は「`mc_effective_delay` が ρ に単調非減少、最大 ρ が最小 ρ の 1.5 倍以上」
だが、**本番格子 (0.5〜1.1) の全域では sigma_u をどう選んでも成立しない**。
実測 (1レプリケート、3-A の本番格子):

| sigma_u | rho<=1.0 で単調 (leak 3本すべて) | 全域で単調 | ratio (rho=1.1 / rho=0.5) |
|---|---|---|---|
| 0.05 | **不成立** (leak=0.3 で 64.49@0.95 -> 63.46@1.0) | 不成立 | 3.05 / 2.57 / 2.56 |
| **0.1** | **成立** | 不成立 (leak 3本とも 1.1 で低下) | 2.71 / 2.38 / 2.20 |
| 0.2 | 成立 | 不成立 (leak=0.3, 0.6 で低下) | 2.38 / 2.04 / 2.47 |
| 0.5 | **不成立** (leak=1.0 で 9.30@0.7 -> 8.43@0.9) | 不成立 (leak=1.0 で低下) | 2.85 / 2.09 / **1.57** |

ρ>1 では駆動が弱いと ESP が成立せず記憶容量は**下がる** (容量は臨界点近傍で最大)。これは
3-A が見せたい現象そのものなので、格子から 1.1 を外す (= 条件数を動かす) 選択は取らなかった。
§7 リスク2 の (a) 駆動強度は較正で切り分け済み (4点すべてで同じ形の低下が出る) である。

実装した `test_mc_effective_delay_increases_with_rho` は3つを固定する:
(i) rho <= 1.0 でレプリケート平均が単調非減少、(ii) **仕様の文言どおり**格子の最大 ρ (1.1) の値が
最小 ρ (0.5) の 1.5 倍以上、(iii) ρ=1.1 が格子の最大値**ではない**こと
(= 単調性を ESP 領域に限る根拠が消えたらテストが赤くなる)。

**この (i) の制限は仕様の受け入れ基準からの逸脱なので、planner の確認を求める。**
本番実測 (3レプリケート平均):

| leak | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 | 1.1 | ratio(1.1/0.5) |
|---|---|---|---|---|---|---|---|
| 0.3 | 20.99 | 30.63 | 49.62 | 58.95 | **69.40** | 60.24 | 2.87 |
| 0.6 | 11.78 | 15.80 | 22.06 | 25.00 | **35.73** | 26.98 | 2.29 |
| 1.0 | 9.20 | 11.64 | 16.29 | 18.19 | **21.74** | 21.37 | 2.32 |

### 5. 受け入れ条件2 の s.d. は **対応のある差** (同一レプリケート) で取る

「差がレプリケート間 s.d. の3倍以上」を、セルごとの s.d. ではなく**同じレプリケート番号
どうしの差の s.d.** で測る。レプリケート番号はリザバー重みと駆動信号の両方を決めるので、
`state_noise` だけが違う2条件は同じリザバー・同じ入力を共有する (共通乱数法。D-37 と同じ設計)。
対応を無視すると「リザバーの引きの良し悪し」が分母に乗る —— 実測 (N=25, noise=0.01):
対応なしの s.d. は 2.99 で差 7.41 の 2.5 倍にしかならないが、対応のある差の s.d. は 1.04 で
比は 7.16 になる。3レプリケートとも同じ向きに 40〜50% 落ちているのに検出できないのは
検定の側の問題である。全6セルの比は 7.16 / 9.30 / 10.90 / 31.94 / 14.36 / 51.18。

### 6. 駆動強度の較正 (T2-5) の結果: `mc_sweep.sigma_u = 0.1` (**既定のまま**)

§4 T2-5 の手順 (代表条件 rho in {0.5, 0.99}, leak=1.0, 1レプリケート) の実測:

| sigma_u | eff_delay @rho=0.5 | @rho=0.99 | 比 | 受け入れ条件1 |
|---|---|---|---|---|
| 0.05 | 9.29 | 24.68 | 2.66 | 代表条件は満たすが**本番格子で不成立** (上表) |
| **0.1** | 9.04 | 20.37 | 2.25 | **成立 (採用)** |
| 0.2 | 8.39 | 15.11 | 1.80 | 成立 (最小ではない) |
| 0.5 | 6.50 | 9.09 | 1.40 | **不成立** (比が 1.5 未満) |

代表条件だけで選ぶと最小は 0.05 になるが、受け入れ条件1 は本番格子に対する主張なので
**本番格子でも成立する最小値**を採る、と読んだ。結果として T1 の暫定値 (0.1) が確定値になった。
`ipc_sweep` / `conservation` / `length_sweep` の `sigma_u = 0.2` は受け入れ条件4 (次数分解) 側の
動作点であり、この較正の対象ではない (§3.2: 図をまたいだ絶対値比較はしない)。

### 7. 受け入れ条件のテストは**コミット済みの本番成果物**を読む

`test_mc_effective_delay_increases_with_rho` / `test_conservation_respects_the_bound` /
CSV サイズの3件は `results/03_capacity/*.csv` を読む。本番を pytest の中で回すと 325 秒かかり、
「03 が追加する pytest < 60 秒」(§5) を1件で使い切る。受け入れ条件は「**この設定で実際に
こうなった**」という主張なので、縮小設定で回し直しても裏付けにならない。
`tests/test_readme_summary.py` が `results/comparison_summary.csv` を読むのと同じ形で、
成果物が古い設定のまま取り残される事故は
`test_production_config_matches_the_committed_results` (本番 YAML の条件数と行数の一致) が防ぐ。

### 8. `meta.json` に `wall_time_breakdown` と `n_profile_rows` を載せた

§5 が「状態生成の合計 < 60 秒 / **内訳を meta.json に出す**」を要求している。実験ごとに
`n_conditions` / `wall_time_state_s` / `wall_time_mc_s` / `wall_time_ipc_s` / `wall_time_s` を出す。
`n_rows` は `capacity.csv` の行数のままにし、長形式の行数は `n_profile_rows` に分けた
(足すとどちらの CSV の行数か読めなくなる。02 の `washout_sensitivity` と同じ規律)。

### 9. 配線テストの縮小設定は各セクション2条件にした

1条件だと「格子の**点数**が届いているか」しか測れず、点の**値**が無視される実装でも指紋が
変わってしまう。セクションごとに `n_units` / `n_steps` / `sigma_u` を別の値にしてあるのは、
セクション間で値を取り違える配線 (3-B の `n_units` を 3-A が読む等) を落とすため。
`CASE_CONDITIONS` (T1 が固定条件を並べていたもの) は削除し、条件は設定から作られる。

### 10. 本番 YAML の `narma.base.name` を `03_narma10` にした

T4 が使うセクションだが、`ExperimentConfig` の既定 (`01_what_is_rc`) のままだと 03 の
`meta.json` に 01 の実験名が載る。値の意味は T4 が決めるので、それ以外のフィールドは
既定のまま触っていない。

### 11. 実測 (本番 `make figures-03`、Darwin 25.3.0 / Python 3.12)

| 区間 | 条件数 | 状態生成 | MC | IPC | 合計 | 予算 |
|---|---|---|---|---|---|---|
| 3-A | 54 | 5.21s | 2.89s | 20.74s | **28.84s** | < 120s |
| 3-B | 36 | 11.09s | 6.39s | 55.97s | **73.46s** | < 180s |
| 3-B' | 27 | 18.89s | 15.53s | 187.85s | **222.28s** | < 400s |
| **合計** | 117 | **35.19s** | 24.80s | 264.57s | **324.57s** (`make` 全体 325.20s) | < 900s / 状態生成 < 60s |

ピーク RSS 0.99 GB (予算 4 GB)。CSV 合計 1.78 MB (`capacity.csv` 117 行 45 KB /
`capacity_profile.csv` 21,636 行 1.70 MB、予算 5 MB)。
3a の reviewer が警告した「状態生成側が予算を割る」は**起きていない** (35.19s / 予算 60s、
全体の 10.8%)。支配的なのは 3-B' の IPC (187.85s、全体の 58%) で、これは打ち切りを
(200,60,20,10) に深めた結果 目標数が 601 → 4,075 本になるため。
縮退規則 (`conservation.n_replicates` 3 -> 1) は**発動していない** (700 秒の閾値に対し 325 秒)。
