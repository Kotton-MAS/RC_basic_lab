# 仕様: rc-basics-03 —— リザバーの能力を測る (MC と IPC)

*要件書: `docs/要件_rc-basics-03.md` / サーベイ: `docs/rc-basics-survey.md` Q2・Q3*
*前提: サイクル1・2a・2b 完了 (439 tests / D-01〜D-22 / `make ci` 緑)*

> ✅ **確定 (ユーザー承認済み・2026-08-18)**: サイクルを2つに割る。
>
> - **3a (測定装置)**: T1・T2 + T5 のうち D-23〜D-28 と `design.md` §11.1
> - **3b (実験と記事の図)**: T3・T4 + T5 の残り (D-29〜D-32、図5枚、実測表)
>
> 3a の終了条件は「診断2本が外部生成の状態行列で動き、保存則が数値で確認できる」(受け入れ条件1・2・3・6)。
> 3b は 3a の診断を配線するだけになるので、実験層の設計判断が測定装置の設計に混ざらない。
>
> ✅ **§8 の3問も確定 (すべて推奨案)**:
>
> | 問 | 決定 |
> |---|---|
> | Q1 IPC のリザバー規模 | **IPC は N=50、MC は N=200** (D-32、3b で記録) |
> | Q2 時間予算 | `make figures-03` < 900秒 (本番) / `make saturation-03` < 1800秒 (別枠・手動) |
> | Q3 `diagnostics → readout.ridge` | **許可**し、guard の禁止対象に `config` を明示的に足して強化 (**D-23 採択**) |
>
> → D-23 は確定したので T1 に着手してよい。

---

## 1. ゴール

リザバーの容量 (MC・IPC) を測る**移植可能な診断2本**を実装し、それを使って ρ・リーク率に対する
容量の移動と、公平な対照下での NARMA10 比較を再現可能な形で出す。

---

## 2. 現状認識

### 2.1 関連箇所 (すべて実測で確認済み)

| 用途 | 場所 | 03 での使い方 |
|---|---|---|
| 診断の署名契約 | `diagnostics/base.py:69-136` | `DiagnosticContext` はそのまま。**フィールドを1個も足さない** |
| Gram 再利用の回帰 | `readout/ridge.py:83-124` | `fit_ridge_from_gram` の `rhs` は **`(F, D_out)` の多出力**を受ける。IPC の 2000 目標は「1本の Gram + 幅 K の rhs」で1回の `solve` に畳める |
| 設計行列 | `readout/design.py:206-231` | 3-C はここを一切触らない (`DelayLineSpec(n_lags)` が遅延線対照そのもの) |
| 01 の公平性機構 | `experiment/runner.py:161-333` | **`run_task(config, task_entry)` を 03 から直接呼ぶ**。`build_tasks` には手を入れない (D-31) |
| 02 の軌道生成 | `experiment/esp.py:88-124, 223-265` | `make_drive` を再利用。切り出し後の参照軌道生成関数を使う |
| 設定ローダ | `config.py:403-478` | `load_config_as(path, cls)`。**`tuple[X, ...]` の要素はスカラのみ**。打ち切り表は `tuple[int, ...]` で表す |
| 実験レジストリ | `main.py:76-96` | `"03"` を1行足す。`out_dir=results/03_capacity` |
| 診断の自動列挙 | `tests/test_diagnostics_base.py:116-243` | 診断を足すと `KNOWN_DIAGNOSTICS` と `MINIMAL_VALID_INPUT` の**両方**への登録が必須 |
| 配線テストの雛形 | `tests/test_config_wiring_esp.py`, `tests/wiring.py` | 03 用に同型 (`test_config_wiring_capacity.py`) を作る |
| 公開 API | `tests/test_public_api_reexport.py:23-65` | 新モジュールは `__init__.py` からの再エクスポート必須 |

### 2.2 既存の慣習で守るもの (3個)

1. **CSV の列順は行 dataclass の宣言順が単一の真実**
2. **設定 dataclass は純データ、値域検証は使う側**。未知キーは `ConfigError` (D-09)
3. **図のラベルは `labels.label(ja, en, cjk=...)` を通す** (D-10)

### 2.3 影響範囲

- 追加: `diagnostics/memory_capacity.py` / `diagnostics/ipc.py` / `diagnostics/_capacity.py` (共有カーネル、非公開) /
  `tasks/narma.py` / `experiment/capacity.py` / `experiment/narma.py` / `experiment/capacity_pipeline.py` /
  `plotting/figures_capacity.py` / `experiments/03_capacity/{config.yaml,run_03.py}`
- 変更: `config.py` / `main.py` (1行) / `Makefile` (2ターゲット) / 各 `__init__.py` / `README.md` /
  `docs/design.md` / `.claude/decisions.yaml`
- **変更しない**: `experiment/runner.py` の `build_tasks` / `ExperimentConfig` / `seeds.py` の既存ストリーム index /
  01・02 の成果物 (バイト単位で不変であることを確認する)

---

## 3. 前提・制約

### ハード制約 (絶対に変えない)

- **D-01**: 診断の署名は `f(X, u=None, y=None, *, ctx)`。`DiagnosticContext` に**フィールドを足さない**
  (`seed` は既にある)。診断固有パラメータは既定値つきキーワード `cfg` (D-15)
- **D-12**: `diagnostics/` は `config` / `reservoir` を (推移的にも) import しない
- **D-13**: 実験ごとに設定 dataclass を分ける。`ExperimentConfig` に 03 のフィールドを1個も足さない
- **D-04 / D-05 / D-08**: 3-C は 01 の `run_task` 経路をそのまま通す
- **D-02**: 誤差指標は NRMSE。CSV には `nmse` も併記済み (先行研究比較に使う)
- **D-09** / **D-10**
- 型: Python 3.12+ / `Any` 禁止 / `# type: ignore` を理由なく使わない

### ソフト制約 (理由があれば変えてよい)

- `config.py` は **package 化しない** (03 では見送る)。現状 410 行、03 で +250 行程度。
  **着手条件を明文化する: 600 行 (非空) を超えたら 04 の冒頭で独立タスクとして package 化する**
- 03 は 02 の `DriveConfig` / `ReservoirSweepConfig` を**再利用しない**
  (`n_pairs` のような 03 で死ぬフィールドが混ざり、配線テストを満たせなくなる)。**関数** (`make_drive`) は再利用する
- 図は5枚 (要件書の4つ + 受け入れ条件2・3 の一次資料として `fig_ipc_conservation.png`)
- 漸近展開による IPC 推定 (arXiv:2502.15769) は v0.1 では実装しない (要件書 未確定5)

---

## 4. タスク分解

### T1: 容量カーネルと MC 診断 (想定所要: **L**) 〔3a〕

1. `diagnostics/_capacity.py` (非公開・共有カーネル):
   - `CapacityProblem`: `Phi = [1, X]` の Gram `G = PhiᵀPhi` を**1回だけ**作って保持する frozen dataclass
   - `capacity_of_targets(problem, Z_chunk, alpha)`: `rhs = PhiᵀZ` を作り `fit_ridge_from_gram` を**1回**呼ぶ。
     容量は Gram 量だけで閉じる形 `C_k = 1 - (zᵀz - 2 w_kᵀ rhs_k + w_kᵀ G w_k) / (zᵀz)` で計算
     (`Phi` も `Z` も再走査しない)
   - `orthonormal_basis(u_lagged, degree, distribution)`: 宣言された入力分布に対して**正規直交**な多項式列
     (`uniform`→`sqrt(2n+1) P_n(u/a)`、`a = sqrt(3) sigma_u` / `normal`→`sigma_u` で正規化した確率論的 Hermite)
2. `diagnostics/memory_capacity.py`:
   - `MemoryCapacityConfig`: `max_delay=400`, `alpha=1e-9`, `threshold_mode="surrogate"`,
     `n_surrogates=100`, `surrogate_quantile=0.99`, `chunk_size=256`
   - scalars: `mc_total` / `mc_total_raw` / `mc_threshold` / `mc_effective_delay` / `mc_ratio` / `n_delays`
   - arrays: `mc_profile`
   - 行選択は `t0 = max(ctx.washout, max_delay)`、**全遅延で同一行集合** (D-24)

**受け入れ基準**

- `tests/test_diagnostics_base.py` が緑 (署名契約・keyword-only・レジストリ完全性)。
  `MINIMAL_VALID_INPUT` には **`u` を伴う専用ファクトリ**を登録する
- `test_mc_total_does_not_exceed_n_units`: N=30, T=5000, ρ∈{0.5,0.9,0.99} で `mc_total <= N * 1.02`
- `test_mc_profile_lengthens_with_spectral_radius`: `mc_effective_delay` が単調非減少、
  ρ=0.99 が ρ=0.5 の 1.5 倍以上
- `test_basis_is_orthonormal_under_declared_input_distribution`: T=200000 で相互相関が `|G/T - I| < 0.02`。
  `distribution="normal"` × Legendre は `ValueError` (D-28)
- `test_capacity_is_monotone_decreasing_in_alpha` (D-25)
- 外部生成の状態系列で完走 (受け入れ条件6) / 単体テスト1本 5 秒以内

### T2: IPC 分解とサロゲートしきい値 (想定所要: **L**) 〔3a〕

1. `diagnostics/ipc.py`: `IpcConfig`: `max_delay_by_degree=(60,20,10,6)`, `max_variables=3`,
   `basis="legendre"`, `input_distribution="uniform"`, `alpha=1e-9`, `threshold_mode="surrogate"`,
   `n_surrogates=100`, `n_surrogate_targets=4`, `surrogate_quantile=0.99`, `chunk_size=256`, `max_targets=200000`
2. **目標の列挙**: 次数 d の目標は「相異なる遅延 `k_1 < ... < k_m` (`m <= max_variables`)、各次数 `n_i >= 1`、
   `Σ n_i = d`、`k_i <= max_delay_by_degree[d-1]`」の全組合せに対する `Π_i P_{n_i}(u[t-k_i])`。
   `max_targets` を超えたら `ValueError` (**黙って切り詰めない**)
3. **ヒートマップの集約規則**: 各目標を `(次数 d, max(k_i))` のセルに割り当てて容量を合算する
4. **しきい値 (D-27)**: 次数ごとに、実目標から決定的に `n_surrogate_targets` 本を選び、
   時間シャッフルで `n_surrogates` 本ずつ作り、**通常の目標とまったく同じ経路**に流す。
   乱数は `ctx.seed` のみ。`surrogate` かつ `ctx.seed is None` なら `ValueError`。
   `chi2` (Dambre 2012 SupMat 3.2) と `none` も実装
5. **性能構造 (D-26)**: 目標は `chunk_size` 列ずつ生成し `rhs` に畳んだら破棄。
   `fit_ridge_from_gram` の呼び出し回数は `ceil(K/chunk_size)` に比例し、**目標数 K には比例しない**

**受け入れ基準**

- scalars: `ipc_total` / `ipc_total_raw` / `ipc_linear` / `ipc_nonlinear` / `ipc_threshold_degree1..d` /
  `n_targets` / `n_targets_kept` / `saturation_ratio`
- `test_ipc_total_does_not_exceed_n_units`: N=25, T=50000 で `ipc_total <= N*1.02` かつ
  **`saturation_ratio >= 0.5`** (保存則が空虚に成立するだけの状態を弾く)
- `test_state_noise_strictly_reduces_total_capacity`: 差がレプリケート間 s.d. の3倍以上
- `test_surrogate_threshold_zeroes_out_independent_targets`: 独立な乱数 X で閾値後が raw の 5% 未満
- `test_surrogate_threshold_requires_ctx_seed_and_is_reproducible` (D-27)
- `test_gram_solve_count_does_not_scale_with_target_count` (D-26)
- `test_all_targets_share_identical_rows` (D-24)
- `test_ipc_config_fields_change_output` / `test_threshold_mode_changes_total_capacity`

### T3: 設定層と実験 3-A / 3-B・図3枚・CLI (想定所要: **L**) 〔3b〕

1. `config.py` に `Capacity03Config` (D-13。`ExperimentConfig` には触らない):
   `seeds` / `drive` / `mc_sweep` (N=200, T=20000) / `ipc_sweep` (N=50, T=100000) /
   `saturation` (N∈{25,50,100} × noise 3点, T=200000) / `mc` / `ipc` / `narma`
2. `experiment/capacity.py`: 1条件 = `(rho, leak_rate, n_units, state_noise, replicate)`。
   **X は条件ごとに1回だけ作り、MC と IPC の両方に同じ配列を渡す** (要件書 設計判断3)。
   軌道生成は **02 から切り出された参照軌道生成関数を使う**
3. `experiment/capacity_pipeline.py`: `capacity.csv` / `ipc_profile.csv` (長形式) / 図3枚 / `meta.json`
4. `plotting/figures_capacity.py`: `plot_mc_sweep` / `plot_ipc_profile` / `plot_memory_nonlinearity` /
   `plot_ipc_conservation`
5. `main.py` に `"03"` を登録、`experiments/03_capacity/*`、`Makefile` に `figures-03` と `saturation-03`
6. `tests/test_config_wiring_capacity.py` を `test_config_wiring_esp.py` と同型で新設

**受け入れ基準**

- `test_all_capacity_config_fields_are_covered` + `test_each_capacity_parameter_changes_output`
- `test_mc_and_ipc_share_the_same_state_matrix`: 参照軌道生成の呼び出し回数が条件数と一致
- `test_reference_states_match_esp_simulate_condition`: 02 の `simulate_condition().states` と**バイト一致**
- `uv run python main.py --experiment 03 --out <tmp>` で CSV2枚 + 図4枚 + meta.json
- 縮小設定は 20 秒以内。本番は §5 の予算表を満たす

### T4: 実験 3-C —— 公平な対照での NARMA10 (想定所要: **M**) 〔3b〕

1. `tasks/narma.py`: 採用式 (D-29):
   `y[t+1] = 0.3 y[t] + 0.05 y[t] Σ_{i=0}^{9} y[t-i] + 1.5 u[t-9] u[t] + 0.1`、`u ~ U[0, 0.5]` i.i.d.
   発散 (`|y| > 1e3` または非有限) は `ValueError` (D-30)。**クリップも再抽選もしない**
2. `experiment/narma.py`: `TaskEntry` を組んで **01 の `run_task` を呼ぶ** (D-31)。
   `n_lags_grid = (10,15,20,25,30)` (要件書「タップ数 k=10〜30」)、`alpha_grid` は 01 と同一 (D-04)。
   探索予算は遅延線が `alpha × n_lags`、ESN が `alpha` のみ (D-08) = **先行と逆向きの非対称性**。
   あわせて 3-C の ESN の MC と軽い IPC を測り `narma10.csv` に併記
3. `plot_narma10_control`: 参照線 NMSE=0.16 / 0.107 を**注記つき**で描く。
   原典未特定であることを図の注と `meta.json` に残す

**受け入れ基準**

- `test_matches_reference_recurrence`: 手計算した先頭5ステップと `1e-12` 以内で一致 (D-29)
- `test_divergence_raises_instead_of_clipping` (D-30)
- `test_narma10_reuses_run_task_and_shares_rows_across_methods` (D-05・D-31)
- `test_narma10_alpha_grid_is_shared_across_methods` (D-04)
- `test_01_artifacts_are_unchanged`: `build_tasks` に NARMA10 が混入していない
- **結果の向きは問わない**。遅延線が上回った場合は `meta.json` の `narma10_verdict` に記録

### T5: 記録 (想定所要: **M**) 〔3a・3b に分割〕

- `docs/design.md` §11: 11.1 容量測定の定義と正規化 / 11.2 **しきい値法の比較** (なし・サロゲート・χ² の総容量、
  **既定を根拠つきで選ぶ**、受け入れ条件3) / 11.3 打ち切り表と目標数・実行時間 /
  11.4 NARMA10 の採用式と先行との差分 / 11.5 実測結果と実行時間表
- `.claude/decisions.yaml` に **D-23〜D-32**
- `README.md` に `make figures-03` / `make saturation-03`
- `tests/test_design_doc.py` に 03 の一次資料の照合を1件

---

## 5. 評価軸 (Check フェーズに渡す)

### 性能観点 (**予算を明示する**)

| 区間 | 予算 | 根拠 |
|---|---|---|
| `make figures-03` 合計 | **< 900 秒** | 02 と同じ予算 (02 実測 87.7 秒) |
| 3-A (MC 掃引, N=200, T=2e4, 18条件×3rep) | < 120 秒 | 実測 4.7us/step → 0.094 s/条件 |
| 3-B (IPC 掃引, N=50, T=1e5, 12条件×3rep) | < 300 秒 | 目標数 ≈ 750 + サロゲート 400 列 |
| 3-B' (保存則, N∈{25,50,100} × noise 3点, T=2e5) | < 300 秒 | 目標数 ≈ 4700 |
| 3-C (NARMA10, T=8000, 3手法×5rep) | < 120 秒 | 01 の実績と同規模 |
| `make saturation-03` (T=1e6) | **< 1800 秒 (予算外・手動)** | 要件書の入力長上端の確認 |
| 03 が追加する pytest 合計 | < 60 秒 | 既存 439 本の実行時間を崩さない |
| ピークメモリ | < 4 GB | T=1e6 × 2395 列を実体化すると 19 GB |

**禁止する構造**: 「(delay, degree) の組ごとに ESN を再実行する」。2395 通り × 4.7 秒 ≈ **3.1 時間**になる。

### 安全性観点

- 01・02 の成果物がバイト単位で不変
- `seeds.py` の既存4ストリームの `spawn_key` を動かさない
- `DiagnosticContext` のフィールドが増えていない (D-01)
- `diagnostics` の import が `config` / `reservoir` を引き込まない (D-23)

### 有効性観点 (**必須**)

- `test_each_capacity_parameter_changes_output`: `Capacity03Config` の全葉
- `test_ipc_config_fields_change_output`: `IpcConfig` の全フィールド
- **`chunk_size` だけは「変えても結果が一致する」ことを要求する** (性能パラメータ。結果が変わったら
  チャンク分割にバグがある)。専用テスト `test_chunk_size_does_not_change_results` として分ける (`rtol=1e-10`)

---

## 6. 意図的な決定 (D-23 以降)

```yaml
- id: D-23
  rule: "diagnostics/ は readout.ridge / readout.design を import してよい。config / reservoir は直接・推移のいずれでも禁止し、推移閉包で機械検査する"
  rationale: "D-12 の文言は『types 以外の自作モジュールを import しない』だが guard は reservoir だけを見ていた。IPC は同一 X に対する多数目標の回帰であり、fit_ridge_from_gram (バイアス列を正則化しない D-03 の唯一の実装) を使わないと同じ閉形式解を2箇所に持つことになり D-03 が片方だけで崩れる。readout.ridge は metrics/types しか import しないため移植性の前提は保たれる。禁止対象に config を明示的に足すことで guard は現状より強くなる"
  guard_test: "tests/test_diagnostics_base.py::test_diagnostics_package_does_not_transitively_import_reservoir_or_config"

- id: D-24
  rule: "MC / IPC は全目標を同一の行集合で回帰する。t0 = max(ctx.washout, 全目標の最大遅延) を単一の基準点とし、遅延ごとに使える行数を変えない"
  rationale: "遅延 k の目標は先頭 k 行が未定義なので、素直に書くと遅延が深いほど標本数が減り容量が系統的に下がる。これは『記憶が減衰している』という測りたい現象とまったく同じ向きに出るため図を見ても気づけない。01 の D-05 と同じ規律を容量測定の軸に適用する"
  guard_test: "tests/test_diagnostics_ipc.py::test_all_targets_share_identical_rows"

- id: D-25
  rule: "容量測定の回帰は固定の微小 alpha (既定 1e-9) を使い、検証分割による alpha 選択を行わない。D-04 の alpha 格子は 3-C にのみ適用する"
  rationale: "容量は『線形読み出しで到達可能な最大の説明率』という定義そのものであり正則化は系統的に過小評価する。一方 alpha=0 は N=200 の Gram (cond ~ 1e18, D-11 の実測) で solve が落ちうるため数値安定の下駄として 1e-9 を置く。D-04 を容量測定へ機械的に広げると Dambre 2012 の保存則と比較できなくなる"
  guard_test: "tests/test_diagnostics_ipc.py::test_capacity_is_monotone_decreasing_in_alpha"

- id: D-26
  rule: "IPC は X を条件ごとに1回だけ計算し、Gram も条件ごとに1回だけ作る。目標は chunk_size 列ずつ生成して rhs に畳み、fit_ridge_from_gram の呼び出し回数は ceil(K/chunk_size) に比例させる"
  rationale: "(delay, degree) ごとに ESN を再実行する素直な設計は実測 4.7us/step x T=1e6 x 2395 通り = 約3.1時間になり実験そのものが回らない。容量は Gram 量だけで閉じるので Phi も Z も再走査は不要。Z 全体を実体化しないのはメモリ制約でもある (19 GB)"
  guard_test: "tests/test_diagnostics_ipc.py::test_gram_solve_count_does_not_scale_with_target_count"

- id: D-27
  rule: "IPC のしきい値処理はシャッフルサロゲート法を既定とし、閾値は次数ごとに推定する。サロゲートは通常の目標と同一経路で計算する。乱数は ctx.seed のみから引き、surrogate かつ ctx.seed is None なら ValueError"
  rationale: "要件書 設計判断1 / サーベイ Q3 (Kubota 2021 系)。有限標本では容量が系統的に過大評価され、その量は目標の周辺分布に依存するため次数ごとに閾値を分ける。サロゲートを別経路で計算すると閾値と容量が別実装からずれるので同じ関数に流す。ctx.seed を必須にするのは、閾値が黙って非再現になると受け入れ条件3 の記録が意味を失うため"
  guard_test: "tests/test_diagnostics_ipc.py::test_surrogate_threshold_requires_ctx_seed_and_is_reproducible"

- id: D-28
  rule: "多項式基底は宣言された入力分布に対して正規直交化する。(input_distribution, basis) の未対応な組は ValueError にし、黙って Legendre として扱わない"
  rationale: "要件書 設計判断2。基底が入力測度に対して直交していないと容量が目標間で二重計上され保存則が破れる。破れ方は『N をわずかに超える』という穏やかな形で出るため図では正常に見える。正規化を振幅に追従させることで sigma_u を自由に振れる (D-17 と両立)"
  guard_test: "tests/test_diagnostics_ipc.py::test_basis_is_orthonormal_and_mismatched_pair_raises"

- id: D-29
  rule: "NARMA10 は y[t+1] = 0.3 y[t] + 0.05 y[t] sum_{i=0}^{9} y[t-i] + 1.5 u[t-9] u[t] + 0.1, u ~ U[0, 0.5] i.i.d. を採用する。記事にはこの式をそのまま載せる"
  rationale: "要件書 未確定3 / サーベイ Q2: 文献の添字が2系統に割れており、どちらを使ったかで数値が変わるのに明記されないことが多い。実験3-C の直接の先行 (Goudarzi 2014) がこの形なので比較可能性を優先する"
  guard_test: "tests/test_tasks_narma.py::test_matches_reference_recurrence"

- id: D-30
  rule: "NARMA10 の発散 (|y| > 1e3 または非有限) は ValueError にする。クリップも自動再抽選もしない"
  rationale: "NARMA10 は入力次第で発散する既知の系であり、黙ってクリップすると『クリップの飽和特性』という別の非線形性を手法に与えることになる (遅延線対照が不当に有利/不利になる)。再抽選も、どのシードが捨てられたかが記録されないと 3-C の公平性の外に穴が開く"
  guard_test: "tests/test_tasks_narma.py::test_divergence_raises_instead_of_clipping"

- id: D-31
  rule: "実験3-C は 01 の run_task を再利用し、TaskEntry を 03 側で組み立てる。build_tasks / ExperimentConfig に NARMA10 を足さない"
  rationale: "要件書 設計判断4。build_tasks に足すと 01 の comparison.csv に行が増えて 01 の成果物が変わり D-13 の分離も崩れる。run_task は task_entry を引数に取るので、公平性の3決定 (D-04/D-05/D-08) を1行も書き写さずに 3-C へ効かせられる"
  guard_test: "tests/test_experiment_narma.py::test_narma10_reuses_run_task_and_shares_rows_across_methods"

- id: D-32
  rule: "IPC 掃引のリザバーは N=50、MC 掃引は N=200 とし、実験ごとにリザバー規模を変える"
  rationale: "保存則の飽和を有限の打ち切りで見せるには必要な目標数が N とともに急増する。N=200 で飽和させるには打ち切りを深くする必要があり実行時間が予算 (900秒) を超える。N=50 なら深い打ち切りでも 1条件 30秒程度で saturation_ratio >= 0.5 に届く。MC は degree=1 のみなので N=200 のまま連載の連続性を保てる"
  guard_test: "tests/test_experiment_capacity.py::test_ipc_reservoir_is_smaller_than_mc_reservoir"
```

> D-23 は D-12 の適用範囲を明確化するもの (禁止対象に `config` を足して guard は強化される)。
> §8 Q3 の回答次第で採否が決まるため、**T1 着手前に確定させること**。

---

## 7. 想定リスク (起きたら止まって相談)

1. **IPC の実行時間が予算 (900 秒) を超える**。打ち切りを浅くする / 条件数を減らす / T を下げる の
   どれを削るかは記事の主張に直結するので、実装者判断で削らない
2. **深い打ち切りでも `saturation_ratio` が 0.5 に届かない**。原因が (a) 基底の正規化バグ
   (b) 打ち切り不足 (c) リザバーの実力 のどれかで対処が正反対になる。まず (a) を切り分ける
3. **3-C で遅延線が ESN を上回る**。結果としては受け入れ (記事になる) が、`n_lags` の上限と
   探索予算の非対称性が効いている可能性があるため、追加条件を回す前に相談する

---

## 8. 不明点 (3問)

**Q1. IPC 用リザバーの規模と保存則の見せ方**
- (a) **IPC は N=50・深い打ち切り、MC は N=200 (推奨)** — 予算内で飽和が見せられる。D-32 として記録
- (b) 全実験 N=200 で統一し打ち切りを浅くする — 保存則が「上限を大きく下回る」だけの図になり受け入れ条件2 が空虚
- (c) N∈{25,50,100} を掃引軸にし飽和の N 依存を図にする — (a) の上位互換だが 3-B' が3倍

**Q2. 実行時間の予算配分**
- (a) **本番 < 900 秒 + 深掘り `saturation-03` < 1800 秒を別ターゲット (推奨)**
- (b) すべて 900 秒に収める — 要件書の「入力長 10^5〜10^6」の上端が未確認のまま残る
- (c) 本番の予算を 3600 秒に — 反復が遅くなり PDCA の回転が落ちる

**Q3. D-12 の適用範囲 (`diagnostics` → `readout.ridge` の import)**
- (a) **許可し guard を「reservoir と config を推移的に引き込まない」へ強化 (推奨、D-23)**
- (b) 容量カーネルを `diagnostics/` 内に自前実装 — D-03 の実装が2箇所に分かれる
- (c) `fit_ridge_from_gram` を `linalg.py` へ移す — 01・02 の触らない予定のファイルに差分が出る

> **質問にせず前提として進める**: NARMA10 の式 (D-29) / しきい値法の既定 (シャッフルサロゲート) /
> 誤差指標 (NRMSE 主・NMSE 併記) / 漸近展開 IPC (v0.1 見送り) / 参照値 0.16・0.107 の原典 (未特定のまま注記)

---

## 9. 受け入れ条件 → タスク対応表

| # | 要件書の受け入れ条件 | タスク | 検証手段 |
|---|---|---|---|
| 1 | MC が N を上限に振る舞い ρ↑ で遅延プロファイルが伸びる | T1 / T3 | `test_mc_total_does_not_exceed_n_units`, `test_mc_profile_lengthens_with_spectral_radius` |
| 2 | IPC_total ≤ N。ノイズ下では厳密に N 未満 | T2 / T3 | `test_ipc_total_does_not_exceed_n_units`, `test_state_noise_strictly_reduces_total_capacity` |
| 3 | しきい値処理の有無で総容量がどれだけ変わるかを記録し既定を根拠つきで選ぶ | T2 / T5 | `test_threshold_mode_changes_total_capacity`, `meta.json.threshold_comparison` |
| 4 | ρ・リーク率で線形/非線形容量の配分が移動する | T3 | `capacity.csv`, `fig_ipc_profile.png`, `fig_memory_nonlinearity.png` |
| 5 | リッジ + 探索予算をそろえた遅延線と ESN の同一条件比較 | T4 | `test_narma10_reuses_run_task_and_shares_rows_across_methods`, `narma10.csv` |
| 6 | MC・IPC が外部生成の状態系列で動く | T1・T2 | `test_diagnostic_accepts_external_state_series` の対象拡張 |
| 7 | 図5枚が1コマンド再生成 + pytest green | T3・T4 | `make figures-03`, `make ci` |

---

## 10. 実装者への注意 (最も壊れやすい3点)

1. **`MINIMAL_VALID_INPUT` に `_minimal_input_no_extras` を割り当てない**。MC/IPC は `u` が無いと
   `ValueError` になるため `u` を伴う専用ファクトリを書く。`test_minimal_valid_input_actually_produces_a_result` は
   `suppress` を持たない別関数なので、安く緑にする逃げ道は無い
2. **`chunk_size` は結果を変えてはいけない**。他の全設定フィールドは「変えたら出力が変わる」ことを
   要求されるが、これだけは**逆向きの要求**である (`rtol=1e-10` で照合)
3. **参照軌道の生成を 03 側で書き直さない**。02 の `simulate_condition` と結果がバイト一致することを
   テストで固定する。ここが分岐すると「02 と 03 で同じ ρ を指しているのに別のリザバーを見ている」という、
   CSV を並べても気づけない食い違いになる

---

## T1 実装時に決めたこと (仕様外の判断)

仕様 §4 T1 に書かれていなかったため実装者が決めた事項。次周の reviewer / fixer は
コードではなくここを読む。実測値は `make ci` 時点のもの。

1. **容量の下限クリップ**: `capacity_of_targets` は容量を **下限 0 でクリップ**し、
   上限側はクリップしない。バイアス列があるので理論上 `C_k >= 0` (平均予測で残差 = 全分散)
   であり、負値は Gram 展開 (`z.T z - 2 w.T rhs + w.T G w`) の桁落ちによる数値誤差にすぎない。
   上限を切らないのは `C_k > 1` が基底や行合わせのバグを意味するため (隠すと保存則の破れが
   見えなくなる)。この決めにより `threshold_mode="none"` は `mc_total == mc_total_raw` になる。
2. **`mc_effective_delay` の定義**: **容量重心** `Σ k C_k / Σ C_k` (しきい値後のプロファイル上)。
   「しきい値を超える最大の遅延」にしないのは、サロゲート閾値が分位点である以上、深い遅延にも
   偽陽性が一定割合出て指標が `max_delay` に張り付くため (実測: N=30, T=5000, rho=0.5 で
   しきい値超えの最大遅延は 335 に達する)。容量が 0 のときは 0 を返す。
3. **基底の正規化は実測の平均と標準偏差で行う** (`(u - mean(u)) / std(u)` を作ってから
   一様なら `sqrt(3)` で割って Legendre、正規ならそのまま Hermite)。中心化まで行うのは、
   U[0, 0.5] のような非ゼロ平均の入力 (NARMA10、D-29) でも経験測度に対して直交させるため。
   次数1は分布によらず `(u - mean)/sigma` に一致するので、`MemoryCapacityConfig` は
   `input_distribution` / `basis` フィールドを持たない。
4. **正規化は系列全体で1回**。遅延ごとに標準化し直すと遅延ごとに別の測度で直交化することに
   なり、保存則が破れる。
5. **MC のしきい値法は `surrogate` と `none` の2つ**。`chi2` (Dambre 2012 SupMat 3.2) は
   T2 (IPC) で共有カーネルに足す。未知の値は既定へフォールバックせず `ValueError`。
6. **MC のサロゲート素材は遅延1の目標に固定**する。次数1の目標はどの遅延でも同じ周辺分布を
   持つのでシャッフル後の容量分布は遅延に依存せず、代表を乱数で選ぶと閾値が非再現になる。
   `MemoryCapacityConfig` に `n_surrogate_targets` は置かない (IPC 側の概念)。
7. **`surrogate` かつ `ctx.seed is None` は `ValueError`** (D-27 の規律を MC にも適用)。
   このため `tests/test_diagnostics_base.py` の `MINIMAL_VALID_INPUT` に登録する専用ファクトリは
   `u` に加えて `ctx.seed` を渡し、既定 `max_delay=400` で完走できる長さ (T=1200) の
   外部生成状態を自前で作る。
8. **`arrays` は `mc_profile` (しきい値後) の1本だけ**。しきい値前の情報は `mc_total_raw` が
   スカラで持つ。3-B の図で生プロファイルが要るなら T3 で足す。
9. **共有カーネルのチャンク API は「チャンクの iterable」を受ける**
   (`capacity_of_chunks(problem, chunks, alpha)`)。IPC が `(T, K)` 全体を実体化せずに
   `chunk_size` 列ずつ作って捨てられるようにするため (D-26 のメモリ制約 19 GB)。
   実体化済みの小さな目標 (サロゲート) 用に `iter_column_chunks` を添える。
10. **`CapacityProblem` は `phi` / `gram` / `bias_column` / `t0` を持つ frozen dataclass
    (`eq=False`)** で、Gram は `CapacityProblem.from_states(X, t0=...)` が構築時に1回だけ作る。
    `eq=False` は numpy 配列を持つ dataclass の `__eq__` が配列比較で `ValueError` になるため。
11. **D-12 の `guard_test` をテスト改名に追随させた** (`..._reservoir` →
    `..._reservoir_or_config`)。決定そのものは変えていない (D-23 が禁止対象に `config` を
    足して guard を強化した結果の改名)。
12. **decisions.yaml には D-23 に加えて D-24 / D-28 も追記した** (T1 で実装が完結し、
    guard_test を実在のテストに紐づけられるため)。仕様 §6 は両者の guard_test を
    `tests/test_diagnostics_ipc.py` に置く想定だが、T2 が未着手なので当面
    `tests/test_diagnostics_memory_capacity.py` の同等テストを指す。**T2 は IPC 側の
    guard_test へ差し替えること**。D-25 / D-26 / D-27 は IPC 固有の要素 (次数ごとの閾値、
    目標数非依存性) を含むので T1 では追記していない。
13. **基底の正規直交性テストの次数上限を分布で変えた**。IPC が実際に使う (uniform, legendre)
    は次数4まで T=200000 / 許容差 0.02 に収まる (実測: 3 シードで 0.005 / 0.005 / 0.013)。
    (normal, hermite) は `He_n` の2乗が重い裾を持ち `E[psi_n^2]` の**標本誤差**が大きいため
    次数4では T=200000 で 0.03〜0.18、T=2000000 でも 0.012〜0.021 と 0.02 に収まらない。
    そこでモンテカルロ検査は正規側を次数2までにし、基底の定義そのもの (バイアス) は
    Gauss-Legendre / Gauss-Hermite 求積で次数4まで誤差 1e-10 以下に固定する別テストを置いた。
14. **`test_mc_profile_lengthens_with_spectral_radius` は T=20000 で回す** (保存則テストは
    仕様通り T=5000)。有限標本による容量のかさ上げは `F/T` の桁で入り、T=5000 では 400 本の
    遅延に散った偽陽性が重心を押し上げて rho の効果を覆う (実測: T=5000 では 5 シード中
    5 シードで単調性 or 1.5 倍が破れる。T=20000 では 5 シードすべて通り比の最小は 1.75)。
    3-A の本番設定も T=20000 なので条件は本番寄りになる。
15. **テスト用リザバーの `input_scale` は 0.1** (準線形領域)。飽和領域 (1.0) では rho を
    上げても線形記憶が伸びず、受け入れ条件1 の現象自体が消える (実測: T=20000,
    input_scale=1.0 で 5 シード中 4 シードが 1.5 倍に届かない)。

---

## T2 実装時に決めたこと (仕様外の判断)

仕様 §4 T2 に書かれていなかったため実装者が決めた事項。次周の reviewer / fixer は
コードではなくここを読む。実測値は `make ci` 時点のもの (500 tests 緑)。

1. **遅延は 1 から始める** (`k_i >= 1`)。`k=0` (現在の入力) を目標に含めない。
   こうすると**次数1の IPC は MC と同じ量になる** (同じ遅延集合・同じ基底・同じ
   行合わせ) ので、`ipc_linear` と `mc_total` を並べたときに定義の差で食い違う
   ことがない。`k=0` を入れると IPC だけが「入力そのものの再現」を容量に数え、
   保存則の解釈が MC 側とずれる。
2. **`chi2` しきい値の式**: `threshold = chi2.ppf(q, df=N) / T_eff`
   (`N` はバイアス列を除く回帰変数の本数 = 状態次元)。状態と無相関な目標の
   決定係数が `chi2_N / T_eff` に漸近することによる (Dambre 2012 SupMat 3.2)。
   **次数に依存しない**ので `ipc_threshold_degree*` は全次数で同じ値になり、
   その差そのものが「次数ごとに推定する意味があるか」の一次資料になる
   (受け入れ条件3・§11.2 で使う)。分位点は `surrogate_quantile` を共用し、
   `chi2_quantile` のような別フィールドを足さない (D-15 の設定は増やさない方針。
   フィールド名が `surrogate_` 始まりなのに `chi2` でも読む点は docstring に明記)。
3. **サロゲート代表の選び方は「次数内の列挙順から等間隔」**。先頭から詰めて
   取らないのは、列挙順の先頭が「変数1本」の目標 (`psi_d(u[t-k])`) に偏っており、
   次数 `d` の周辺分布の代表として1形だけを見ることになるため。等間隔なら
   変数本数の違う目標が混ざる。乱数で選ぶのは D-27 が禁じている。
4. **列挙順は「次数 → 変数本数 → 遅延の組 (辞書順) → 次数の割り当て」で完全に
   決定的**にした。3 の代表選択がこの順序に依存するので、順序は再現性の一部で
   ある (`test_targets_are_enumerated_in_degree_order` が次数昇順を固定する)。
5. **しきい値を容量より先に計算する**。`ctx.seed` の渡し忘れを、K 本ぶんの回帰を
   回し切ってから告げるのではなく着手前に落とすため。乱数はしきい値でしか
   使わないので、順序を変えても再現値は変わらない。
6. **`max_targets` の検査は閉形式の数え上げ (`count_targets`) で行い、列挙する前に
   落とす**。実際に列挙してから数えると、打ち切りを1桁深くした設定
   (次数3・`max_delay=1000`・`max_variables=3` なら 5 億通り) で上限の検査に
   到達する前にメモリと時間が尽きる。`count_targets` と実際の列挙が一致することは
   `test_target_enumeration_matches_the_declared_rule` が5設定で固定する。
7. **`arrays` は3本**: `ipc_heatmap` (`(次数, 遅延)`、列 index `j` が遅延 `j+1`、
   幅は全次数の最大遅延で、打ち切りより深い列は 0) / `ipc_by_degree` (しきい値後) /
   `ipc_by_degree_raw` (しきい値前)。生の次数別容量を持つのは §11.2 の
   しきい値法比較がスカラ3つ (`ipc_total` / `ipc_total_raw` / 閾値) だけでは
   「どの次数が落ちたか」を書けないため。
8. **`ipc_nonlinear` は `ipc_total - ipc_linear`** (次数2以上の合計)。
   `saturation_ratio` は**しきい値後**の `ipc_total / N`。`n_targets_kept` は
   「しきい値後の容量が厳密に正である目標の本数」と定義した (しきい値を
   超えなかった目標と、下限クリップで 0 になった目標を同じ『残らなかった』側に
   置く。`threshold_mode="none"` でも容量 0 の目標は数えない)。
9. **容量の下限クリップは T1 の決定 (「T1 実装時に決めたこと」1) をそのまま継承**
    する (共有カーネルの `capacity_of_targets` が担当する)。このため
    `threshold_mode="none"` では `ipc_total == ipc_total_raw` になる。
10. **`IpcConfig` の全フィールド検査から `chunk_size` と `max_targets` を外した**。
    `chunk_size` は逆向きの要求 (§10-2)、`max_targets` は上限であって
    「超えたときに `ValueError` にする」ことが唯一の観測可能な効果なので、
    それぞれ `test_chunk_size_does_not_change_results` /
    `test_target_enumeration_raises_instead_of_truncating` が担当する。
    `basis` と `input_distribution` は**対で**意味を持つ (片方だけ変えると D-28 で
    `ValueError`) ため、対で入れ替えた設定を両フィールドの検査に使い、片方だけ
    変えたときに落ちることは D-28 の guard_test が固定する。
11. **`n_surrogate_targets` が次数内の目標数を超える場合は切り詰める**
    (`ValueError` にしない)。打ち切りが浅い次数 (例: 次数4 の `max_delay=1`) で
    代表が4本取れないのは設定ミスではなく打ち切りの当然の帰結であり、ここで
    落とすと浅い設定での比較実験ができなくなる。目標そのものの切り詰め
    (`max_targets`) とは性質が違う —— あちらは**測定対象**が黙って減るが、
    こちらは**閾値の推定精度**が下がるだけで、`n_surrogates` 本の
    シャッフルは変わらず走る。
12. **`n_surrogates` / `n_surrogate_targets` の値域検証は
    `threshold_mode="surrogate"` のときだけ行う** (MC の前例に合わせた。使わない
    設定値で落とすと `chi2` 比較のたびに無関係なフィールドを直す羽目になる)。
13. **`TargetSpec` / `count_targets` / `enumerate_targets` を公開する**
    (`__all__` に入れる)。3b の `ipc_profile.csv` (長形式) が目標の内訳を書き出す
    のに要るのと、目標数の見積り (§5 の予算表) をテストから直接引けるようにする
    ため。目標の**列**を作る `_target_column` / `_iter_target_chunks` は非公開の
    ままにした (D-26 のチャンク規律を外から破れないように)。
14. **テスト用リザバーの `input_scale` は 0.5** (MC 側は 0.1)。準線形すぎると
    総容量が次数1にほぼ集中し「次数分解」という測定対象そのものが痩せるため
    (実測 N=25・T=50000・rho=0.9: `input_scale=0.1` では次数3 の容量が 3.97、
    0.5 では 6.82)。
15. **既定設定の目標数は 601 本** (次数1: 60 / 次数2: 210 / 次数3: 220 /
    次数4: 111)。`test_default_config_enumerates_601_targets` が内訳ごと固定する。
    §5 の予算表が想定した「目標数 ≈ 750」と同じ桁で、実測は N=25・T=50000 の
    1条件が **0.83 秒** (サロゲート 1600 列込み)。仕様の目標数見積りは変えていない。
16. **保存則テストの実測値**: N=25・T=50000・rho=0.9・`input_scale=0.5` で
    `ipc_total = 17.076` (上限 25.5)、`saturation_ratio = 0.683` (要求 0.5 以上)、
    `ipc_linear = 10.245` / `ipc_nonlinear = 6.831`、`n_targets_kept = 257 / 601`。
    偶数次の容量がほぼ 0 (次数2 = 0.001、次数4 = 0.006) なのはバグではなく、
    バイアスの無い `tanh` リザバーが奇関数で、対称な入力に対して偶数次の
    多項式と相関しないため。3b で `input_scale` にオフセットを入れるなら
    ここが動くはずなので、図の読み方として §11 に残すとよい。
