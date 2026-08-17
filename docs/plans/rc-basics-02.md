# 仕様: rc-basics-02 —— ESN を動かして測る (ESP・スペクトル半径・リーク率)

*サイクル2 / 対象要件: `docs/要件_rc-basics-02.md` / 前提: サイクル1 (D-01〜D-12, make ci 緑)*

> ⚠ planner が **分割不足** と判定 (L タスク3本)。§8 Q2 の分割判断を先に決めること。

---

## 1. ゴール

リザバー実装に依存しない ESP 判定・条件付き Lyapunov 指数・実効時定数の診断層を作り、それを使って「入力があれば ρ>1 でも ESP は成立しうる」を図で示す (図4枚 + CSV2枚を1コマンドで再生成)。

---

## 2. 現状認識

### 2.1 関連箇所 (実測で確認したもの)

| 場所 | 何が既にあるか | 02 での使い方 |
|---|---|---|
| `diagnostics/base.py:34-49` | `DiagnosticContext(washout, dt, seed, companion_states)` | 第2軌道以降は `companion_states` で渡す (D-01)。**新フィールド `propagator` を追加**する |
| `diagnostics/base.py:83-97` | `Diagnostic` Protocol | 02 の新診断もこの署名 |
| `diagnostics/base.py:109-146` | `validate_diagnostic_input` が `companion_states` の形状一致まで検査済み | ESP 判定の入力検証をそのまま再利用 |
| `diagnostics/state_space.py:37-103` | 診断1本の実装形 (validate → resolve_context → scalars/arrays/params) | 02 の診断はこの形を踏襲 |
| `reservoir/esn.py:209-262` | `step(x, u, rng)` / `run(u, x0, rng)` が公開済み。`x0` 配線済み | 2軌道生成と摂動法の伝播器アダプタ |
| `config.py:103-121, 176-213` | `ExperimentConfig` (01 専用) + 再帰ローダ。未知キーは `ConfigError` (D-09) | ローダを型パラメータ化して 02 用設定クラスを載せる |
| `config.py:24` | `config` が `reservoir.esn.ESNConfig` を import | **D-12 の理由そのもの**。診断設定は `diagnostics/` 側に置き `config` が import する |
| `seeds.py:16-29, 41-75` | 3ストリーム (`RESERVOIR/TASK/SPLIT`) と `_STREAM_INDEX` | 初期状態対用に **4本目 `PROBE`** を足す |
| `experiment/state_space.py:113-160` | 「診断は純関数、実験層が行列を渡すだけ」の前例 | 02 の実験層もこの分業を踏襲 |
| `experiment/runner.py:161-191, 336-349` | `plan_replicate` / `run_experiment` (D-04/D-05/D-08 を内包) | **2-D はこれを丸ごと再利用**する (washout だけ差し替え) |
| `experiment/pipeline.py:110-150` | 5成果物を1関数で書き出す形 | 02 用に同形の `esp_pipeline` を作る |
| `main.py:29-45` | `EXPERIMENTS` の docstring に「02 着手時はこの構造ごと見直すこと」と明記済み | `(ローダ, パイプライン, YAML パス)` の組に変える |
| `tests/test_config_wiring.py:159-202, 324-338` | 全設定フィールドの配線を強制 | **02 の設定を別クラスにすればこのテストは無傷**。02 用に同型のテストを新設 |
| `tests/test_diagnostics_base.py:53-82, 99-176` | `pkgutil` で診断を機械列挙し実行時に D-01 契約を検査 | 02 の新診断は**自動的に**検査対象に入る |
| `tests/test_seeds.py:19-37` | D-06 guard が `for changed in SeedStream` を回す | **`SeedStream` に値を足すとここが落ちる**。同タスクで手当てが要る |

### 2.2 既存の慣習で守るもの (3個)

1. **診断は純関数、配線は実験層**: `diagnostics/` は行列を受け取って `DiagnosticResult` を返すだけ。どの行列を渡すかは `experiment/` が決める。
2. **CSV の列順は dataclass の宣言順が単一の真実** (`runner.CSV_COLUMNS`)。02 も `EspRow` / `WashoutRow` の宣言順を単一の真実にする。
3. **図のラベルは必ず `style.label(ja, en)` の対** (D-10)。片方だけ書くと `labels.label` が `ValueError`。

### 2.3 影響範囲

- 新規: `diagnostics/esp.py` / `diagnostics/timescale.py` / `experiment/esp.py` / `experiment/esp_pipeline.py` / `plotting/figures_esp.py` / `experiments/02_esp_and_dynamics/{config.yaml,run.py}`
- 変更: `diagnostics/base.py` (フィールド追加) / `config.py` (ローダ汎用化 + 02 設定クラス) / `seeds.py` (4本目) / `main.py` (ディスパッチ) / `tests/test_seeds.py` / `tests/test_config_wiring.py` / `docs/design.md` / `.claude/decisions.yaml` / `README.md`
- **変更しない**: `reservoir/esn.py` / `readout/` / `metrics.py` / `tasks/` / `experiment/{runner,split,report,summary,pipeline}.py` の公開シグネチャ

---

## 3. 前提・制約

### ハード制約 (絶対に変えない)

- **D-01**: 診断の署名は `f(X, u=None, y=None, *, ctx: DiagnosticContext | None = None) -> DiagnosticResult`。X/u/y/ctx 以外に**必須**引数を足さない。新診断は `test_all_diagnostics_conform_to_d01_signature_contract` に自動で乗る。
- **D-12**: `diagnostics/` は `rc_basics_lab.types` 以外の自作モジュールを import しない。診断の設定 dataclass は `diagnostics/` に定義し、`config.py` がそれを import する。伝播器も **Protocol として `diagnostics/base.py` に定義**し、ESN アダプタは `experiment/` 側に置く。
- **D-04 / D-05 / D-08**: 2-D は既存 `run_experiment` を再利用するため自動的に守られる。
- **D-09**: 02 の YAML も未知キーで即 `ConfigError`。
- **D-10**: 新規図の全ラベルに ja/en の対を用意する。新規依存を追加しない。
- 既存公開シグネチャの無断変更禁止: `load_config` / `make_rng` / `ESN.run` / `ESN.step` は**呼び出し互換を保つ** (追加は可、破壊は不可)。
- 依存は現状のまま (numpy / scipy / matplotlib / pyyaml)。
- ブランチは `feat/rc-basics-lab` を継続。

### ソフト制約

- 図・CSV のファイル名は要件書の指定に従う。ただし **2-D の行は列が異なるため `washout_sensitivity.csv` を追加**する (要件書の6成果物 +1)。
- 診断モジュール名は要件書の `diagnostics/esp.py` を採用する。
- 掃引格子の点数・系列長は実測 wall time を見て調整してよい (性能予算を超えないこと)。

---

## 4. タスク分解

### T1: 診断層 —— ESP 判定・条件付き Lyapunov・実効時定数 (想定所要: **L**)

**何をするか**

1. `diagnostics/base.py` に伝播器の Protocol と ctx フィールドを追加する。

```
class StatePropagator(Protocol):
    """時刻 t の状態 x から時刻 t+1 の状態を返す。x=X[t] を渡したら X[t+1] に一致すること。"""
    def __call__(self, x: FloatArray, t: int) -> FloatArray: ...
```

`DiagnosticContext` に `propagator: StatePropagator | None = None` を**既定値つきで**追加 (D-01 が許す唯一の拡張方法)。`reservoir` を一切 import しない構造的型付けなので D-12 に適合する。

2. `diagnostics/esp.py` を新設し、**設定 dataclass 2本と診断関数 2本**を置く。

   - `EspConfig(abs_tol=1.0e-6, rel_tol=1.0e-3, window=200, fit_skip=50, floor=1.0e-14)` (frozen)
   - `LyapunovConfig(method="perturbation", delta=1.0e-8, renorm_interval=1, max_growth=1.0e3, check_propagator=True, propagator_tol=1.0e-10)` (frozen)
   - `esp_convergence(X, u=None, y=None, *, ctx=None, cfg: EspConfig = DEFAULT_ESP) -> DiagnosticResult`
     - `ctx.companion_states` は**1本以上必須**。0本なら `ValueError`。
     - 距離は **RMS/ユニット距離** `d_i[t] = ||X[t] - C_i[t]|| / sqrt(N)` (N 非依存の無次元量)。
     - `arrays`: `distance` (最悪ペアの曲線)、`distance_all` (ペア × T)。
     - `scalars`: `d_initial` / `d_tail` / `converged` / `decay_rate_per_step` / `n_pairs` / `n_fit_points`。
     - 判定規則: `converged = (d_tail <= max(cfg.abs_tol, cfg.rel_tol * d_initial))`。全ペアの最悪値で判定。
     - `T < ctx.washout + cfg.window` なら `ValueError`。
   - `conditional_lyapunov(X, u=None, y=None, *, ctx=None, cfg: LyapunovConfig = DEFAULT_LYAPUNOV) -> DiagnosticResult`
     - `ctx.propagator` が `None` なら `ValueError`。
     - `cfg.method != "perturbation"` は `ValueError` (**解析 Jacobian 版の差し込み口**)。
     - `cfg.check_propagator` が真なら、数点で `propagator(X[t], t) ≈ X[t+1]` を検査し不一致なら `ValueError`。
     - 手順: `t = washout .. T-2` を `renorm_interval` ごとに区切り、区間頭で `x̃ = X[t] + delta * e`、区間末で `g = ||x̃ - X[t+end]|| / sqrt(N) / delta` を測り `log g` を累積、再正規化して次区間へ。`g > cfg.max_growth` なら `ValueError`。
     - `scalars`: `lyapunov_per_step` / `lyapunov_per_time` / `n_intervals` / `max_observed_growth`。

3. `diagnostics/timescale.py` を新設。

   - `TimescaleConfig(max_lag=200)` (frozen)
   - `autocorrelation_time(...)`: washout 後に各ユニットを中心化して自己相関を lag 0..max_lag で求め、ユニット平均 ACF の **1/e 交差点**を線形補間で求める。
   - `scalars`: `tau_1e` (交差が無ければ `nan`) / `tau_censored` / `tau_integrated` / `max_lag`。`arrays`: `acf`。

**受け入れ基準**

- [ ] `test_all_diagnostics_conform_to_d01_signature_contract` が新診断3本を含めて緑 (列挙件数 2 → 5 を明記して固定)。
- [ ] `test_diagnostics_package_does_not_transitively_import_reservoir` が緑のまま。
- [ ] **受け入れ条件6**: `test_works_on_externally_generated_states` —— ESN を一切 import せず外部生成系列で動く。
- [ ] `test_matches_analytic_exponent_for_linear_map`: `x -> A x` (ρ ∈ {0.5, 0.9, 1.1}) で `lyapunov_per_step` が `log ρ` と相対誤差 1e-6 以内。
- [ ] `test_matches_analytic_timescale_for_ar1`: AR(1) で `tau_1e` が `-1/log φ` と相対誤差 2% 以内。**φ を上げると単調増加**も固定。
- [ ] **D-16 guard**: `test_verdict_is_monotone_in_tolerance`。
- [ ] **D-18 guard**: `test_inconsistent_propagator_raises` / `test_unknown_lyapunov_method_raises` / `test_growth_beyond_max_growth_raises`。
- [ ] **有効性**: `test_esp_config_fields_change_output` —— 3設定クラスの**全フィールド**で出力が変わる。

**実装時に決めたこと (T1 実装者追記。仕様に書かれていなかった選択)**

決定は T5 で `.claude/decisions.yaml` / `docs/design.md` §9 に転記する。ここでは「何を決めたか + 理由」を残す。

*`diagnostics/esp.py` — `esp_convergence`*

1. **複数ペアの最悪値の取り方**: `d_tail = max_i median(d_i[末尾 window])`、`d_initial = min_i d_i[0]`。判定はこの2値と閾値の1本の不等式で行う。理由: ペアごとの判定の AND にすると、報告した `d_initial` / `d_tail` と `converged` が別の組み合わせから来ることになり、CSV の3列を見ても判定を再現できなくなる。この取り方は AND より厳しい側 (安全側) に倒れる。
2. **`distance` / `distance_all` は washout を切らず全 T を返す**。理由: 2-A の減衰図は過渡そのものを見せる図であり、切ると図が描けない。washout は判定 (末尾窓) と当てはめ開始位置にだけ効く。
3. **`decay_rate_per_step` の当てはめ範囲**: `t >= washout + fit_skip` かつ `d[t] > floor` の点で `log d` を最小二乗直線当てはめ。2点未満なら `nan` (`n_fit_points` を併記)。理由: 収束後に距離が丸めの床 (または厳密な 0) に張り付いた区間を含めると傾きが 0 側へ引かれる。「測れなかった」を 0 と区別するため `nan` を返す。
4. **`cfg` の値域検証は診断関数側で行う** (`abs_tol/rel_tol >= 0`, `window >= 1`, `fit_skip >= 0`, `floor > 0`)。理由: 「設定 dataclass は純データ、検証は使う側」という既存の慣習 (`ESNConfig` / `SplitConfig`) に合わせた。

*`diagnostics/esp.py` — `conditional_lyapunov`*

5. **摂動方向は乱数の単位ベクトル**。`ctx.seed` が `None` のときはモジュール定数 `_DIRECTION_SEED = 20240202` を使う。理由: 固定方向 (全成分同符号など) が Jacobian の主方向と直交すると指数を過小評価しうる。既定シードを固定するので `ctx.seed` を渡さなくても再現する。
6. **`delta` は RMS/ユニットで解釈**する (摂動ベクトルの L2 ノルムは `delta * sqrt(N)`)。理由: D-18 の「delta=1e-8 (RMS/ユニット)」を距離の定義 (D-16) と同じ土俵に載せるため。
7. **末尾の端数区間も使う** (長さが `renorm_interval` に満たない最後の区間を捨てない)。`lyapunov_per_step = Σ log g / Σ 区間長`。理由: 捨てると `renorm_interval` を変えたときに使うデータ量まで変わり、比較が濁る。
8. **`growth <= 0` は `ValueError`** (摂動が完全消失し `log` が取れない)。理由: `-inf` を CSV に流すと以降の集計が静かに壊れる。
9. **伝播器の整合検査は `[washout, T-2]` を5等分した時刻で行い、RMS/ユニット距離を `propagator_tol` と比較**する。理由: 全時刻を検査すると計算量が2倍になる。等間隔にするのは、入力インデックスのずれが特定区間だけで起きることはないため。
10. **`check_propagator=False` にしても `max_growth` が引っかかる場合がある** (1ステップずれた伝播器は初回区間で `growth ~ 1e8` になる)。テストでは伝播器検査単体の効きを見るため `max_growth` を上げて切り分けている。入力が弱い条件では成長率が `max_growth` に届かず、整合検査だけが唯一の防波堤になる。

*`diagnostics/timescale.py`*

11. **分散 0 のユニットは平均 ACF から除外**し、使ったユニット数を `params["n_units_used"]` に出す。全ユニットが定数なら `ValueError`。理由: 飽和して定数になったユニットで 0 除算する。黙って落とすと「全ユニットが死んでいるのに ACF が返る」が通るので本数を出力に残す。
12. **`tau_integrated` は初期正値列の和** (`acf` が最初に非正になる直前までの和)。理由: 素直に `max_lag` まで足すと、大ラグ側の推定誤差が積み上がって発散する。
13. **ラグの単位はステップ。`ctx.dt` では割らない**。理由: 2-B で重ねる理論線 `-1/log(1-a)` がステップ単位の量。
14. **`max_lag` は `scalars` に出し `params` には入れない**。理由: `DiagnosticResult.to_row` は params と scalars のキー衝突を `ValueError` にするため。
15. **`n_samples >= max_lag + 2` を要求**する (満たさなければ `ValueError`)。

*既存テストへの変更 (D-01 guard)*

16. `test_all_diagnostics_conform_to_d01_signature_contract` の「契約が許す全呼び出しパターンが呼べる」ブロックで **`ValueError` を許容**するようにした。`esp_convergence` / `conditional_lyapunov` は `ctx` に必須データが無いと `ValueError` を投げる仕様 (§4 T1) であり、これは署名契約ではなく入力要件だから。**代わりに、必須データをそろえた `ctx` での呼び出しが `DiagnosticResult` を返すことを必須の assert として追加**した (`ValueError` を投げ続けるだけの診断が契約テストを通り抜ける穴を塞ぐため)。
17. `test_diagnostic_enumeration_finds_all_known_diagnostics` を **集合の完全一致 + 件数 5 の固定**に変えた (従来は2本の包含のみ)。
18. `diagnostics/base.py` のモジュール docstring に、診断固有パラメータの渡し方として **`cfg` (D-15) を第1の形、frozen dataclass の `__call__` (D-01 の F-1-006 追記分) を第2の形**として併記した。docstring が「ctx に足すな、callable にしろ」だけのままだと D-15 と矛盾して読めるため。

*仕様に無いテストの追加*

19. `test_all_config_fields_have_a_case` (`tests/test_diagnostics_esp.py`) —— 3設定クラスのフィールド追加時に `test_esp_config_fields_change_output` のケース登録を強制する。理由: `test_all_config_fields_are_covered` (01) と同じ役割。これが無いと「全フィールドで出力が変わる」を名乗ったまま網羅性が静かに落ちる。

*実測値 (T1 完了時)*

- `lyapunov_per_step` vs `log ρ` (相対誤差): ρ=0.5 → 5.0e-10 / ρ=0.9 → 8.8e-12 / ρ=1.1 → 2.2e-09
- `tau_1e` vs `-1/log φ` (相対誤差): φ=0.8 → 0.35% / φ=0.9 → 0.46% / φ=0.95 → 0.75%
- `tau_1e` は φ=0.5 以下だと 1/e 交差が lag 1〜2 に来て**線形補間の誤差**が 6% に達する。テストは φ ∈ {0.8, 0.9, 0.95} を対象にした (実験 2-B のリーク率の範囲はこちらに入る)。

### T2: 設定層・乱数層 —— 02 用設定クラスと4本目のストリーム (想定所要: **L**)

**何をするか**

1. `config.py` のローダを型パラメータ化 (**追加のみ**)。`load_config_as[T](path, cls) -> T` を新設し、`load_config` はそれに委譲。
2. `config.py` に 02 用の設定クラス群 (**`ExperimentConfig` には1フィールドも足さない**。理由は D-13)。
   `EspSeedConfig` / `DriveConfig` / `EspDecayConfig` / `TimescaleSweepConfig` / `EspMapConfig` / `WashoutSweepConfig` / `Esp02Config`。
   `EspConfig` / `LyapunovConfig` / `TimescaleConfig` は **`diagnostics/` から import** (D-12 の許可された向き)。
3. `seeds.py` に4本目のストリーム `SeedStream.PROBE`。`make_rng_for(base_seed, stream, replicate)` を新設し `make_rng` は委譲。
   `test_streams_are_independent` の列挙を `SeedConfig` のフィールドに変え、`PROBE` は新テストで担保。
4. **配線テストの共有化**: `WiringCase` / `apply_case` / `_leaf_paths` を `tests/wiring.py` に切り出し、`tests/test_config_wiring_esp.py` を新設。

**受け入れ基準**

- [ ] `tests/test_config_wiring.py` の全ケースが**無改変の意味論で**緑。
- [ ] **D-13 guard**: `test_all_esp_config_fields_are_covered` —— `washout.base.*` を除く全葉が被覆され、除外集合が `_leaf_paths(ExperimentConfig)` と**完全一致**。
- [ ] **有効性**: `test_every_esp_parameter_changes_output`。
- [ ] `test_every_esp_field_round_trips_yaml` / `test_unknown_key_raises_for_esp_config`。
- [ ] **D-14 guard**: `test_probe_stream_is_independent_of_seed_config_streams`。
- [ ] `load_config(path)` の既存呼び出しが無改変で通る。

### T3: 実験 2-A / 2-B / 2-C と図3枚・CSV・CLI (想定所要: **L**)

**何をするか**

1. `experiment/esp.py` (配線層。`reservoir` と `diagnostics` の両方を import してよい唯一の場所)
   - `make_drive(sigma, n_steps, rng)`: 一様 i.i.d.、**標準偏差 σ を指定量**とし振幅 `a = sqrt(3) σ`。`sigma == 0.0` は厳密なゼロ系列。
   - `make_initial_states(n_units, n_pairs, rng)`: `U[-1,1]^N` を `n_pairs + 1` 本。**片方をゼロ状態にしない**。
   - `esn_propagator(esn, u)`: `lambda x, t: esn.step(x, u[t + 1])`。
   - `evaluate_condition(...) -> EspRow`。
   - `EspRow` (宣言順が CSV 列順): `experiment`, `replicate`, `seed_*`, `rho`, `leak_rate`, `input_scale`, `sigma_u`, `input_amplitude`, `input_drive_std`, `n_units`, `density`, `n_steps`, `washout`, `window`, `n_pairs`, `d_initial`, `d_tail`, `converged`, `decay_rate_per_step`, `lyapunov_per_step`, `lyapunov_per_time`, `tau_1e`, `tau_censored`, `tau_integrated`, `wall_time_s`。
2. `plotting/figures_esp.py` (D-10 準拠)
   - `plot_esp_decay` / `plot_leak_timescale` (理論線 `-1/log(1-a)` を重ねる) / `plot_esp_map` (ρ×σ、`converged` 率、λ=0 等高線、σ=0 は「no input」として別枠)。副題に **Gallicchio (2019) の再実演**を明記。
3. `experiment/esp_pipeline.py::run_and_report_esp`。`meta.json` に `esp_defaults` と `verdict_lyapunov_agreement` を載せる。
4. `main.py` の `EXPERIMENTS` を `dict[str, ExperimentSpec]` に変える。

**受け入れ基準**

- [ ] **受け入れ条件1**: `test_no_input_decay_matches_spectral_radius` —— σ=0 で ρ ∈ {0.5,0.8,0.95} は収束、{1.2,1.5} は非収束。`decay_rate_per_step` が `log ρ` と 20% 以内。
- [ ] **受け入れ条件2**: `test_strong_input_restores_esp_above_unit_spectral_radius` —— ρ=1.5・σ≥1.0 で収束する条件が存在し、同 ρ・σ=0 は非収束。**図ではなくデータで固定**。
- [ ] **受け入れ条件3**: `test_lyapunov_sign_agrees_with_verdict_away_from_boundary` (`|λ| > 0.01` の全条件)。境界近傍の件数は `meta.json` に記録。
- [ ] **受け入れ条件4**: `test_timescale_is_monotone_in_leak_rate`。
- [ ] **D-17 guard**: `test_input_strength_is_standard_deviation_not_amplitude`。
- [ ] `test_artifacts_are_regenerated_in_one_command` (PNG dpi を `conftest.png_dpi` で実測)。
- [ ] 性能: 本番 `wall_time_s < 900`、02 関連テスト合計 < 60 秒。

### T4: 実験 2-D —— washout 感度 (想定所要: **M**)

**何をするか**

- `run_washout_sweep(config)`: 実体は **`dataclasses.replace` で `washout` を差し替えて既存 `run_experiment` を呼ぶループ**。公平性 (D-04/05/08) は既存経路が担保。
- **交絡の除去 (D-19)**: washout を増やすと `t0` が上がり `n_usable` が縮むため、`pad_series=True` のとき `length = base_length + (max(grid) - washout)` として**行数を格子全体で一定に保つ**。`pad_series=False` は交絡ありの設計を再現するモード。
- 対象は MG と遅延パリティの両方 (図の主役は MG、パリティは「washout に反応しない対照」)。
- `WashoutRow`: `task`, `method`, `washout`, `replicate`, `alpha`, `n_lags`, `nrmse`, `nrmse_std`, `n_train`, `n_val`, `n_test`, `t0`, `pad_series`, `wall_time_s`。
- `plot_washout_sensitivity`: 01 の本番値 (washout=200) に垂直線を引き、変動幅を数値注記。

**受け入れ基準**

- [ ] **受け入れ条件5**: `test_washout_sweep_quantifies_performance_variation` —— NRMSE の (最大/最小) 比が `meta.json` に記録され 1.0 でない。
- [ ] **D-19 guard**: `test_washout_sweep_holds_training_size_constant` —— `pad_series=True` で行数一致、`False` で `n_train` 単調減少。
- [ ] `test_washout_zero_is_worst_or_equal_for_mackey_glass` (**破れたら記事の主張が変わるので止まって相談**)。
- [ ] **受け入れ条件7**: `test_all_four_figures_and_two_csv_in_one_command` + `make ci` 緑。

### T5: 記録 —— design.md / decisions.yaml / README と閾値感度 (想定所要: **M**)

- `docs/design.md` §9: 距離定義・閾値・窓の根拠、**閾値感度の実測表** (`abs_tol` 3 × `window` 3 の9通りで ESP 成立境界がどれだけ動くか)、δ と再正規化間隔の根拠、入力強度の定義、2-D のタスク選定と `pad_series` の理由、λ と判定の不一致件数。
- `.claude/decisions.yaml` に D-13〜D-19 を追記。
- `README.md` に 02 の実行方法・成果物・実測値の要約 (数値は生成物と突き合わせる形。手書きを増やすなら突き合わせテストも足す)。
- 要件書の未確定事項1・2・3・5 に「決定済み」を追記 (4 は次回サーベイ課題)。

**受け入れ基準**

- [ ] `check_decisions.py` が D-13〜D-19 について緑。
- [ ] `tests/test_design.py` が §9 の既定値と**コード上の既定値の一致**を検証 (散文とコードの乖離を機械で殺す)。
- [ ] `results/esp_threshold_sensitivity.csv` が9行以上を持ち design.md の表と行数一致。

---

## 5. 評価軸 (Check フェーズに渡す)

### 機能観点
- ρ<1 無入力で指数減衰・ρ>1 で非減衰 (減衰率が `log ρ` と 20% 以内)。
- 強入力で ρ>1 でも ESP。**図の目視ではなくデータで固定**。
- λ の符号と ESP 判定の整合 + `meta.json` の不一致件数。
- リーク率と時定数の単調性 = AR(1) 解析解 (T1) と実 ESN (T3) の二段構え。

### 性能観点
- 本番 `wall_time_s < 900 秒`。内訳も `meta.json` に記録。
- pytest 全体の増分 < 90 秒。超えたら格子ではなく `n_steps` / `n_units` を削る。

### 安全性観点 (壊れたら困るもの)
- `tests/test_config_wiring.py` の 01 側ケースが**意味論的に無改変**。`WIRING_CASES` の件数が減っていたら差し戻し。
- D-01 / D-12 / D-06 の guard が緑のまま。特に D-12 guard は伝播器 Protocol 導入で最も壊れやすい。
- `load_config` / `make_rng` / `ESN.run` / `ESN.step` の既存呼び出しが無改変で通る。
- **01 の成果物 (`comparison.csv` の指紋) が 02 の変更で1バイトも変わらない**。

### 有効性観点
- `Esp02Config` の全葉で `test_every_esp_parameter_changes_output` が強制。
- 診断側の設定は `ExperimentConfig` の外にあるため経路が違う。`test_esp_config_fields_change_output` (T1) が**診断単体のレベルで**同じことを強制する。この二重化が無いと「YAML から診断設定が届いていない」が黙って通る。
- `lyapunov.method` / `drive.distribution` は値域が1点なので `CHANNEL_ERROR` として検査。

---

## 6. 意図的な決定 (D-13〜D-19)

```yaml
- id: D-13
  rule: "実験ごとに設定 dataclass を分ける。02 の設定は Esp02Config に置き、ExperimentConfig (01 専用) には1フィールドも足さない。ローダは load_config_as(path, cls) で共有する"
  rationale: "test_each_parameter_changes_output は『全フィールドが 01 のパイプライン出力を変える』ことを要求する。02 用フィールドを相乗りさせると満たせないフィールドが必ず生まれ、逃がすために例外チャネルを増やすと配線漏れの検出力そのものが落ちる (このテストは本リポジトリ最大の失敗モード『設定したのに効いていない』への唯一の防衛線)。設定を分ければ 01 の被覆は無傷のまま 02 に同型の被覆を作れる"
  guard_test: "tests/test_config_wiring_esp.py::test_all_esp_config_fields_are_covered"

- id: D-14
  rule: "初期状態対の生成に4本目の乱数ストリーム SeedStream.PROBE を使う。01 の SeedConfig には probe を足さず、make_rng_for 経由で 02 の EspSeedConfig から渡す"
  rationale: "ESP 判定の頑健性は『同じ重み・同じ入力で初期状態対だけ振ったときに判定が変わらないか』で測るため、初期状態は reservoir / drive と独立に振れる必要がある。reservoir ストリームから引くと初期状態を変えた瞬間に重みも変わり、この検査が原理的に不可能になる。一方 01 の SeedConfig に probe を足すと D-13 と同じ理由で配線テストが破れる"
  guard_test: "tests/test_seeds.py::test_probe_stream_is_independent_of_seed_config_streams"

- id: D-15
  rule: "診断の設定値 (閾値・窓・摂動サイズ) は DiagnosticContext ではなく、既定値つきキーワード引数 cfg で渡す。DiagnosticContext には実行時データだけを入れる"
  rationale: "D-01 の拡張規約は『ctx への既定値つきフィールド追加』だが、02〜05 の全診断の設定を ctx に積むと、移植先が診断1本を使うために全診断の設定を知る羽目になる。cfg は既定値つきキーワード引数なので D-01 の guard は完全に維持される。境界は『系そのものを表すか (ctx) / 判定基準を表すか (cfg)』"
  guard_test: "tests/test_diagnostics_esp.py::test_esp_config_is_passed_as_defaulted_keyword"

- id: D-16
  rule: "ESP 判定の距離は RMS/ユニット距離 ||x_a - x_b|| / sqrt(N)。判定は『末尾 window ステップの中央値 <= max(abs_tol, rel_tol * d_initial)』。既定 abs_tol=1e-6, rel_tol=1e-3, window=200。複数ペアは最悪値で判定。初期状態は両方とも U[-1,1]^N から独立に引き、片方をゼロ状態にしない"
  rationale: "(1) sqrt(N) 正規化により閾値が N に依存せず、N の異なるメモリスタ配列へそのまま移植できる。(2) abs_tol=1e-6 は float64 の eps から十分離れ、初期距離 O(1) から6桁の収縮を要求するので描画の丸めに紛れない。(3) window=200 は 01 の washout と同じ桁。(4) 片方をゼロ状態にすると、無入力では 0 が不動点であるため『2軌道の分離』ではなく『単一軌道の原点への収束』を測ることになり ESP と別の量に化ける。既定値は結論を変えうるので design.md §9 に感度表を残す"
  guard_test: "tests/test_diagnostics_esp.py::test_verdict_is_monotone_in_tolerance"

- id: D-17
  rule: "入力強度は駆動信号の標準偏差 σ_u で定義する (振幅ではない)。一様分布では振幅 a = sqrt(3) σ_u。掃引中 input_scale は固定し、変えるのは信号側だけ。CSV には sigma_u / input_amplitude / input_drive_std の3列を出す"
  rationale: "要件_02 設計判断4。2-C の横軸が『振幅か分散か』で曖昧になると、記事の主張 (Scholarpedia の再実演) と数値の対応が取れない。標準偏差を主軸に取るのは分布形が変わっても比較可能で、Manjunath & Jaeger 2013 の定式化と同じ土俵に乗るため。振幅も併記するのは Scholarpedia の文言が振幅で書かれているため。input_scale を同時に動かすと『信号を強くした』のか『重みを大きくした』のか分離できなくなる"
  guard_test: "tests/test_experiment_esp.py::test_input_strength_is_standard_deviation_not_amplitude"

- id: D-18
  rule: "条件付き Lyapunov 指数は摂動法で推定する。delta=1e-8 (RMS/ユニット)、再正規化間隔 1 ステップ、method='perturbation' 以外は ValueError。伝播器は ctx.propagator で受け取り、propagator(X[t], t) が X[t+1] に一致することを実行時に検査する"
  rationale: "要件_02 未確定2・設計判断2。delta=1e-8 は sqrt(eps)=1.5e-8 近傍で、線形性からのずれと丸め誤差が釣り合う古典的な最適点。間隔1は |λ|<0.5/step の範囲で分離が線形域を外れない最も安全な選択で、推定値が解析解と直接照合できる。method を文字列にしてあるのは解析 Jacobian 版の差し込み口であり、未実装の値を黙って受理しない。propagator の整合検査は『参照軌道と別の入力で伝播している』配線ミスを実行時に落とすためで、この種のミスは λ が"それらしく"出るためレビューでは見つからない"
  guard_test: "tests/test_diagnostics_esp.py::test_matches_analytic_exponent_for_linear_map"

- id: D-19
  rule: "washout 感度実験では、washout を増やしても訓練/検証/テストの行数が変わらないよう系列長を伸ばして補償する (pad_series=True が既定)。補償なしは pad_series=False として残し対比に使う"
  rationale: "make_split は n_usable = T - max_start_offset - t0 で行数を決めるため、washout を増やすと訓練データ量が同時に減る。素直に掃引すると『washout の効果』と『訓練データ量の効果』が交絡し、受け入れ条件5 が別の量の測定になる。この交絡は滑らかな単調曲線として出るため図を見ても気づけない"
  guard_test: "tests/test_experiment_washout.py::test_washout_sweep_holds_training_size_constant"
```

---

## 7. 想定リスク (起きたら止まって相談)

1. **2-C で「強入力なら ρ>1 でも ESP」が再現できない**。確認順序: (a) `input_scale` が弱すぎないか (2-C では 1.0 固定を推奨)、(b) 判定窓が短すぎないか、(c) `n_steps` が過渡を含んでいないか。それでも出なければ格子を無制限に広げる前に相談。
2. **λ の符号と ESP 判定が広範囲で食い違う**。境界近傍以外で不一致が 5% を超えたら実装誤りの可能性が高い (最有力は伝播器の入力インデックスずれ)。
3. **実行時間が予算 (15分) を超える**。削り方 (格子を粗く vs 系列を短く vs レプリケート減) は結論の頑健性に直結するので、数値を測ってから相談。

---

## 8. 不明点 (3問) —— **すべてユーザー承認済み**

**Q1. 02 の設定クラスを分けるか (D-13)** → **(A) `Esp02Config` 新設**で決定。
`ExperimentConfig` には1フィールドも足さない。ローダは `load_config_as(path, cls)` で共有する。

**Q2. 3つの L タスクを2サイクルに割るか** → **(A) 2サイクルに割る**で決定。
- **サイクル 2a = T1 + T2** (診断層 + 設定/乱数層)。図は作らず `make ci` 緑で締める。
  成果物はサイクル3の IPC がそのまま乗る土台なので単独で価値が確定する
- **サイクル 2b = T3 + T4 + T5** (実験・図・記録)

**Q3. 2-C の格子規模** → **(A) ρ16 × σ7 × 3rep、T=3000、N=200** で決定。
N=200 はサイクル1と同じで連載を通した比較可能性が保たれる。予算 15 分。

**前提として置くもの**: 駆動入力は i.i.d. 一様乱数 (MG は使わない) / 2-D は MG と遅延パリティ両方 / 図の言語は D-10 のまま / Gallicchio の ESP index との対応精査は**実装スコープ外** (記事執筆時のサーベイ課題)。

**前提として置くもの**: 駆動入力は i.i.d. 一様乱数 (MG は使わない) / 2-D は MG と遅延パリティ両方 / 図の言語は D-10 のまま / Gallicchio の ESP index との対応精査は**実装スコープ外** (記事執筆時のサーベイ課題)。

---

## 9. 受け入れ条件 → タスク対応表

| # | 要件書の受け入れ条件 | タスク | 検証手段 |
|---|---|---|---|
| 1 | ρ<1 で指数減衰・無入力 ρ>1 で非減衰 | T3 (診断は T1) | `test_no_input_decay_matches_spectral_radius` + `fig_esp_decay.png` |
| 2 | 強入力で ρ>1 でも ESP | T3 | `test_strong_input_restores_esp_above_unit_spectral_radius` + `fig_esp_map.png` |
| 3 | λ の符号が判定と整合 | T1 + T3 | `test_matches_analytic_exponent_for_linear_map` / `test_lyapunov_sign_agrees_with_verdict_away_from_boundary` |
| 4 | リーク率と自己相関減衰が単調 | T1 + T3 | `test_matches_analytic_timescale_for_ar1` / `test_timescale_is_monotone_in_leak_rate` |
| 5 | washout 長の性能変動が定量化 | T4 | `test_washout_sweep_quantifies_performance_variation` |
| 6 | ESP 判定が外部生成系列でも動く | T1 | `test_works_on_externally_generated_states` (+ D-12 guard) |
| 7 | 図4枚が1コマンド再生成 + pytest green | T3 + T4 | `test_all_four_figures_and_two_csv_in_one_command` + `make ci` |

---

## 実装者への注意 (最も壊れやすい3点)

1. **`SeedStream` に値を足した瞬間 `test_streams_are_independent` (D-06 guard) が落ちます。** T2 で列挙方法の手当てを同時に行うこと (guard を消すのではなく対象を明示し、PROBE 用の検査を別テストで足す)。
2. **`DiagnosticContext` に `propagator` を足すとき、Protocol は必ず `diagnostics/base.py` に置くこと。** `experiment/` 側に置いて import すると D-12 guard が別プロセス検査で落ちます。
3. **`X[t]` は `u[t]` を処理した後の状態です。** 伝播器は `propagator(x, t) = esn.step(x, u[t+1])` であり、`u[t]` を渡すと 1 ステップずれた λ が"それらしい値"で出ます。`check_propagator` を既定で有効にしたまま実装すること。
