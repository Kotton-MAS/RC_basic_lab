# 仕様: rc-basics-02 —— ESN を動かして測る (ESP・スペクトル半径・リーク率)

*サイクル2 / 対象要件: `docs/series/要件_rc-basics-02.md` / 前提: サイクル1 (D-01〜D-12, make ci 緑)*

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

*サイクル2a round 2 での修正 (F-02-2-002)*

20. `_iter_diagnostic_callables()` (`tests/test_diagnostics_base.py`) の列挙述語を **`inspect.isfunction` 限定から「関数、または `__call__` の戻り値アノテーションが `DiagnosticResult` である public callable インスタンス (クラス自体は除外)」へ拡張**した。理由: 上記18で docstring に書いた第2形 (frozen dataclass の `__call__`) が、`isfunction` 限定の列挙では構造的に拾えず、第2形の診断が `diagnostics/` に現れた瞬間に `MINIMAL_VALID_INPUT` 登録の強制・D-15 guard・D-01 契約テストの3つが同時に静かに無効化される穴があった (reviewer-architecture / オーケストレータが実測)。**03 の IPC (サロゲート本数・最大遅延/次数を持つ) はこのルールが名指しで推奨する対象**であり、03 着手前に塞ぐ必要があった。安全な変異注入 (`diagnostics/_tmp_second_form.py` を一時追加して検証後に即削除) で3guardすべての有効性を実測済み。既存5診断 (すべて第1形) は列挙件数・内容とも変わらない。詳細は `docs/review-findings-02.md` の F-02-2-002。

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

**実装時に決めたこと (T2 実装者追記。仕様に書かれていなかった選択)**

決定は T5 で `.claude/decisions.yaml` / `docs/design.md` §9 に転記する。

*`test_every_esp_parameter_changes_output` の意味 (最重要。T3 で必ず読むこと)*

1. **サイクル 2a には消費側 (実験層 `experiment/esp.py`) が無いため、格子・系列長・
   ユニット数のような葉は「出力が変わる」を実測できない**。ここで「出力」を
   テスト側で作った消費関数で捏造すると、テストがテスト自身を検査することになり
   検出力は 0 になる。そこで `tests/test_config_wiring_esp.py` は葉を3チャネルに
   割り当て、**割り当てに漏れが無いこと**を機械的に固定する設計にした。
   - `CHANNEL_SEEDS` (`seeds.*` 3件): `esp_stream_seed` + `make_rng_for` は T2 の
     実装なので、**実際に乱数列が変わり他ストリームが1バイトも動かないこと**を実測する。
   - `CHANNEL_DIAGNOSTIC` (`esp.*` / `lyapunov.*` / `timescale.*` 12件): 効きの実測は
     T1 の `test_esp_config_fields_change_output` に委譲する。委譲が閉じていることは
     `test_diagnostic_sections_cover_the_diagnostic_config_classes`
     (セクションの全葉 = 設定クラスの全フィールド) と T1 の
     `test_all_config_fields_have_a_case` (全フィールドにケースがある) の対で担保する。
   - `CHANNEL_PENDING` (残り 28件): **T3 / T4 で本物のチャネルへ書き換える**。
     2a では「YAML を往復して値がその葉にだけ届く」(= T2 が実装したローダそのもの)
     だけを実測する。
   - 全チャネル共通で `_changed_leaves(base, changed) == {case.field}` を assert し、
     差し替えが他の葉へ波及しないことを固定している。
2. **先送りを時限装置にした**: `test_pending_cases_disappear_once_the_experiment_layer_exists`
   は `importlib.util.find_spec("rc_basics_lab.experiment.esp")` が真になった瞬間に
   `CHANNEL_PENDING` が残っていると失敗する。**T3 の実装者は、実験層を作った時点で
   このテストが赤くなる**。pending ケースを本物の出力チャネル (01 の `CHANNEL_ROWS` /
   `CHANNEL_ERROR` に相当) へ書き換えるまで緑にならない。
   併せて `PENDING_SECTIONS` により、2a で実測できるセクション (`seeds` / `esp` /
   `lyapunov` / `timescale`) を pending へ逃がすことを禁じている。
   `drive.distribution` と `lyapunov.method` は §5 の指示どおり値域1点だが、
   前者は消費側 (`make_drive`) が無いので現状 pending、後者は T1 の診断が既に
   `ValueError` にするので `CHANNEL_DIAGNOSTIC` に置いた。

*`config.py`*

3. **`EspSeedConfig` は `reservoir` / `drive` / `probe` の3本**とし `split` を持たない。
   2-A/2-B/2-C は分割しないため。`esp_stream_seed(seeds, SeedStream.SPLIT)` は
   `ValueError` (2-D の分割は `washout.base.seeds.split` を使う)。理由: 使われない
   フィールドを設定に残すと「設定しても効かない葉」が生まれ、D-13 が避けたい状況を
   自分で作ることになる。
4. **フィールド名とストリーム名が1対1でない** (`drive` ↔ `SeedStream.TASK`) ため、
   対応は `esp_stream_seed` の明示的な `match` で書き、`getattr` を使わない。
   理由: `seeds._base_seed` と同じ流儀。「他ストリームのシードを参照していない」が
   コードの形から読める。
5. **各実験セクション (`decay` / `timescale_sweep` / `esp_map`) は `ESNConfig` を
   内包せず、そのセクション固有の掃引軸だけを平らに持つ** (`decay.rho_grid` /
   `decay.leak_rate` / `timescale_sweep.leak_rate_grid` / `esp_map.sigma_grid`
   など)。`input_scale` / `n_units` / `density` / `n_replicates` はセクションに
   重複させず、`Esp02Config.reservoir` (`ReservoirSweepConfig`) に1本だけ
   集約する (round 2 実装で確定。F-02-1-004)。理由 (2つ):
   - `ESNConfig` を内包すると掃引軸である `spectral_radius` が「YAML から
     設定できるが実行時に上書きされる死んだフィールド」になり、D-13 が
     防ごうとしている状態をそのまま作る。
   - 3セクションが `n_units` 等を別々に持つと、値が食い違っても
     `ConfigError` にならず黙って条件が割れる。1本に集約することで、
     §8 Q3 でユーザーが承認した「N=200 をサイクル1との連続性のため
     連載を通して固定する」をコードで構造的に保証する。
6. **`drive` セクションに `distribution` / `n_steps` / `washout` / `n_pairs` を集約**し、
   実験セクションごとに重複させない。理由: 同じ量が3か所にあると、片方だけ直した
   ときに黙って条件が食い違う。
7. **値域検証は行わない** (`ConfigError` は型不一致と未知キーのみ)。理由: 「設定
   dataclass は純データ、検証は使う側」という既存の慣習 (`ESNConfig` / `SplitConfig` /
   T1 の決定4)。`drive.distribution` の値域検査は T3 の `make_drive` が持つ。
8. **既定値**: `esp_map.rho_grid` = `linspace(0.4, 1.9, 16)` (小数第3位で丸め)、
   `esp_map.sigma_grid` = `(0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)` の7点 (§8 Q3 の 16×7)。
   `decay.rho_grid` = `(0.5, 0.8, 0.95, 1.2, 1.5)` (受け入れ条件1 の値そのもの)。
   `timescale_sweep.leak_rate_grid` = `(0.1, 0.2, 0.3, 0.5, 0.7, 1.0)`。
   `washout.grid` = `(0, 50, 100, 200, 400, 800)` (01 の本番値 200 を含む)。
   `n_steps=3000` / `n_units=200` は §8 Q3。**T3 で実測 wall time を見て調整してよい**
   (ソフト制約)。
9. `Esp02Config.name` の既定値は `"02_esp_and_dynamics"` (T3 で作る実験ディレクトリ名)。

*`seeds.py`*

10. **`_STREAM_INDEX[PROBE] = 3` (末尾に追加)**。既存3本の index を動かすと 01 の
    成果物がバイト単位で再現しなくなる。`make_rng` は `make_rng_for` への委譲に
    したが、`SeedSequence(entropy, spawn_key)` の作り方は一切変えていない
    (`test_make_rng_for_matches_make_rng` で固定)。
11. **`make_rng(config, SeedStream.PROBE, ...)` は `ValueError`**。理由: `SeedConfig` に
    `probe` が無いことを黙って既定値で埋めると D-14 が形骸化する。`_base_seed` の
    `match` は 4 ケース全部を書き切っており、ストリームを足すと mypy が
    「Missing return」で落ちる (列挙の網羅性が型で守られる)。

*テストの共有化 (`tests/wiring.py`)*

12. 切り出したのは `WiringCase` / `case` / `apply_case` / `leaf_paths` / `plain` /
    `assert_yaml_has_all_leaves` とチャネル定数。`_leaf_paths` → `leaf_paths`、
    `_plain` → `plain` に改名 (モジュール外から呼ぶため)。`case()` に `note` 引数を
    足したが既定値 `""` なので **01 のケース定義は1文字も変わっていない**
    (`git diff` 上、ケース定義行の差分は 0 行 / `WIRING_CASES` は 40 件のまま)。
    **判定そのものは共有していない**。「出力」が何かは実験ごとに違う (01 は結果行、
    02 は乱数列や設定オブジェクト) ため、判定を共有すると弱い方に引きずられる。

*テストの追加 (仕様に無いもの)*

13. `test_seed_config_fields_map_one_to_one_onto_streams` —— D-06 guard の列挙を
    `SeedStream` 全体から `SeedConfig` のフィールドへ変えたことで、対応が崩れると
    検査対象が静かに縮む経路ができた。それを塞ぐ。
14. `test_make_rng_for_matches_make_rng` —— 01 の乱数列が委譲によって変わっていない
    ことをバイトで固定する。
15. `test_esp_config_does_not_leak_into_experiment_config` —— `ExperimentConfig` の
    フィールド集合を凍結する (D-13 の「1フィールドも足さない」を直接検査)。
16. `test_streams_differ_from_each_other_under_identical_seeds` は `make_rng_for` を
    使う形に変え、対象を **4ストリーム全部**に広げた (従来は `SeedConfig` の3本)。

*コード規約 (実装中に判明したもの)*

17. ruff の `RUF001` / `RUF002` により、**ソース中の文字列・docstring にギリシャ文字
    `ρ` `σ` と `×` を書けない** (ASCII と紛らわしい文字として弾かれる)。`rho` /
    `sigma` / `x` と書く。`λ` は対応する ASCII が無いため許容される (T1 の
    テストが実際に使っている)。

### T3: 実験 2-A / 2-B / 2-C と図3枚・CSV・CLI (想定所要: **L**)

**何をするか**

1. `experiment/esp.py` (配線層。`reservoir` と `diagnostics` の両方を import してよい唯一の場所)
   - `make_drive(sigma, n_steps, rng)`: 一様 i.i.d.、**標準偏差 σ を指定量**とし振幅 `a = sqrt(3) σ`。`sigma == 0.0` は厳密なゼロ系列。
   - `make_initial_states(n_units, n_pairs, rng)`: `U[-1,1]^N` を `n_pairs + 1` 本。**片方をゼロ状態にしない**。
   - `esn_propagator(esn, u)`: `lambda x, t: esn.step(x, u[t + 1])`。
   - `evaluate_condition(...) -> EspRow`。
   - `EspRow` (宣言順が CSV 列順): `experiment`, `replicate`, `seed_*`, `rho`, `leak_rate`, `input_scale`, `sigma_u`, `input_amplitude`, `input_drive_std`, `n_units`, `density`, `n_steps`, `washout`, `window`, `n_pairs`, `d_initial`, `d_tail`, `converged`, `decay_rate_per_step`, `lyapunov_per_step`, `lyapunov_per_time`, `tau_1e`, `tau_censored`, `tau_integrated`, `wall_time_s`。
     このうち `input_scale` / `n_units` / `density` は `Esp02Config.reservoir`
     (`ReservoirSweepConfig`) 由来であり、セクション固有の YAML キーではない
     (実装メモ5)。
2. `plotting/figures_esp.py` (D-10 準拠)
   - `plot_esp_decay` / `plot_leak_timescale` (理論線 `-1/log(1-a)` を重ねる) / `plot_esp_map` (ρ×σ、`converged` 率、λ=0 等高線、σ=0 は「no input」として別枠)。副題に **Gallicchio (2019) の再実演**を明記。
3. `experiment/esp_pipeline.py::run_and_report_esp`。`meta.json` に `esp_defaults` と `verdict_lyapunov_agreement` を載せる。
4. `main.py` の `EXPERIMENTS` を `dict[str, ExperimentSpec]` に変える。

**受け入れ基準**

- [ ] **受け入れ条件1**: `test_no_input_decay_matches_spectral_radius` —— σ=0 で ρ ∈ {0.5,0.8,0.95} は収束、{1.2,1.5} は非収束。`decay_rate_per_step` が `log ρ` と 20% 以内。
- [ ] **受け入れ条件2**: `test_strong_input_restores_esp_above_unit_spectral_radius` —— ρ=1.5・σ≥1.0 で収束する条件が存在し、同 ρ・σ=0 は非収束。**図ではなくデータで固定**。
- [ ] **受け入れ条件3**: `test_lyapunov_sign_agrees_with_verdict_away_from_boundary` (`|λ| > 0.01` の全条件)。境界近傍の件数は `meta.json` に記録。
      → **実装時に非対称な要求へ変更** (下記の決定1。ユーザー承認済み)。
- [ ] **受け入れ条件4**: `test_timescale_is_monotone_in_leak_rate`。
- [ ] **D-17 guard**: `test_input_strength_is_standard_deviation_not_amplitude`。
- [ ] `test_artifacts_are_regenerated_in_one_command` (PNG dpi を `conftest.png_dpi` で実測)。
- [ ] 性能: 本番 `wall_time_s < 900`、02 関連テスト合計 < 60 秒。

**実装時に決めたこと (T3 実装者追記。仕様に書かれていなかった選択)**

決定は T5 で `.claude/decisions.yaml` / `docs/design.md` §9 に転記する。
1〜4 は着手前にユーザー承認済み (実測を添えて確認した)。5 以降は実装中の判断。

*承認を取った4件 (受け入れ条件の成否に直結するもの)*

1. **λ の符号と ESP 判定の整合の要求を非対称にする → D-20 として登録予定**。
   「`λ>0` なのに収束 (**偽の ESP**) は全条件で 0 件」と「`σ_u >= 0.5` の全条件で
   `sign(λ)<0 ⇔ converged==1`」の2本を要求し、「`λ<0` なのに非収束」は許容して
   件数と内訳を `meta.json` に残す。理由: 条件付き Lyapunov 指数は参照軌道まわりの
   **局所**量なので**多安定性を原理的に検出できない**。tanh は奇関数なので `x*` が
   不動点なら `-x*` も不動点であり、どちらも局所安定 (λ<0) でありながら初期状態に
   よって行き先が割れる (= ESP 不成立)。ρ=1.1・σ=0 で4軌道を直接観測して確認した
   (末尾200ステップの時間標準偏差 4.7e-16 = 全軌道が不動点に到達、うち3本が同一点、
   1本が距離 0.526 の別の点)。§7 のリスク2 が疑えという「伝播器の入力インデックス
   ずれ」は先に否定済み (下記の実測値を参照)。**この非対称性を「揃っていないから」と
   対称化しないこと**。対称化すると実在の現象を実装バグとして潰すことになる。
2. **02 の ESN は `bias_scale = 0.0` で構成する** (`experiment/esp.py::BIAS_SCALE`)。
   `ReservoirSweepConfig` に `bias_scale` の葉は無く `ESNConfig` の既定 0.1 が効くが、
   定数バイアスは `[1; u]` の先頭成分に掛かる**振幅一定の入力そのもの**であり、
   `sigma_u = 0` を「無入力」と呼べなくなる。実測: `bias_scale=0.1` では無入力・
   ρ=1.2 でも2軌道が収束し受け入れ条件1 が成立しない。D-17 が入力強度を駆動信号の
   標準偏差で定義している以上、その定義に入らない常時入力は 0 にする。
3. **`DriveConfig.n_pairs` の既定を 3 → 10 に上げる**。無入力・ρ>1 の ESN は `+x*` /
   `-x*` の対をなす吸引子を持つことがあり、比較軌道が k 本すべて参照軌道と同じ側へ
   落ちる確率が約 `2^-k` 残る。実測: `n_pairs=3` では特定のリザバー draw
   (replicate 2) で ρ=1.05〜1.5・無入力が「収束」と誤判定され、受け入れ条件1 が
   全レプリケートでは成立しなかった。10 本にすると全レプリケートで正しく非収束に
   なり、ρ<1 側の判定は変わらない (n_pairs=30 まで上げても偽陰性なし)。
   本番 wall time は 43.2s → 83.3s (予算 900s の 9%)。
4. **`esp_convergence` に渡す ctx の washout を 0 固定にする**
   (`experiment/esp.py::ESP_DISTANCE_WASHOUT`)。λ と自己相関の ctx には
   `drive.washout` を渡す (両者は別の要求)。`esp_convergence` で washout が効くのは
   減衰率の当てはめ開始位置だけで、判定 (末尾 window の中央値) には効かない。一方
   2-A は**過渡そのものを見せる図**であり、距離が `floor` に届く前に当てはめを
   始めないと減衰率が測れない。実測: 無入力 ρ=0.5 の距離は t≈46 で 1e-14 を割る
   ため、当てはめ開始が `washout(200) + fit_skip(50) = 250` だと当てはめ点が 0 点に
   なり `decay_rate_per_step` が `nan` になる (ρ=0.8 も同様)。あわせて 02 の
   `Esp02Config.esp` の既定を `EspConfig(fit_skip=10)` にした
   (`config._esp_criteria_for_02`)。D-16 の診断側の既定 (`DEFAULT_ESP`, fit_skip=50)
   は変えていない。guard は `test_decay_fit_starts_before_the_distance_underflows`
   と `test_production_yaml_can_measure_the_decay_rate`。

*実装中の判断 (承認不要な範囲)*

5. **`evaluate_condition` の戻り値を `EspRow` ではなく `ConditionOutcome`
   (`row` + 距離曲線 + ACF) にした**。理由: 2-A の図は距離曲線そのものが主役で、
   行だけ返すと図のために全条件をもう一度回すことになる (実測 83 秒が倍になる)。
   `ConditionOutcome.row` が `EspRow` なので CSV 列順の単一の真実は変わらない。
6. **`meta.py::collect_meta_for(config, seeds)` と
   `experiment/report.py::write_meta_for(...)` を追加**し、既存の `collect_meta` /
   `write_meta` はそこへの委譲にした (署名は不変)。理由: 実験ごとに設定クラスは
   分かれる (D-13) が、`meta.json` の項目と書き出し規律は1か所に置きたい。
   `load_config` / `load_config_as` と `make_rng` / `make_rng_for` と同じ形。
7. **`main.EXPERIMENTS` の値は `(ローダ, パイプライン, YAML パス)` の組ではなく
   `ExperimentSpec(config_path, run)`** にした。理由: ローダの戻り値型が実験ごとに
   違う (`ExperimentConfig` / `Esp02Config`) ため、組のままでは `Any` を使わずに型を
   付けられない。「設定を読んでパイプラインへ渡す」までを1つの `run` に閉じると、
   実験ごとの型が関数の内側に収まりレジストリは単一の型で書ける。
8. **02 の CLI は `experiments/02_esp_and_dynamics/run_02.py`** (01 の `run.py` と
   名前を揃えていない)。理由: mypy がリポジトリ配下の同名トップレベルモジュールを
   `Duplicate module named "run"` として解析を止める。実験ディレクトリ名は数字始まりで
   パッケージにできず (`01_what_is_rc is not a valid Python package name`)、
   `[tool.mypy]` の設定は変更しない制約がある。03 以降も `run_<番号>.py` を使う。
9. **本番 YAML の `lyapunov.max_growth` は `1000.0` と書く**。`1.0e3` は YAML 1.1 では
   指数部に符号が無いため**文字列**として読まれ `ConfigError` になる (実測)。
   同種の誤りを CI で落とすため `test_esp_config_yaml_matches_the_real_experiment` を
   足し、本番 YAML を実際にローダへ通すようにした (01 の同名テストと同じ役割)。
10. **`Makefile` に `figures-02` を追加**し、出力先を `results/02_esp_and_dynamics/`
    にした。理由: 01 と同じ `results/` に出すと `meta.json` が衝突して 01 の成果物が
    上書きされる。`make ci` の構成は変えていない。
11. **2-A の図は横軸を「測れている区間 + 余白」で切る** (`_decay_x_limit`)。理由:
    系列長 3000 をそのまま横軸に取ると、測れている区間 (ρ=0.5 で 46 ステップ、
    ρ=0.95 でも 680 ステップ) が左端に潰れ、図の主役である傾きの違いが読めない。
    切ったことは軸ラベルに数値で書く (本番では「系列長 3000 のうち先頭 885 を表示」)。
12. **`plot_esp_map` の横軸は sigma_u の順位**にした (値そのものではない)。理由:
    格子が 0.05〜2.0 と等比的に広がるので、値を軸に取ると強入力側が潰れて読めない。
    λ=0 の等高線は各軸 2 点以上あるときだけ描く (格子を縮めても図が落ちないように)。
13. **`CHANNEL_PENDING` の T3 分 (20件) を実チャネルへ書き換えた**。
    `name` → `CHANNEL_META` / `drive.distribution` → `CHANNEL_ERROR` / 残り 18 件 →
    `CHANNEL_ROWS`。セクション固有の葉 (`decay.*` / `timescale_sweep.*` /
    `esp_map.*`) には `scope` を付け、**担当する実験の行だけを変え他の実験の行を
    バイト単位で変えない**ことまで要求する。`PENDING_SECTIONS` は
    `frozenset({"washout"})` に絞り、T3 のセクションが pending へ戻る経路を塞いだ。
    時限装置の実効性は変異注入で確認済み (1件を pending へ戻すと
    `test_pending_cases_disappear_once_the_experiment_layer_exists` が発火)。
14. **`test_experiment_registry_covers_the_experiment_directories` を追加**。
    `experiments/` のディレクトリ番号と `main.EXPERIMENTS` のキーが一致することを
    強制する (実験を足して登録し忘れると CLI から静かに消えるため)。

*実測値 (T3 完了時。本番設定 `experiments/02_esp_and_dynamics/config.yaml`)*

- **wall_time_s = 83.26 秒** / 369 行 (2-A 15 + 2-B 18 + 2-C 336)。内訳は
  2-A 3.38s / 2-B 4.06s / 2-C 75.82s。予算 900 秒に対して 9%。
- **受け入れ条件1** (σ_u=0、3レプリケート):

  | ρ | converged | `decay_rate_per_step` (rep0/1/2) | `log ρ` | 相対誤差 |
  |---|---|---|---|---|
  | 0.5 | 1 / 1 / 1 | -0.7466 / -0.7318 / -0.7286 | -0.6931 | 7.71% / 5.58% / 5.12% |
  | 0.8 | 1 / 1 / 1 | -0.2324 / -0.2349 / -0.2290 | -0.2231 | 4.13% / 5.26% / 2.61% |
  | 0.95 | 1 / 1 / 1 | -0.05147 / -0.05152 / -0.05142 | -0.05129 | 0.35% / 0.44% / 0.24% |
  | 1.2 | 0 / 0 / 0 | 約 +1e-5 (減衰しない) | +0.1823 | — |
  | 1.5 | 0 / 0 / 0 | 約 +2e-5 (減衰しない) | +0.4055 | — |

- **受け入れ条件2** (ρ=1.5、3レプリケート全て): σ_u=0 / 0.05 / 0.1 / 0.2 / 0.5 は
  `converged=0`、**σ_u=1.0 と 2.0 は `converged=1`** (`d_tail` は 2.4e-16 / 1.4e-16)。
  記事の目玉はデータで固定された。
- **受け入れ条件3**: `n_rows=369` / 境界近傍 (`|λ|<=0.01`) 37 件 / 比較対象 332 件。
  **`n_false_esp = 0`**。「λ<0 なのに非収束」27 件 (8.13%)。
  強駆動 (σ_u>=0.5) は **158 件中 0 件の不一致**。
  不一致の内訳: σ_u = 0.0 (8件) / 0.05 (7件) / 0.1 (8件) / 0.2 (4件)、
  ρ = 1.1 (6件) / 1.2 (6件) / 1.3 (4件) / 1.4 (4件) / 1.5 (5件) / 1.6 (2件)。
  **σ_u >= 0.5 と ρ <= 1.0 には1件も無い**。
- **受け入れ条件4** (ρ=0.9, σ_u=0.5): `tau_1e` のレプリケート平均は
  リーク率 0.1→1.0 で 13.389 / 6.297 / 3.712 / 1.931 / 1.111 / 0.615 (単調非増加)。
  理論線 `-1/log(1-a)` は 9.491 / 4.481 / 2.804 / 1.443 / 0.831 / (a=1 は 0)。
  実測が理論線より大きいのは線形域の近似が再帰項を無視しているため (単調性は一致)。
- **伝播器のインデックスずれは否定済み**: `check_propagator=True` のまま 336 条件が
  通過。`u[t]` に差し替えると全条件で `ValueError`、検査を切ると λ が
  +15.17 / +17.16 / +16.78 (正しい値は -0.019 / -0.064 / -0.234) になる。
- **図**: 3枚とも 200 dpi (`conftest.png_dpi` で実測)。
- **01 の成果物は 1 バイトも変わっていない**: `comparison.csv` (`wall_time_s` 除く) と
  `comparison_summary.csv` が再生成前後で完全一致。
- テスト: 339 → 365 件 (+26)、全体 9.80 秒 (増分 6.7 秒。予算 90 秒に対して 7%)。

### T4: 実験 2-D —— washout 感度 (想定所要: **M**)

**何をするか**

- `run_washout_sweep(config)`: 実体は **`dataclasses.replace` で `washout` を差し替えて既存 `run_experiment` を呼ぶループ**。公平性 (D-04/05/08) は既存経路が担保。
- **交絡の除去 (D-19)**: washout を増やすと `t0` が上がり `n_usable` が縮むため、`pad_series=True` のとき ~~`length = base_length + (max(grid) - washout)`~~ として**行数を格子全体で一定に保つ**。`pad_series=False` は交絡ありの設計を再現するモード。
  → **この式は実装時に修正した。正しくは `length = base_length + (t0(washout) - t0(min(grid)))`** (下記「実装時に決めたこと」1。仕様の式は符号が逆で、かつ `t0 = max(washout, first_valid)` の飽和を無視している。**この式のまま実装すると行数は一致しない**)。
- 対象は MG と遅延パリティの両方 (図の主役は MG、パリティは「washout に反応しない対照」)。
- `WashoutRow`: `task`, `method`, `washout`, `replicate`, `alpha`, `n_lags`, `nrmse`, `nrmse_std`, `n_train`, `n_val`, `n_test`, `t0`, `pad_series`, `wall_time_s`。
- `plot_washout_sensitivity`: 01 の本番値 (washout=200) に垂直線を引き、変動幅を数値注記。

**受け入れ基準**

- [ ] **受け入れ条件5**: `test_washout_sweep_quantifies_performance_variation` —— NRMSE の (最大/最小) 比が `meta.json` に記録され 1.0 でない。
- [ ] **D-19 guard**: `test_washout_sweep_holds_training_size_constant` —— `pad_series=True` で行数一致、`False` で `n_train` 単調減少。
- [ ] `test_washout_zero_is_worst_or_equal_for_mackey_glass` (**破れたら記事の主張が変わるので止まって相談**)。
- [ ] **受け入れ条件7**: `test_all_four_figures_and_two_csv_in_one_command` + `make ci` 緑。

**実装時に決めたこと (T4 実装者追記。仕様に書かれていなかった / 仕様と違えた選択)**

決定は T5 で `.claude/decisions.yaml` / `docs/design.md` §9 に転記する。

*仕様の式を変えた1件 (最重要。D-19 の本体)*

1. **補償の式を `length = base_length + (max(grid) - washout)` から
   `length = base_length + (t0(washout) - t0(min(grid)))` に変えた**
   (`experiment/washout.py::variant_for`)。理由は2つあり、どちらも実測で確認した。
   - **仕様の式は符号が逆**である。`n_usable = length - max_start_offset - t0` で
     `t0` は washout について増加するので、行数をそろえるには washout が大きい側の
     系列を**伸ばす**必要がある。仕様の式は逆に短くするため、補償なしより差が広がる
     (実測: 変異注入すると行数が 348/310/230/110 と激しく割れる)。
   - **`t0 = max(washout, first_valid)` の飽和がある**ため、washout の差分で補償しても
     そろわない。本番格子では遅延線の最大ラグ 64 のせいで washout=0 と 50 が
     どちらも `t0 = 64` になり、`washout - min(grid)` で補償すると washout=0 だけ
     行数がずれる (実測: 228 対 230)。**`t0` の差分で補償すれば両方の問題が消える**。
   基準を格子の最小値に取ったので補償は常に「伸ばす」側に働き、01 の本番設定より
   短い系列で測ることはない (本番では length 8200 → 8200〜8936)。
   `test_washout_sweep_holds_training_size_constant` は上記2つの誤った式を
   **どちらも実測で落とすことを変異注入で確認済み**。
2. **`t0` を系列生成前に知るため `readout/design.py` に `first_valid_for(spec)` を
   追加**した (公開関数の**追加のみ**。`_validate_inputs` もこれを呼ぶ形にしたので
   予測経路と実経路が同じ値を返すことが構造で保証される)。理由: 補償量を決めるには
   系列を作る前に `t0` が要るが、「遅延線なら `n_lags`」を `experiment/` 側へ書き写すと
   手法を足したときに予測と実際の `t0` が黙って食い違う。
   guard は `test_padding_uses_the_same_t0_as_the_runner`。

*`experiment/washout.py`*

3. **`WashoutRow` は長形式 (1行 = 1 (課題, 手法, washout, レプリケート))** とし、
   `nrmse_std` だけ1段粗い粒度 (**同じ (課題, 手法, washout) のレプリケート間標準偏差**)
   を各行に載せた。理由: 仕様の宣言順は `replicate` と `nrmse_std` を両方持つ。
   `alpha` / `n_lags` / `n_train` は明らかにレプリケート単位なので長形式が唯一の
   読み方であり、`nrmse_std` を群の値として重複させれば全フィールドが意味を持つ。
   図の誤差棒と「変動幅がレプリケート間のばらつきより大きいか」の判断が CSV1枚で
   完結する副次効果もある。
4. **`meta.json` の `n_rows` は `esp_diagnostics.csv` の行数のまま**にし、2-D の行数は
   `washout_sensitivity.n_rows` に分けた。理由: 足し込むと「どちらの CSV の行数か」が
   `meta.json` から読めなくなる。列の違う2枚を1つの数で代表させない。
5. **`MethodSensitivity` に `replicate_std_max` / `spread` /
   `exceeds_replicate_noise` を足した**。理由: 受け入れ条件5 は「比が 1.0 でない」
   としか要求しないが、比が 1.0 でないことは「変動が測れた」以上を意味しない。
   実測では**全 (課題, 手法) で変動幅がレプリケート間のばらつきより小さい**ので、
   この列が無いと成果物だけを見た読者が「washout に性能が反応した」と読む。
6. **主役は MG x ESN に固定** (`HEADLINE_TASK` / `HEADLINE_METHOD`)。掃引に主役の組が
   無ければ `summarize_washout_sensitivity` は `ValueError`。理由: 黙って別の組で
   代用すると `meta.json` の数値が何の変動幅なのか読めなくなる。
7. **`nrmse_min == 0` のとき比は `nan`** (1.0 で埋めない)。理由: 1.0 は「変動が無かった」
   と読めてしまい、「比が定義できなかった」と区別できない。

*図 (`plotting/figures_esp.py::plot_washout_sensitivity`)*

8. **2パネルにし、右を「01 の本番値で正規化した比」にした**。理由: 絶対値だけだと
   MG の 7e-4 と パリティの 1.0 という**水準差**が支配的で、この図の主張である
   1% 未満の変動が読めない。垂直線 (washout=200) と数値注記は両パネルに置く。
9. **注記には比だけでなくレプリケート間 s.d. と判定文を書く** (決定5と同じ理由)。

*テスト*

10. `test_all_four_figures_and_two_csv_in_one_command` は **`ESP_ARTIFACTS` の中身を
    数えず、出力ディレクトリを実際に走査**して数え、`set(produced) == set(ESP_ARTIFACTS)`
    まで assert する。理由: 宣言だけを見るテストは、宣言と実体が食い違ったとき
    (図を落として宣言を消し忘れた / その逆) に黙って通る。
11. `test_washout_sweep_holds_training_size_constant` の**補償なし側は狭義単調減少を
    要求しない** (決定1の `t0` 飽和のため、本番格子では washout=0 と 50 の行数が等しい)。
    要求は「非増加」+「両端で実際に減る」+「`t0` が増えた区間では必ず減る」の3本。
12. `test_padding_does_not_disturb_the_rows_that_are_actually_used` を追加。系列を
    伸ばしても既存の行が1つも書き換わらないこと (補償は末尾に足すだけ) を実測する。
    ここが崩れると「行数は同じだが中身が別物」になり、格子点間の比較が washout の
    効果でなくなる。
13. **配線テストの `washout.*` を `CHANNEL_PENDING` から新設の `CHANNEL_WASHOUT` へ
    書き換え、`PENDING_SECTIONS` を空集合にした**。`CHANNEL_WASHOUT` は
    「2-D の行の指紋が変わる」だけでなく「**2-A/2-B/2-C の行がバイト単位で変わらない**」
    まで要求する。時限装置が pending ゼロで正しく緑になること、および1件を pending へ
    戻すと発火することを変異注入で確認済み。
    `WashoutSweepConfig.base` の既定は 01 の本番設定なので、配線テストとパイプライン
    テストの縮小設定には**専用の縮小 `base` を書いた** (既定のままだと1ケースで数十秒)。

*実測値 (T4 完了時。本番設定 `experiments/02_esp_and_dynamics/config.yaml`)*

- **wall_time_s = 87.69 秒** (T3 の 83.26 秒 + 2-D の 4.4 秒)。予算 900 秒に対して 9.7%。
  成果物は CSV2枚 + 図4枚 + `meta.json`、2-D は 180 行 (6 washout x 2課題 x 3手法 x 5rep)。
- **D-19 の補償 (`pad_series=True`)**: 全格子点で
  `(n_train, n_val, n_test) = (3968, 1190, 2778)` で一致。`t0` は
  washout 0/50/100/200/400/800 に対し 64/64/100/200/400/800 (0 と 50 が同じなのは
  遅延線の最大ラグ 64 による飽和)。系列長は 8200〜8936 に伸びる。
- **受け入れ条件5 (MG x ESN、補償あり)**: NRMSE の (最大/最小) 比 = **1.00763**
  (7.0730e-4 @ washout=400 .. 7.1269e-4 @ washout=800)。
  変動幅 5.40e-6 に対し**レプリケート間 s.d. の最大は 9.22e-5** (17倍)。
  → `exceeds_replicate_noise = false`。**washout に性能が反応したとは言えない**。
- **全 (課題, 手法) の比 (補償あり)**: MG 線形 1.00012 / MG 遅延線 1.00413 /
  MG ESN 1.00763 / パリティ 線形 1.00031 / パリティ 遅延線 1.00094 /
  パリティ ESN 1.00452。**6組すべてで `exceeds_replicate_noise = false`**。
- **`test_washout_zero_is_worst_or_equal_for_mackey_glass` は成立する**が余裕は薄い:
  washout=0 → 7.0936e-4、washout=200 → 7.0772e-4 (差 **0.23%**)。同じ格子点の
  レプリケート間 s.d. が 8.5e-5 (平均の 12%) なので、**順序は成り立つが有意ではない**。
  記事で「washout を短くすると悪化する」と書けるだけの効果は無い。
- **交絡ありモード (`pad_series=False`) との対比 (D-19 の証拠)**: `n_train` は
  3968/3968/3950/3900/3800/3600 と縮み、MG x ESN の NRMSE は
  7.0936e-4 / 7.0936e-4 / 7.0996e-4 / 7.1063e-4 / 7.1209e-4 / 7.1752e-4 と
  **完全に単調増加**する (比 1.01151)。**補償を入れるとこの単調性は消える**
  (最小が washout=400、最大が 800 の非単調なノイズになる)。
  「washout を長く取りすぎると悪化する」と読めてしまう滑らかな曲線は、
  **訓練データ量の効果だった**ことがデータで示された。
- **予測「遅延パリティは washout に反応しない対照」の検証結果 (否定的結果)**:
  パリティ x ESN の比 1.00452 は MG x ESN の 1.00763 より小さく**方向としては
  予測どおり**だが、**両者とも変動幅がレプリケート間のばらつきの内側**であり、
  「MG は反応するがパリティは反応しない」という対比は**データで支持されない**。
  補償を入れた後は**どちらも反応しない**というのが実測である。
  閾値や設定を動かして予測に合わせることはしていない。
- **図**: 4枚とも 200 dpi (`conftest.png_dpi` で実測)。
- **01 の成果物は 1 バイトも変わっていない**: `comparison.csv` (`wall_time_s` 除く) と
  `comparison_summary.csv` が再生成前後で完全一致。
- テスト: 365 → 383 件 (+18)、全体 17.83 秒 (増分 8.0 秒。予算 90 秒に対して 9%)。
  うち 4.4 秒は `test_washout_zero_is_worst_or_equal_for_mackey_glass` /
  `test_production_grid_quantifies_the_variation` が共有する本番格子の掃引
  (記事の数値は本番格子で語るので縮小設定で代用しない)。

### T5: 記録 —— design.md / decisions.yaml / README と閾値感度 (想定所要: **M**)

- `docs/design.md` §9: 距離定義・閾値・窓の根拠、**閾値感度の実測表** (`abs_tol` 3 × `window` 3 の9通りで ESP 成立境界がどれだけ動くか)、δ と再正規化間隔の根拠、入力強度の定義、2-D のタスク選定と `pad_series` の理由、λ と判定の不一致件数。
- `.claude/decisions.yaml` に D-13〜D-19 を追記。
- `README.md` に 02 の実行方法・成果物・実測値の要約 (数値は生成物と突き合わせる形。手書きを増やすなら突き合わせテストも足す)。
- 要件書の未確定事項1・2・3・5 に「決定済み」を追記 (4 は次回サーベイ課題)。

**受け入れ基準**

- [x] `check_decisions.py` が D-13〜D-19 について緑。→ **D-13〜D-22 を追記し 22 件で OK**。
- [x] `tests/test_design.py` が §9 の既定値と**コード上の既定値の一致**を検証 (散文とコードの乖離を機械で殺す)。
      → 新設した `tests/test_design_doc.py` が担当 (下記の決定2)。
- [x] `results/esp_threshold_sensitivity.csv` が9行以上を持ち design.md の表と行数一致。
      → `results/02_esp_and_dynamics/esp_threshold_sensitivity.csv` (9行)。

**実装時に決めたこと (T5 実装者追記。仕様に書かれていなかった選択)**

*閾値感度の測り方 (仕様は「9通りで境界がどれだけ動くか」としか書いていない)*

1. **臨界 rho を「rho の昇順で収束率が過半数を割る最初の rho」と定義**し、格子内に
   境界が無い場合は `nan` を返す (`experiment/threshold.py::critical_rho`)。理由:
   「格子の上端」と「境界が格子の外にある」を同じ値で表すと、格子を広げたときに
   表の読み方が変わる。実測では sigma_u=2.0 が唯一の `nan` (rho=1.9 まで全点 ESP 成立)。
2. **感度掃引は軌道を1回だけ作り、9通りの `cfg` で判定だけをやり直す**
   (`run_threshold_sweep`)。`esp_convergence` は `(states, companions, cfg)` の純関数
   なので判定基準を変えても軌道は変わらない。素直に9回掃引すると実行時間も9倍
   (実測 60.7 秒 → 9分超) になる。このために `experiment/esp.py` へ
   **`simulate_condition` / `Trajectories` を追加**し (公開関数の追加のみ)、
   `evaluate_condition` はそれに委譲する形にした。
3. **基準の組 (`abs_tol=1e-6`, `window=200`) が格子に無ければ `ValueError`**。
   理由: `max_abs_shift` / `n_sigma_shifted` は「基準からのずれ」なので、基準が
   格子に無い掃引を通すと何からのずれか読めない数値が CSV に出る。
4. **`esp_threshold_sensitivity.csv` を `ESP_ARTIFACTS` に入れない**。
   `run_02.py --threshold-sweep` (= `make threshold-02`) でのみ再生成する。理由:
   記事に載る成果物ではなく「既定値が結論を作っていない」ことの根拠であり、
   2-C の格子をもう一度回すので `make figures-02` に足すと 88 秒 → 149 秒になる。
   `test_all_four_figures_and_two_csv_in_one_command` (出力ディレクトリを実走査する
   T4 の決定10) が本体の成果物を7点に固定しているので、混ぜると本体側が壊れる。
5. **CSV の列は「`ThresholdRow` の宣言順 + `critical_rho_by_sigma` を sigma_u 別の
   列に展開」**とした。理由: sigma_u 格子は設定値なので列名を dataclass の
   フィールドとして固定できない。既存の慣習 (「宣言順が CSV 列順の単一の真実」) を
   1フィールドの展開という形で緩めており、展開規則は
   `threshold_csv_columns` / `threshold_row_as_dict` の対に閉じて
   `test_threshold_csv_header_matches_rows` が固定する。

*記録の機械検査 (仕様は「`tests/test_design.py` が検証」とだけ書いていた)*

6. **検査は `tests/test_design.py` ではなく新設の `tests/test_design_doc.py` に置いた**。
   理由: サイクル1 の `tests/test_design.py` は**設計行列** (`build_design_matrix`) の
   テストであって design.md のテストではない (仕様の参照先が誤り)。既存ファイルに
   同居させると、ファイル名から中身が読めなくなる。
7. **§9 の既定値表は3列目に「コード上の出どころ」をドット区切りで書く**形にし、
   テストが `importlib` で解決して2列目の値と突き合わせる。frozen + slots の
   dataclass は既定値がクラス属性として残らないため、dataclass 型への属性参照は
   `dataclasses.fields` の既定値 (または `default_factory()`) として解決する。
   理由: 「出どころを書けない値は表に載せられない」という制約が同時に効き、
   根拠の無い数値が表に紛れ込む経路も塞がる。変異注入 (`window` 200→250) で発火を確認済み。
8. **感度表は行数だけでなく全セルを CSV と突き合わせる**
   (`test_design_table_values_match_the_threshold_csv`)。仕様の受け入れ基準は
   「行数一致」だが、行数だけでは値のドリフトが素通りする。変異注入
   (臨界 rho 1.7→1.6) で発火を確認済み。
9. **`max_observed_growth` を `EspRow` (= CSV の列) には足さない**。理由: 公開 API
   (CSV スキーマ) の変更になるうえ、必要なのは「線形域を外れていない」という
   1つの主張だけである。本番格子の実測値は design.md §9.3 に置き、
   `test_perturbation_growth_stays_far_below_the_runaway_limit` で
   「`max_growth` に対して2桁以上の余裕」を縮小条件で固定した (D-11 と同じ形)。

*ソースの実測値の訂正 (T3 の暫定値が残っていたもの)*

10. `experiment/esp.py` の `STRONG_DRIVE_SIGMA` と `VerdictAgreement` の docstring に
    あった「不一致 25 件 / 強駆動 140 条件」を、本番実測の **27 件 / 158 条件**に
    直した (`meta.json` の `verdict_lyapunov_agreement` が正)。T5 は記録の
    タスクなので、design.md・decisions.yaml と食い違う数値をソースに残さない。

*README (仕様は「手書きを増やすなら突き合わせテストも足す」とだけ書いていた)*

11. **README に足した数値は「臨界 rho の行」と「`meta.json` のキー名つきの4値」だけ**に
    絞り、`tests/test_readme_summary.py` に突き合わせを追加した。理由: 数値を
    `wall_time_s = 87.69` のように**生成物のキー名と同じ形**で書くと、読者にとっての
    出どころとテストの正規表現が同じものになる。テストの**件数**は書かない
    (design.md §7 の注記どおり)。変異注入 (臨界 rho / 件数 / wall time) で発火を確認済み。

*実測値 (T5 完了時)*

- `check_decisions.py`: **22 件で OK**、全 guard_test が緑。
- `esp_threshold_sensitivity.csv`: 9 行。**9通りすべてで臨界 rho は不変**
  (`n_sigma_shifted = 0`)。判定が動いたのは `window=400` の 1 条件のみ (208 → 207)。
- 臨界 rho (基準): sigma_u = 0 / 0.05 / 0.1 / 0.2 / 0.5 / 1.0 / 2.0 に対し
  1.0 / 1.1 / 1.1 / 1.3 / 1.5 / 1.7 / 格子外 (>1.9)。
- `max_observed_growth` (2-C 336 条件): 最大 **1.5372** / 中央値 1.0873 /
  `max_growth=1000` に対して 650 倍の余裕。
- 多安定性の直接観測 (rho=1.1, sigma_u=0, rep0): 4軌道とも末尾200ステップの時間 s.d. が
  6.4e-16、`d(traj0,traj1)=1.2e-16` / `d(traj0,traj3)=1.6e-16` に対し
  `d(traj0,traj2)=5.3e-01`、かつ `||x0* + x2*||/sqrt(N)=1.3e-16`
  (= `x2*` はちょうど `-x0*`)。この条件の λ は -0.032 で `converged=0`。
- `make threshold-02` の wall time: **60.70 秒** (本体 87.69 秒とは別枠)。

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
  rule: "washout 感度実験では、washout を増やしても訓練/検証/テストの行数が変わらないよう系列長を伸ばして補償する (pad_series=True が既定)。補償量は washout の差分ではなく t0 の差分 length = base_length + (t0(washout) - t0(min(grid))) とし、t0 は系列を作る前に readout.design.first_valid_for から予測する。補償なしは pad_series=False として残し対比に使う"
  rationale: "make_split は n_usable = T - max_start_offset - t0 で行数を決めるため、washout を増やすと訓練データ量が同時に減る。素直に掃引すると『washout の効果』と『訓練データ量の効果』が交絡し、受け入れ条件5 が別の量の測定になる。この交絡は滑らかな単調曲線として出るため図を見ても気づけない (実測: 補償なしだと MG x ESN の NRMSE が washout に対して完全に単調増加し、比 1.0115。補償を入れると単調性が消え比 1.0076 の非単調なノイズになる)。補償量を washout の差分にしないのは t0 = max(washout, 各手法の first_valid) が飽和するためで、本番格子では遅延線の最大ラグ 64 により washout=0 と 50 がどちらも t0=64 になり、washout 差分での補償では washout=0 だけ行数がずれる (実測 228 対 230)"
  guard_test: "tests/test_experiment_washout.py::test_washout_sweep_holds_training_size_constant"

# T3 実装中に生まれた決定 (ユーザー承認済み)。T5 で decisions.yaml へ転記する。
- id: D-20
  rule: "λ の符号と ESP 判定の整合の要求は非対称にする。『λ>0 なのに収束 (偽の ESP)』は |λ|>0.01 の全条件で 0 件を要求し、『λ<0 なのに非収束』は許容して件数と σ_u / ρ の内訳を meta.json の verdict_lyapunov_agreement に残す。両者の完全一致は駆動が十分ある領域 (σ_u >= 0.5) でのみ要求する"
  rationale: "条件付き Lyapunov 指数は参照軌道まわりの局所量なので多安定性を原理的に検出できない。tanh は奇関数なので x* が不動点なら -x* も不動点であり、どちらも局所安定 (λ<0) でありながら初期状態によって行き先が割れる (= ESP 不成立)。ρ=1.1・σ_u=0 で4軌道を直接観測して確認済み (全軌道が不動点に到達し、3本が同一点・1本が距離 0.526 の別の点)。対称な一致を要求すると、この実在の現象を実装バグとして潰すことになる。逆向き (λ>0 なのに収束) は理論上あり得ないので 0 件を厳格に要求し、伝播器のインデックスずれや閾値の緩みはそちらで落とす。本番格子 336 条件の実測で偽の ESP は 0 件、σ_u>=0.5 の 158 件も不一致 0 件"
  guard_test: "tests/test_experiment_esp.py::test_lyapunov_sign_agrees_with_verdict_away_from_boundary"

- id: D-21
  rule: "02 の ESN は bias_scale=0 で構成し、無入力条件 (sigma_u=0) を真の無入力にする。比較軌道は n_pairs=10 本引く"
  rationale: "(1) 定数バイアスは [1; u] の先頭成分に掛かる振幅一定の入力そのものであり、D-17 が入力強度を駆動信号の標準偏差で定義している以上その定義に入らない常時入力は 0 にする。実測: bias_scale=0.1 では無入力・ρ=1.2 でも2軌道が収束し受け入れ条件1 が成立しない。(2) 無入力・ρ>1 の ESN は +x* / -x* の対をなす吸引子を持つことがあり、比較軌道が k 本すべて参照軌道と同じ側へ落ちる確率が約 2^-k 残る。実測: n_pairs=3 では特定の draw で ρ=1.05〜1.5・無入力が『収束』と誤判定された。10 本で全レプリケートが正しく非収束になり、ρ<1 側の判定は変わらない"
  guard_test: "tests/test_experiment_esp.py::test_no_input_decay_matches_spectral_radius"

- id: D-22
  rule: "esp_convergence に渡す ctx の washout は 0 に固定する (ESP_DISTANCE_WASHOUT)。drive.washout は λ と自己相関の ctx にだけ渡す。02 の esp.fit_skip の既定は 10"
  rationale: "esp_convergence で washout が効くのは減衰率の当てはめ開始位置だけで、判定 (末尾 window の中央値) には効かない。2-A は過渡そのものを見せる図であり、距離が floor に届く前に当てはめを始めないと減衰率が測れない。実測: 無入力 ρ=0.5 の距離は t≈46 で 1e-14 を割るため、当てはめ開始が washout(200)+fit_skip(50)=250 だと当てはめ点が 0 点になり decay_rate_per_step が nan になる (ρ=0.8 も同様)。過渡を捨てるのは λ と自己相関の側の要求なので、ctx を分ける"
  guard_test: "tests/test_experiment_esp.py::test_decay_fit_starts_before_the_distance_underflows"
```

---

## 7. 想定リスク (起きたら止まって相談)

1. **2-C で「強入力なら ρ>1 でも ESP」が再現できない**。確認順序: (a) `input_scale` が弱すぎないか (2-C では 1.0 固定を推奨)、(b) 判定窓が短すぎないか、(c) `n_steps` が過渡を含んでいないか。それでも出なければ格子を無制限に広げる前に相談。
2. **λ の符号と ESP 判定が広範囲で食い違う**。境界近傍以外で不一致が 5% を超えたら実装誤りの可能性が高い (最有力は伝播器の入力インデックスずれ)。
   → **T3 で発火した (実測 8.13%)。調査の結果、実装誤りではなく多安定性という実在の現象**だった。
   伝播器のインデックスずれは先に否定済み (`check_propagator` が全条件で通過。`u[t]` へ差し替えると全条件で `ValueError` になり、検査を切ると λ が +15〜+17 という桁違いの値になる)。
   結論は D-20 として登録する。**次にこの比率を見たときは、まず「λ<0 なのに非収束」の一方向に偏っているか、σ_u <= 0.2 かつ ρ > 1 に限局しているかを確認すること**。
   偏っていれば多安定性、偏っていなければ実装を疑う。`λ>0 なのに収束` が 1 件でもあれば実装誤りである。
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
