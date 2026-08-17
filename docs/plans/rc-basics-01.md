# 仕様: PDCA サイクル1 — rc-basics-01 実験実装

## 1. ゴール

3ベースライン（線形 / 遅延線 / ESN）を**同一 API・同一分割・同一 alpha 格子**で比較できる `rc_basics_lab` 基盤を構築し、記事01の図2枚と `comparison.csv` を1コマンドで再生成できる状態にする。

## 2. 現状認識

### リポジトリ内の関連箇所

| パス | 現状 | 本サイクルでの扱い |
|---|---|---|
| `pyproject.toml:1-16` | `name = "template"`, `dependencies = []`, `pythonpath = ["."]` | 改名 + 依存追加 + `[build-system]` 追加 |
| `pyproject.toml:22-44` | ruff (line-length 88, T20 で print 禁止), mypy strict + `disallow_any_explicit` | **変更しない**。この制約下で書く |
| `Makefile:42` | `ci: lock-check lint fmt-check type test` | 変更しない（`figures` ターゲットのみ追加可） |
| `main.py:1-12` | テンプレの雛形（ログ出力のみ） | 不明点 Q2 参照 |
| `tests/test_main.py:1-11` | `main.py` のテスト | 同上 |
| `.github/workflows/python-ci.yaml:31` | `make ci` を呼ぶだけ | 変更しない |
| `.claude/decisions.yaml` | **未作成** | T1 で雛形から作成 |

### 既存の慣習で守るべきもの（3個）

1. **検証の単一の真実は `Makefile`**。テスト実行方法を増やさない（`make ci` が Stop フック / CI / ローカルの共通入口）
2. **`Any` 禁止 / `# type: ignore` 禁止**。numpy は `npt.NDArray[np.float64]`、設定オブジェクトは frozen dataclass、多態は `Protocol`
3. **`print()` 禁止**（ruff T20）。実験スクリプトの進捗も `logging` を使う

### 影響範囲

新規モジュール群のみ。既存コードへの破壊的変更は `pyproject.toml`（パッケージ名・依存・build-system）と `uv.lock` に限定される。ここで確定する `diagnostics/base.py` の型と `readout/design.py` の API は **サイクル 02〜05 全部が乗る土台**であり、後から変えると4サイクル分に波及する（本仕様が最も時間を割いた箇所）。

## 3. 前提・制約

### ハード制約（絶対に変えない）

- `[tool.ruff.lint]` / `[tool.mypy]` の設定を緩めない（`disallow_any_explicit`, `strict`, `T20` を含む）
- `make ci` の構成（`lock-check lint fmt-check type test`）を変えない
- 依存は **numpy / scipy / matplotlib / pyyaml** のみ。可視化・設定・検証のために新規ランタイム依存を足さない（dev 依存の `types-PyYAML` は可）
- `diagnostics` は ESN クラスに一切依存しない。`import` レベルで `reservoir` を参照したらレビュー不合格
- 全実験が **CPU で5分以内**に完走する
- ライセンス Apache-2.0 / 個人名義
- 作業ブランチ `feat/rc-basics-lab` 上で作業する（`main` では auto-commit が動かない）

### ソフト制約（理由があれば変えてよい）

- ディレクトリ名 `RC_basic_lab` は変えない（Python パッケージ名 `rc_basics_lab` / 配布名 `rc-basics-lab` で吸収する）
- スペクトル半径は `numpy.linalg.eigvals`（密行列・決定論的）で算出する。ARPACK は反復初期値で微小に揺れ再現性が落ちるため、N ≤ 1000 の本連載規模では密で押す
- 検証分割は時系列なのでシャッフルせず連続区間で切る
- 図は `savefig.dpi = 200`（retina 相当）

### 未確定事項への回答（ユーザー承認方針2に基づき、文献既定値を初期値として YAML 項目化）

| 未確定 | 決定値 | 根拠 |
|---|---|---|
| 1. Mackey-Glass | τ=17, β=0.2, γ=0.1, n=10 / RK4 h=0.1 / サンプリング間隔 Δt=1.0（10ステップごと）/ 積分バーンイン 1000 サンプル破棄 / 有効長 8000 / washout 200 / 残り 7800 を 0.5:0.15:0.35 = 3900/1170/2730 に連続分割 | τ=17 は Farmer(1982) 以来のカオス域標準値で、RC ベンチマークの事実上の既定。Δt=1.0 は Lukoševičius の実践ガイド系の慣例。分割比は検証で alpha と k を選ぶ余地を残しつつテストを最大化する配分 |
| 2. 遅延パリティ | **既定: n_bits=2, delay=1**（`y[t] = u[t-1]·u[t-2]`, u[t] は ±1 の i.i.d.）。副条件として n_bits=3, delay=2 も YAML の条件リストに入れて併記 | 下記「なぜこの設定なら受け入れ条件2を必ず満たすか」参照 |
| 3. 誤差指標 | **NRMSE に統一**。定義は `RMSE / std(y_true)`（ddof=0、評価区間の目標系列の標準偏差で正規化）。CSV には `rmse` / `nmse`(=NRMSE²) / `nrmse` / `sign_accuracy` を併記 | NRMSE=1 が「平均予測と同等」を意味するため、パリティ課題の失敗判定が解釈可能になる。`nmse` 併記は 03 の未確定4（先行研究との比較）への前倒し対応 |
| 4. リポジトリ名 | ディレクトリは `RC_basic_lab` のまま。`[project] name = "rc-basics-lab"`、パッケージ `src/rc_basics_lab/`、import 名 `rc_basics_lab` | 要件書の `rc-basics-lab` を配布名として満たしつつ、ディレクトリ改名という不可逆・広範囲な操作を避ける |

#### なぜ「n_bits=2, delay=1」なら受け入れ条件2を必ず満たすか（最重要）

`u[t] ∈ {-1,+1}` i.i.d.、`y[t] = u[t-1]·u[t-2]` とすると、任意の `k` について

```
E[y[t]] = 0,   E[y[t]·u[t-k]] = E[u[t-1] u[t-2] u[t-k]] = 0
```

（k∉{1,2} なら独立3変数の積で期待値0、k∈{1,2} なら `u²=1` により残る1変数の期待値が0）。
つまり **`{1, u[t], u[t-1], …, u[t-k]}` の張る線形空間に対して目標が厳密に直交する**。したがって母集団最適な線形予測子は恒等的に0であり、`NRMSE → 1.0`、符号正解率 → 0.5 になる。これは「経験的に苦戦する」ではなく**解析的に失敗が保証されている**。

重要な副産物として、**遅延線 `[1, u[t..t-k]]` も同じ理由で失敗する**。これは要件書の「最も安い対照は遅延線」という連載の芯（記事01 §6）に対して、「遅延線が失敗する＝記憶ではなく非線形性が要る」という最も鮮明な実演になる。一方 tanh リザバーは状態内に `u[t-1]u[t-2]` 型の交差項を生成するため解ける（古典的な temporal XOR / delayed parity 課題。Bertschinger & Natschläger 2004 系の設定）。

数値受け入れライン（5シード平均、テスト区間）:
- 線形・遅延線: `NRMSE >= 0.95` かつ `sign_accuracy <= 0.60`
- ESN: `NRMSE <= 0.50` かつ `sign_accuracy >= 0.90`
- 両者の NRMSE の ±1σ 区間が重ならない

`sign(0)` の扱いは `+1` に固定する（それでも正解率は約0.5）。

## 4. タスク分解

### T1: リポジトリ基盤 — パッケージ化・設定層・シード層・計量層

- **何をするか**
  - `pyproject.toml`: `name = "rc-basics-lab"`, `[build-system] hatchling`, `[tool.hatch.build.targets.wheel] packages = ["src/rc_basics_lab"]`。`uv add numpy scipy matplotlib pyyaml` / `uv add --group dev types-PyYAML`
  - mypy が scipy/matplotlib のスタブ欠如で落ちる場合、**`[[tool.mypy.overrides]]` の `ignore_missing_imports` のみ**で対処する（呼び出し側に `# type: ignore` を撒かない）
  - `src/rc_basics_lab/types.py`: `FloatArray: TypeAlias = npt.NDArray[np.float64]` を単一定義
  - `src/rc_basics_lab/config.py`: frozen dataclass 群 + `load_config(path) -> ExperimentConfig`。**未知キーは `ConfigError` で即エラー**（配線されていないパラメータの検出装置）
  - `src/rc_basics_lab/seeds.py`: `SeedConfig(reservoir:int, task:int, split:int)` と `make_rng(cfg, stream: SeedStream, replicate: int) -> np.random.Generator`。`np.random.SeedSequence` の `spawn_key` でストリームを分離
  - `src/rc_basics_lab/metrics.py`: `nrmse` / `rmse` / `nmse` / `sign_accuracy`
  - `src/rc_basics_lab/meta.py`: `collect_meta(config) -> dict[str, object]`（git commit hash・UTC timestamp・numpy/scipy バージョン・設定ダンプ・シード）。git 取得失敗時は `"unknown"` にフォールバックし例外を投げない
  - `docs/design.md` を作成し、上表「未確定事項への回答」の根拠を転記
  - `.claude/decisions.yaml` を雛形から作成し §6 の D-01〜D-10 を追記
- **触るファイル**: `pyproject.toml`, `uv.lock`, `src/rc_basics_lab/{__init__,types,config,seeds,metrics,meta}.py`, `tests/test_{config,seeds,metrics}.py`, `docs/design.md`, `.claude/decisions.yaml`
- **受け入れ基準**
  - `uv run python -c "import rc_basics_lab"` が通り、`make ci` が緑
  - `tests/test_config.py::test_unknown_key_raises` — YAML に未定義キーを混ぜると `ConfigError`
  - `tests/test_metrics.py::test_nrmse_of_mean_predictor_is_one` — 目標の平均を返す予測子の NRMSE が `1.0`（`pytest.approx`, rtol=1e-12）
  - `tests/test_seeds.py::test_streams_are_independent` — `reservoir` シードのみ変えると reservoir ストリームの乱数列が変わり、`task`/`split` ストリームは**バイト単位で不変**。3ストリームすべてについて対称に検証
  - `tests/test_meta.py::test_meta_is_json_serializable` — `json.dumps` が通り、`commit` キーが存在する
- **想定所要**: M
- **実装時に決めたこと（仕様に無かった箇所。T3〜T5 はこれに合わせる）**
  - `ExperimentConfig` の初期セクションは `name / n_replicates / seeds / split / ridge / mackey_glass / delay_parity` とした。理由: §3 で数値が確定しているものだけを T1 で持ち、`esn` セクションは T3 の `ESNConfig` を再利用して T4 で追加する（同じ dataclass を2箇所に書かないため）。ローダはフィールド型から再帰的に構築するので、セクション追加時にローダ側の変更は不要
  - `SplitConfig` に `max_start_offset: int = 200` を置いた。理由: T4 の「split シードで系列内の開始オフセットを選ぶ」（D-06）にオフセット上限のパラメータが要るため。値は washout と同じ 200
  - `RidgeConfig` に `n_lags_grid: tuple[int, ...] = (1, 2, 4, 8, 16)` を置いた。理由: D-08 の「検証で選ぶのは alpha と遅延線の `n_lags` のみ」を YAML で表現するため。alpha 格子と同じく単一キーに集約する
  - `nrmse` は `std(y_true) == 0` のとき `ValueError`。理由: 0 除算で inf/nan を返すと下流の集計が静かに壊れる。定数目標は本連載に現れない
  - `sign_accuracy` は `sign(0) = +1`（§3 の指定を実装に反映）
  - 設定値の型変換は暗黙の緩和をしない（`int` フィールドに float を渡すと `ConfigError`、`bool` は `int` として受理しない）。理由: D-09 と同じ動機で、静かな誤読を作らない
  - `src/rc_basics_lab/py.typed` を追加した。理由: これが無いと、インストール済みパッケージとして `rc_basics_lab` を import する外部ファイル（T4/T5 の `experiments/*/run.py` が該当しうる）で mypy が型を読めず、`Diagnostic` プロトコル適合の検査（D-01 の guard）が静かに無効化される
  - `.claude/decisions.yaml` には D-01/D-02/D-06/D-09 のみ記載した。理由: `check_decisions.py` が guard_test の実在を検証するため、T3〜T5 のテストに紐づく D-03/D-04/D-05/D-07/D-08/D-10 は各担当タスクで追記する（ファイル冒頭のコメントに明記済み）

### T2: 診断層インターフェース（02〜05 の土台）+ PCA 診断 + ダミー実装

- **何をするか**
  - `src/rc_basics_lab/diagnostics/base.py` に以下を確定する（**このサイクルの最重要成果物**）:

  ```
  DiagnosticContext (frozen dataclass, 全フィールドに既定値):
      washout: int = 0
      dt: float = 1.0
      seed: int | None = None
      companion_states: tuple[FloatArray, ...] = ()   # 02 の第2軌道・摂動軌道用
      # 以後の拡張は「既定値つきフィールドの追加」のみ許可する

  DiagnosticResult (frozen dataclass):
      name: str
      scalars: Mapping[str, float]        # 例 {"mc_total": 12.3}
      arrays: Mapping[str, FloatArray]    # 例 {"mc_profile": ...}
      params: Mapping[str, str]           # meta.json / CSV へそのまま流す
      def to_row(self) -> dict[str, float | str]   # CSV 化の単一経路

  Diagnostic (Protocol):
      def __call__(
          self,
          X: FloatArray,                       # (T, N) 必須・位置引数
          u: FloatArray | None = None,         # (T, D_in)
          y: FloatArray | None = None,         # (T, D_out)
          *,
          ctx: DiagnosticContext | None = None,
      ) -> DiagnosticResult: ...

  validate_diagnostic_input(X, u, y, ctx) -> None   # 形状不整合で ValueError
  ```

  - **設計根拠（02〜05 先読みの結果）**: 要件書の literal な `f(X, u, y)` を位置引数として保ちつつ、拡張は「既定値つき keyword」1点に集約する。02 の ESP は2軌道を要求するが（要件_02 設計判断1）、これを第2位置引数にすると MC/IPC と署名が割れる。`ctx.companion_states` に逃がすことで、02 の ESP・条件付き Lyapunov（摂動軌道は任意本数）、04 の Lyapunov 時間正規化（`ctx.dt`）、03 のシャッフルサロゲート（`ctx.seed`）がすべて**署名変更なし**で乗る。全診断が `DiagnosticResult` を返すため、CSV 出力・meta.json 出力の下流コードを 02〜05 で書き直さない
  - `src/rc_basics_lab/diagnostics/dummy.py`: `state_mean_norm` — `X` だけを使い `DiagnosticResult` を返す最小実装（移植性テストの被験体）
  - `src/rc_basics_lab/diagnostics/state_space.py`: `state_pca(X, u, y, *, ctx)` — 中心化 → 特異値分解 → `explained_variance_ratio`, `cumulative_ratio`, `n_components_95`（累積95%到達に要する主成分数）, `participation_ratio`（`(Σλ)²/Σλ²`）, `pc_scores`（先頭2成分）を返す。これが実験1-B の数値的裏付けと `fig_state_space.png` の両方を供給する
  - 02〜05 用の空パッケージ（`diagnostics/__init__.py` に将来モジュール名をコメントで予約）
- **触るファイル**: `src/rc_basics_lab/diagnostics/{__init__,base,dummy,state_space}.py`, `tests/test_diagnostics_base.py`, `tests/test_diagnostics_state_space.py`
- **受け入れ基準**
  - `tests/test_diagnostics_base.py::test_dummy_diagnostic_conforms_to_protocol` — `d: Diagnostic = state_mean_norm` の代入が mypy strict を通り（`make type` で検証）、かつ `d(X)` が `u`/`y`/`ctx` なしで動く
  - `tests/test_diagnostics_base.py::test_diagnostic_accepts_external_state_series` — **ESN を一切 import せず**、`rng.standard_normal((300, 20))` で作った外部状態系列で `state_mean_norm` と `state_pca` が動く（受け入れ条件6・移植性の担保）
  - `tests/test_diagnostics_base.py::test_diagnostics_package_does_not_import_reservoir` — `diagnostics` 配下の全モジュールのソースに `rc_basics_lab.reservoir` が現れない（AST または文字列検査）
  - `tests/test_diagnostics_state_space.py::test_participation_ratio_of_isotropic_gaussian` — `(5000, 10)` の等方ガウスで `participation_ratio` が 10 に対し相対誤差 10% 以内
  - `tests/test_diagnostics_state_space.py::test_low_rank_input_has_smaller_effective_dimension` — rank 3 の合成データで `n_components_95 == 3`
  - `validate_diagnostic_input` が `X.shape[0] != u.shape[0]` で `ValueError` を送出
- **想定所要**: M
- **実装時に決めたこと（仕様に無かった箇所。02〜05 はこれに従う）**
  - `validate_diagnostic_input(X, u=None, y=None, ctx=None)` の `ctx` は**位置引数**（`Diagnostic.__call__` の `ctx` は仕様どおり keyword-only）。理由: 検証関数は診断の内部実装であり、プロトコルの一部ではない
  - 1次元の `X` / `u` / `y` は受理せず `ValueError`。理由: `(T,)` を `(T, 1)` と黙って解釈すると `(1, T)` の取り違えを検出できなくなる。呼び出し側で明示的に `reshape` させる
  - `validate_diagnostic_input` は形状に加えて `ctx.washout`（0 以上・系列長未満）、`ctx.dt`（正）、`ctx.companion_states`（各要素が `X` と同形状）も検証する。理由: 02/04 でこれらが実際に使われる前に、配線ミスを1か所で落とす
  - `DiagnosticResult.to_row()` は `{"diagnostic": name} + params + scalars` を返し、`arrays` は含めない。キー衝突は `ValueError`。理由: 静かな上書きで CSV の列が消えるのを防ぐ
  - `DiagnosticResult` の `scalars` / `arrays` / `params` は既定値 `{}`（空）。理由: ダミー診断や将来の診断が3種すべてを埋める必要をなくす
  - `base.resolve_context(ctx)` を追加（`None` → 既定 `DiagnosticContext`）。理由: 各診断が同じ `if ctx is None` を書き写すのを避ける
  - `state_pca` の `n_components_95` は「累積寄与率が 0.95 **以上**になる最小の主成分数」。閾値はモジュール定数 `_CUMULATIVE_THRESHOLD` に置き、`params["cumulative_threshold"]` として結果にも残す
  - `state_pca` は共分散行列を作らず SVD で解き、固有値を `s**2 / (T-1)`（不偏）とする。理由: 条件数の悪化を避ける
  - `state_pca` は `pc_scores` として先頭2成分（`U[:, :2] * s[:2]`）を返す。状態が定数（全分散 0）なら `ValueError`
  - `diagnostics/__init__.py` は `Diagnostic` / `DiagnosticContext` / `DiagnosticResult` / `validate_diagnostic_input` / `state_mean_norm` / `state_pca` を再輸出し、02〜05 のモジュール名を docstring で予約する

### T3: ESN コア + 設計行列の単一 API + リッジ/alpha 選択

- **何をするか**
  - `src/rc_basics_lab/reservoir/esn.py`:
    - `ESNConfig(frozen)`: `n_units, spectral_radius, leak_rate, input_scale, bias_scale, density, activation="tanh", state_noise=0.0`
    - `ESN.__init__(config, rng)` — `W_in ~ U[-input_scale, input_scale]`（密, shape `(N, 1+D_in)`）、`W` は密度 `density` のスパース → `numpy.linalg.eigvals` で実測したスペクトル半径で除し `spectral_radius` に正規化
    - `ESN.step(x, u, rng=None) -> FloatArray` と `ESN.run(u, x0=None, rng=None) -> FloatArray` を**今サイクルで両方公開する**。`x0`（02 の2初期状態）、`state_noise`（04 のノイズ注入）、`step`（04 の閉ループ）を先に切っておき、02/04 で公開 API を変更しない
    - 更新式: `x[t] = (1-a)·x[t-1] + a·tanh(W_in·[1;u[t]] + W·x[t-1] + noise)`
    - 既定値（design.md に根拠を記録・**検証で調整しない**）: MG 用 `N=200, ρ=0.9, leak=0.3, input_scale=0.5, density=0.1` / パリティ用 `N=200, ρ=0.9, leak=1.0, input_scale=1.0, density=0.1`
  - `src/rc_basics_lab/readout/design.py`（**受け入れ条件1の本体**）:

  ```
  PassthroughSpec(bias: bool = True)
  DelayLineSpec(n_lags: int, bias: bool = True)
  ReservoirSpec(include_input: bool = True, bias: bool = True)
  FeatureSpec = PassthroughSpec | DelayLineSpec | ReservoirSpec

  DesignMatrix(frozen):
      phi: FloatArray                 # (T, F)
      first_valid: int                # 有効行の開始 index
      feature_names: tuple[str, ...]

  build_design_matrix(spec, u, states=None) -> DesignMatrix
  ```

    - 3ベースラインは `FeatureSpec` の差**のみ**で切り替わる。`ReservoirSpec` に `states=None` を渡したら `ValueError`
    - `first_valid` は遅延線で `n_lags`、他で 0。**実験ランナーは全手法の `first_valid` の最大値と washout の最大値を取って単一の `t0` とし、全手法をまったく同じ行集合で学習・評価する**
  - `src/rc_basics_lab/readout/ridge.py`:
    - `fit_ridge(phi, y, alpha) -> FloatArray` — 閉形式 `(ΦᵀΦ + α·D)⁻¹ΦᵀY`、`D = diag(0, 1, 1, …)`（**バイアス列は正則化しない**）、`scipy.linalg.solve(..., assume_a="pos")`
    - `select_alpha(phi_tr, y_tr, phi_val, y_val, alphas) -> AlphaSelection(alpha, val_nrmse, curve)` — 検証 NRMSE 最小、同点なら大きい alpha（保守側）
    - 既定 alpha 格子: `numpy.logspace(-8, 2, 11)`（YAML の `ridge.alpha_grid` で上書き可、**全手法が同一のこの1キーを読む**）
- **触るファイル**: `src/rc_basics_lab/reservoir/{__init__,esn}.py`, `src/rc_basics_lab/readout/{__init__,design,ridge}.py`, `tests/test_reservoir.py`, `tests/test_design.py`, `tests/test_ridge.py`
- **受け入れ基準**
  - `tests/test_reservoir.py::test_spectral_radius_matches_config` — 生成した `W` の実測スペクトル半径が `config.spectral_radius` に rtol 1e-10 で一致
  - `tests/test_reservoir.py::test_spectral_radius_is_deterministic` — 同一シードで2回生成した `W` がバイト一致
  - `tests/test_reservoir.py::test_leak_rate_changes_state_autocorrelation` — leak 0.1 と 1.0 で状態のラグ1自己相関が有意に異なる（**値を変えたら出力が変わる**）
  - `tests/test_reservoir.py::test_x0_and_state_noise_are_wired` — `x0` を変えると初期の状態が変わり、`state_noise>0` で同一入力の2回実行が一致しない（02/04 用 API の配線確認）
  - `tests/test_design.py::test_three_specs_share_one_api` — 3つの `FeatureSpec` が同じ `build_design_matrix` 呼び出しで処理され、列数が `1+D`, `1+D(k+1)`, `1+D+N` になる
  - `tests/test_design.py::test_delay_line_first_valid_equals_n_lags`
  - `tests/test_design.py::test_n_lags_changes_column_count` — `n_lags` を変えると列数と `first_valid` が変わる
  - `tests/test_ridge.py::test_bias_column_is_not_penalized` — 定数目標 `y = c` に対し、alpha を極端に大きくしてもバイアス係数が `c` に収束し、他の係数は0に潰れる
  - `tests/test_ridge.py::test_alpha_changes_coefficient_norm` — alpha 単調増加に対し係数ノルムが単調減少
  - `tests/test_ridge.py::test_closed_form_matches_naive_solution` — 小さい系で `np.linalg.lstsq` ベースの素朴解と一致
- **想定所要**: L
- **実装時に決めたこと（仕様に無かった箇所。T4・T5 と 02〜05 はこれに合わせる）**
  - `ESN.__init__(config, rng, *, n_inputs: int = 1)` とした。理由: `W_in` の形状 `(N, 1+D_in)` に入力次元が要るが、`D_in` は課題側が決める量であり YAML の構造ハイパーパラメータではない。`ESNConfig` に入れると「設定したのに課題と食い違う」経路を作るため、コンストラクタの keyword 引数にした（仕様の `ESN(config, rng)` 呼び出しはそのまま有効）
  - `bias_scale` は `W_in` の**先頭列（定数入力 1 に対応）だけ**に適用し、残りの入力列は `input_scale` を使う。既定値は `bias_scale = 0.1`。理由: 仕様は `W_in ~ U[-input_scale, input_scale]` とだけ書いており `bias_scale` の作用点が未定義だった。定数入力の寄与が入力の寄与を上回ると tanh が飽和側に張り付くため、入力スケールより小さい値を既定にした（実験で使う値は T4 の YAML で明示する）
  - `ESNConfig` の値の検証は `ESN.__init__` で行い、dataclass 側には `__post_init__` を置かない。理由: T1 の設定 dataclass 群と同じ「純粋なデータ保持」に揃える。YAML 起因の失敗は `ConfigError`、値域の失敗は `ValueError` と発生源が分かれる
  - `state_noise > 0` かつ `rng is None` は `ValueError`。理由: 黙ってノイズ無しで走ると「設定したのに効いていない」実験になる（D-09 と同じ動機）
  - `ESN.run` は `ESN.step` を逐次呼ぶのとビット単位で同一の結果を返す（入力射影を系列全体でまとめて計算する最適化をしていない）。理由: 04 の閉ループで `step` に切り替えた瞬間に軌道が変わる事故を防ぐ。`tests/test_reservoir.py::test_run_equals_repeated_step` で固定した
  - `ESN.W` / `ESN.W_in` は生成後に read-only（`setflags(write=False)`）。理由: 学習は読み出し層だけで行うという分担を機械的に守る
  - `activation` は `"tanh"` 以外を `ValueError`。理由: 未対応の値を黙って tanh として扱わない
  - 設計行列の `first_valid` より手前の行は **NaN** で埋め、`fit_ridge` は非有限値を `ValueError` にする。理由: 0 埋めにすると `t0` の取り違えが「少しずれた学習結果」として静かに通り、D-05 の防衛線をすり抜ける
  - `build_design_matrix` は内部で `FeatureSpec` を `_Layout(bias, input_lags, use_states)` に正規化し、組み立て経路は1本に合流させる。手法ごとの分岐は `_layout_of` の `match` 1か所のみ（受け入れ条件1）
  - 特徴名は `bias`, `u{次元}_lag{ラグ}`, `x{ユニット}`。`bias_column_index(feature_names)` を公開し、`fit_ridge(..., bias_column=...)` に渡す。理由: `bias=False` の設計行列に `D = diag(0,1,...)` を当てると先頭の実特徴だけが無罰則になり、D-03 が静かに別物になる
  - `fit_ridge` / `select_alpha` の `y` は `(T, D_out)` の2次元のみ受理する（1次元は `ValueError`）。理由: 診断層の入力規約（T2）と揃える
  - `select_alpha` の `alphas` に既定値を持たせない（呼び出し側が `config.ridge.alpha_grid` を渡す）。理由: 格子の既定値が config と ridge の2箇所に存在すると D-04 が静かに破れる。同点判定は昇順走査＋`<=` 更新で「大きい alpha が残る」を実現
  - `readout.predict(phi, coefficients)` を公開した。理由: `select_alpha` が内部で使う予測を T4 のランナーが書き写さずに済むようにする（1行だが、書き写すと `first_valid` の扱いが分岐しうる）
  - ESN の既定値（MG 用・パリティ用）は `docs/design.md` §6 に根拠付きで記録した。`ESNConfig` の dataclass 既定値は MG 用の組に一致させてある

### T4: タスク2種 + 実験1-A ランナー + comparison.csv

- **何をするか**
  - `src/rc_basics_lab/tasks/base.py`: `TaskData(frozen)`: `u: FloatArray (T, D_in)`, `y: FloatArray (T, D_out)`, `name: str`, `params: Mapping[str, str]`。`TaskGenerator` Protocol: `(cfg, rng) -> TaskData`
  - `src/rc_basics_lab/tasks/mackey_glass.py`: 上表の値で RK4 積分 → サブサンプル → `horizon` ステップ先予測（既定 1）を目標に整形
  - `src/rc_basics_lab/tasks/delay_parity.py`: ±1 の i.i.d. 入力、`y[t] = Π_{i=delay}^{delay+n_bits-1} u[t-i]`。`n_bits`, `delay` を設定項目化
  - `src/rc_basics_lab/experiment/split.py`: 連続分割。`split` シードストリームで**長めに生成した系列内の開始オフセット**を選ぶ（split シードを実際に配線するため。未使用パラメータを作らない）。1レプリケート内では全手法が同一分割を共有する
  - `src/rc_basics_lab/experiment/runner.py`: (task × method × replicate) を回し、`t0` 整合 → 検証で alpha（と遅延線の `n_lags`）を選択 → テスト評価 → 長形式 `DataFrame` 相当の行を組み立てて CSV へ
  - `experiments/01_what_is_rc/config.yaml`, `experiments/01_what_is_rc/run.py`（`argparse` で `--config`, `--out`。`logging` で進捗）
  - 出力: `results/comparison.csv`（列: `task, method, replicate, seed_reservoir, seed_task, seed_split, alpha, n_lags, rmse, nrmse, nmse, sign_accuracy, n_train, n_val, n_test, t0, wall_time_s`）と `results/meta.json`
  - シード本数は既定 5（YAML `n_replicates`）
- **触るファイル**: `src/rc_basics_lab/tasks/*`, `src/rc_basics_lab/experiment/*`, `experiments/01_what_is_rc/{config.yaml,run.py}`, `tests/test_tasks_*.py`, `tests/test_experiment_fairness.py`
- **受け入れ基準**
  - `tests/test_tasks_mackey_glass.py::test_tau_changes_trajectory` — τ=17 と τ=30 で軌道が異なる
  - `tests/test_tasks_mackey_glass.py::test_is_chaotic_not_periodic` — 生成系列の自己相関が長ラグで減衰する（周期解に落ちていない）
  - `tests/test_tasks_mackey_glass.py::test_horizon_changes_target` — `horizon` 1→5 で目標配列が変わる
  - `tests/test_tasks_parity.py::test_target_is_orthogonal_to_lagged_inputs` — 生成系列で `|corr(y, u[t-k])| < 0.05`（k=0..10）。**解析的失敗保証の数値確認**
  - `tests/test_tasks_parity.py::test_n_bits_and_delay_change_target` — `n_bits`・`delay` それぞれの変更で目標が変わる
  - `tests/test_tasks_parity.py::test_linear_baselines_fail_and_esn_solves_delay_parity` — 縮小設定（N=100, 学習2000点, 3シード）で `NRMSE(linear) >= 0.9`, `NRMSE(delay_line) >= 0.9`, `NRMSE(esn) <= 0.6` かつ `sign_accuracy(esn) >= 0.85`。実行時間10秒以内（**受け入れ条件2の機械的担保**）
  - `tests/test_experiment_fairness.py::test_alpha_grid_is_shared_across_methods` — ランナーが構築する全 (method, task) の alpha 格子が同一オブジェクト値。手法別 alpha 格子キーが YAML にあれば `ConfigError`
  - `tests/test_experiment_fairness.py::test_all_methods_share_identical_rows` — 1レプリケート内で全手法の train/val/test の行 index 集合が完全一致
  - `tests/test_experiment_fairness.py::test_split_seed_changes_boundaries` — `seeds.split` を変えると分割の開始オフセットが変わり、`seeds.reservoir` を変えても変わらない
  - `results/comparison.csv` に MG・パリティ × 3手法 × 5レプリケートの30行が出て、集計で平均±標準偏差が計算できる（受け入れ条件3）
  - 実験全体の実測実行時間が **5分未満**（`meta.json` の `wall_time_s` で確認）
- **想定所要**: L

### T5: 作図層 + README + 設定配線テスト + CI 仕上げ

- **何をするか**
  - `src/rc_basics_lab/plotting/style.py`: `setup_style() -> StyleContext`。候補 CJK フォント（`Hiragino Sans`, `Noto Sans CJK JP`, `IPAexGothic`, `Yu Gothic`）を `matplotlib.font_manager` から探索し、見つかれば設定、無ければ `logger.warning` + `cjk_available=False`。`savefig.dpi=200`, `figure.dpi=100`
  - `src/rc_basics_lab/plotting/labels.py`: `label(ja: str, en: str, *, cjk: bool) -> str`。CJK 不在時は英語ラベルに切り替える（**豆腐文字を出さない**）
  - `src/rc_basics_lab/plotting/figures.py`: `plot_comparison(rows, path)` — タスク別に手法を並べた点+誤差棒（NRMSE=1 の水平基準線を引き「平均予測と同等」を明示）。`plot_state_space(pca_state, pca_input, path)` — 左: PC1-PC2 散布図の並置、右: 累積寄与率曲線（`n_components_95` を注記）
  - `experiments/01_what_is_rc/run.py` に作図まで含める（1コマンドで CSV + PNG2枚 + meta.json）
  - `Makefile` に `figures-01:` ターゲット（`ci` の構成は変えない）
  - `README.md` を書き換え: 3コマンド再現手順（`uv sync --locked` / `uv run pytest -q` / `uv run python experiments/01_what_is_rc/run.py --config experiments/01_what_is_rc/config.yaml`）、Apache-2.0、成果物の説明
  - `LICENSE`（Apache-2.0）追加
  - `tests/test_config_wiring.py`: **本サイクル最重要の有効性テスト**。`config.yaml` の全数値・列挙パラメータを列挙した `parametrize` で、各パラメータを別値に差し替えると縮小パイプラインの出力（`comparison.csv` 相当の行の指紋）が変わることを確認する。新しいパラメータを足したのにここに追加しなければ、`test_all_config_fields_are_covered` が落ちる
  - `docs/design.md` に実測結果（各手法の NRMSE、PCA の `n_components_95`）を追記
- **触るファイル**: `src/rc_basics_lab/plotting/*`, `experiments/01_what_is_rc/run.py`, `Makefile`, `README.md`, `LICENSE`, `tests/test_plotting_style.py`, `tests/test_config_wiring.py`, `docs/design.md`
- **受け入れ基準**
  - `tests/test_plotting_style.py::test_labels_fall_back_to_english_without_cjk_font` — CJK フォント探索を monkeypatch で空にすると `label()` が英語を返し、例外を出さない
  - `tests/test_plotting_style.py::test_savefig_dpi_is_retina` — `setup_style()` 後に `rcParams["savefig.dpi"] >= 200`
  - `tests/test_config_wiring.py::test_each_parameter_changes_output` — 全パラメータについて緑
  - `tests/test_config_wiring.py::test_all_config_fields_are_covered` — `ExperimentConfig` の全フィールドが上記 parametrize に登場する（配線漏れの構造的検出）
  - 1コマンドで `results/{comparison.csv, fig_comparison.png, fig_state_space.png, meta.json}` が生成され、PNG が 200 dpi 以上
  - `n_components_95(reservoir_states) > n_components_95(delay_embedded_input)` が `comparison` 実行時に数値として記録される（受け入れ条件4）
  - `make ci` 緑、GitHub Actions 緑
- **想定所要**: M

> L タスクは T3・T4 の2本（分割不足の警告閾値 3本には未達）。

## 5. 評価軸（Check フェーズに渡す）

### 機能観点
- 3手法が `FeatureSpec` の差だけで切り替わる: `tests/test_design.py::test_three_specs_share_one_api` + `runner.py` に手法ごとの `if` 分岐が学習・評価パスに存在しないこと（レビューで目視）
- 遅延パリティで線形・遅延線が失敗し ESN が解ける: `test_linear_baselines_fail_and_esn_solves_delay_parity`
- PCA が状態空間の高次元性を示す: `n_components_95` の比較値が `comparison` 実行ログと `docs/design.md` に残る
- 1コマンド再生成: クリーンな `results/` から実行し4ファイルが出る

### 性能観点
- 実験全体の実測 wall time < **300 秒**（`meta.json` に記録。超えたら系列長かレプリケート数を YAML で下げる）
- `uv run pytest -q` 全体 < **60 秒**（Stop フックの既定タイムアウト120秒に対する余裕）
- 図生成のメモリピークがローカルで問題にならないこと（N=200・T=8000 なので実質自明。計測不要）

### 安全性観点
- `diagnostics` が `reservoir` に依存しないこと（02〜05 の移植性が壊れる = 連載の設計目標が崩れる）
- `first_valid` / `t0` の整合が崩れると**全ベースライン比較が無効になる**。`test_all_methods_share_identical_rows` が最終防衛線
- alpha 格子の共通化が崩れると結論が逆転しうる（要件書 設計判断3、esn-vla-uq §11 の教訓）
- 機密情報なし・外部通信なし・データ再配布なし（05 まで発生しない）

### テスト観点
- 新規: `tests/test_{config,seeds,metrics,meta,diagnostics_base,diagnostics_state_space,reservoir,design,ridge,tasks_mackey_glass,tasks_parity,experiment_fairness,plotting_style,config_wiring}.py`
- 既存: `tests/test_main.py` は不明点 Q2 の回答次第（削除 or CLI エントリのテストへ置換）
- 数値テストは全て固定シード。`pytest.approx` の許容値を明示し、「たまたま通る」閾値を作らない

### 有効性観点（パラメータ配線の担保 — 本サイクル必須）
本サイクルは YAML に十数個のパラメータを新設するため、**「設定したのに効いていない」が最大の失敗モード**である。以下を受け入れ基準に含める:

1. `test_config_wiring.py::test_each_parameter_changes_output` — 各パラメータの値変更がパイプライン出力を変える（parametrize で網羅）
2. `test_config_wiring.py::test_all_config_fields_are_covered` — 上記に未登録のフィールドがあれば失敗（将来のパラメータ追加時も自動で強制される）
3. `test_unknown_key_raises` — YAML のタイプミスが黙って無視されない
4. 個別の値変更テスト: `test_leak_rate_changes_state_autocorrelation`, `test_n_lags_changes_column_count`, `test_alpha_changes_coefficient_norm`, `test_tau_changes_trajectory`, `test_horizon_changes_target`, `test_n_bits_and_delay_change_target`, `test_x0_and_state_noise_are_wired`, `test_split_seed_changes_boundaries`

## 6. 意図的な決定（`.claude/decisions.yaml` に追記）

```yaml
- id: D-01
  rule: "診断関数の署名は f(X, u=None, y=None, *, ctx: DiagnosticContext|None) に固定し、拡張は DiagnosticContext への既定値つきフィールド追加のみで行う"
  rationale: "要件_01 設計判断1。02 の ESP は2軌道、03 の IPC はサロゲート用シード、04 は dt を要するが、これらを位置引数にすると診断ごとに署名が割れて memristor-rc-lab への移植性が失われる。ctx.companion_states / ctx.dt / ctx.seed に逃がせば 02〜05 の全診断が署名変更なしで乗る"
  guard_test: "tests/test_diagnostics_base.py::test_dummy_diagnostic_conforms_to_protocol"

- id: D-02
  rule: "誤差指標は NRMSE = RMSE / std(y_true) (ddof=0) に統一する。範囲正規化 (max-min) は使わない"
  rationale: "要件_01 未確定3。NRMSE=1 が『平均予測と同等』を意味するため、遅延パリティで線形手法が失敗したことを解釈可能な形で示せる (範囲正規化では 1 に意味が無い)。CSV には nmse を併記し 03 の先行研究比較に備える"
  guard_test: "tests/test_metrics.py::test_nrmse_of_mean_predictor_is_one"

- id: D-03
  rule: "リッジ回帰はバイアス列を正則化しない (罰則行列 D = diag(0,1,1,...))"
  rationale: "バイアスを縮めると目標の平均がずれ、NRMSE=1 という基準線の意味が壊れる。D-02 の解釈性が前提とする性質であり、実装の簡略化で潰されやすい"
  guard_test: "tests/test_ridge.py::test_bias_column_is_not_penalized"

- id: D-04
  rule: "ridge alpha の探索格子は YAML の単一キー ridge.alpha_grid に集約し、全手法・全タスクが同一格子を読む。手法別の alpha 格子キーは ConfigError とする。alpha は (手法, タスク, レプリケート) ごとに検証分割で最良化する"
  rationale: "要件_01 設計判断3。esn-vla-uq §11 の教訓 —— alpha を固定すると手法比較で逆の結論が出うる。『格子は揃え、条件ごとに最良化する』が連載全体の規律であり、これが崩れると 01〜05 の全比較が無効になる"
  guard_test: "tests/test_experiment_fairness.py::test_alpha_grid_is_shared_across_methods"

- id: D-05
  rule: "1レプリケート内では全手法が完全に同一の行 index で学習・評価する。t0 = max(全手法の first_valid, washout) を単一の基準点にする"
  rationale: "遅延線は先頭 n_lags 行を失うため、素直に実装すると手法ごとに評価行が変わり比較が無効になる。要件_01 受け入れ条件1 の『同一分割』の実体はこれ"
  guard_test: "tests/test_experiment_fairness.py::test_all_methods_share_identical_rows"

- id: D-06
  rule: "乱数はリザバー生成 / タスク生成 / 分割の3ストリームに SeedSequence の spawn_key で分離し、1ストリームの変更が他ストリームの乱数列に影響しない"
  rationale: "要件_01 設計判断4。単一 Generator を共有すると『リザバーだけ変えたときの分散』が測れず、受け入れ条件3 の平均±標準偏差の意味が曖昧になる。split ストリームは系列内の開始オフセット選択に実配線し、未使用パラメータを作らない"
  guard_test: "tests/test_seeds.py::test_streams_are_independent"

- id: D-07
  rule: "遅延パリティの既定は n_bits=2, delay=1、入力は ±1 の i.i.d.、目標は該当ラグの積 (y[t] = u[t-1]*u[t-2])"
  rationale: "要件_01 未確定2・受け入れ条件2。この符号化では目標が {1, u[t-k]} の張る線形空間に厳密に直交するため、線形回帰と遅延線の失敗が finite-sample の偶然ではなく解析的に保証される。遅延線も失敗する点が『最も安い対照は遅延線』(記事01 §6) に対する最も鮮明な実演になる。n_bits/delay は掃引軸として YAML 化し、n_bits=3, delay=2 を副条件として併記する"
  guard_test: "tests/test_tasks_parity.py::test_target_is_orthogonal_to_lagged_inputs"

- id: D-08
  rule: "ESN の構造ハイパーパラメータ (N, rho, leak, input_scale, density) は文献既定値に固定し、検証分割で調整しない。検証で選ぶのは alpha (全手法) と遅延線の n_lags のみ"
  rationale: "要件_01 スコープ『網羅探索はやらない・探索予算を揃えることを優先』。結果として探索予算は対照 (遅延線: alpha x n_lags) の方が ESN (alpha のみ) より大きく、非対称性は ESN に不利な側へ倒れる。Goudarzi 2014 が ESN 側だけ最適化して逆の非対称性を作った (survey Q2) のと反対の立場を明示的に取る"
  guard_test: "tests/test_experiment_fairness.py::test_esn_hyperparameters_are_not_validation_selected"

- id: D-09
  rule: "実験設定 YAML の未知キーは ConfigError で即座に失敗させる (寛容な読み飛ばしをしない)"
  rationale: "本サイクルで十数個のパラメータを新設するため、キーのタイプミスが黙って無視されると『設定したのに効いていない』が発生し、実験結果が丸ごと無意味になる。この失敗は LLM レビューでは確率的にしか見つからない"
  guard_test: "tests/test_config.py::test_unknown_key_raises"

- id: D-10
  rule: "日本語フォント対応に新規依存 (japanize-matplotlib 等) を追加せず、実行時に CJK フォントを探索し、見つからない場合はラベルを英語にフォールバックする"
  rationale: "要件_01 制約『依存は最小構成を維持』と『図は日本語フォント対応』の両立。フォント設定だけをフォールバックすると CI 上で豆腐文字の図が生成されるため、ラベル文字列ごと切り替える。記事用の図はローカル (CJK あり) で生成し、CI は生成の成否のみ検証する"
  guard_test: "tests/test_plotting_style.py::test_labels_fall_back_to_english_without_cjk_font"
```

## 7. 受け入れ条件 → タスク対応表

| # | 要件書の受け入れ条件 | 主担当タスク | 検証方法 |
|---|---|---|---|
| 1 | 3ベースラインが同一 API で切り替わり、同一分割・同一 alpha 格子で比較できる | **T3** (API) / T4 (ランナー) | `test_three_specs_share_one_api`, `test_all_methods_share_identical_rows`, `test_alpha_grid_is_shared_across_methods` |
| 2 | 遅延パリティで線形が解けず ESN が解けることを数値で示せる | **T4** | `test_target_is_orthogonal_to_lagged_inputs`, `test_linear_baselines_fail_and_esn_solves_delay_parity`, `comparison.csv` |
| 3 | MG 予測で3手法の誤差 + シード5本以上の平均±標準偏差 | **T4** | `comparison.csv` に30行、`n_replicates: 5`、`fig_comparison.png` の誤差棒 |
| 4 | リザバー状態の PCA が入力空間より高次元に広がる | **T2** (診断) / T5 (図) | `state_pca` の `n_components_95` 比較値 + `fig_state_space.png` |
| 5 | 図2枚が1コマンドで再生成、retina 解像度 | **T5** | `run.py` 1発で4ファイル、`test_savefig_dpi_is_retina` |
| 6 | 診断層インターフェース `f(X,u,y)` の定義 + ダミー実装 + テスト | **T2** | `test_dummy_diagnostic_conforms_to_protocol`, `test_diagnostic_accepts_external_state_series`, `test_diagnostics_package_does_not_import_reservoir` |
| 7 | pytest green + 最小 CI + README (3コマンド再現) | **T1** (基盤) / **T5** (README・仕上げ) | `make ci` 緑、GitHub Actions 緑、README の3コマンドを実行して再現 |

## 8. 想定リスク（発生したら止まって相談）

1. **遅延パリティで ESN が解けない**（N・ρ・leak・input_scale が不足し `NRMSE(esn) > 0.6` に留まる）。受け入れ条件2 が満たせない。**このとき閾値を緩めて「示せた」ことにするのは禁止**。ESN の構造ハイパーパラメータを検証で選び始めるのも D-08 違反。取りうる手は「n_bits=2, delay=1 よりさらに易しい条件（delay=0、つまり `u[t]·u[t-1]`）に落とす」か「N を 200→500 に上げる」のどちらかで、いずれも設計判断なので相談する
2. **mypy strict + `disallow_any_explicit` が scipy / matplotlib の型情報と衝突**し、`# type: ignore` を撒かないと通らない状況になる。`[[tool.mypy.overrides]]` の `ignore_missing_imports` で解決できない衝突が出た時点で相談する（CLAUDE.md の「理由なく `# type: ignore` を使わない」に抵触するため、握り潰さない）
3. **実験の実行時間が5分制約を超える**。特に MG の RK4 積分（h=0.1 で 80,000 ステップ）と ESN の逐次ループが Python ループだと遅い。ベクトル化できない部分の最適化が必要になった場合、系列長を削るか C 拡張的な手段に踏み込むかは設計判断なので相談する
