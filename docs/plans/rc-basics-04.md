# 仕様: rc-basics-04 —— カオス時系列の自由走行予測 (+ 03 からの技術的負債8件)

*要件書: `docs/要件_rc-basics-04.md` / 前サイクル仕様: `docs/plans/rc-basics-03b.md` (§「04 への申し送り」)*
*前提: サイクル1・2a・2b・3a・3b-1・3b-2 完了 (`make ci` 緑 / D-01〜D-40 が全件 guard_test つき)*

> **本書の位置づけ**: 要件書の実装と、`rc-basics-03b.md` の「04 への申し送り」
> (reviewer-architecture 集約、8件・着手順) を1本の仕様にまとめたもの。
> **負債8件は要件書の実装と混ぜず、前段の独立タスク (T1〜T3) に置く**。理由は §3.3。
> 規律に従い、本書には**テスト件数を書かない** (01〜03 で3回ずれた)。

---

## 1. ゴール

学習済み read-out の出力を入力へ戻す自走 (closed-loop) 系を、**外部生成の状態系列生成器でも動く**形で
実装し、Lorenz アトラクタの再現・有効予測時間 (Lyapunov 時間正規化)・自走の3態マップを
1コマンドで再生成できる CSV・図5枚として出す。その前段として 03 から持ち越した設計負債8件を片付ける。

---

## 2. 現状認識

### 2.1 関連箇所 (すべて実ファイルで確認済み)

| 用途 | 場所 | 04 での使い方 |
|---|---|---|
| 閉ループ用の1ステップ更新 | `reservoir/esn.py:209-231` | `ESN.step(x, u, rng=None)`。**自走は `run` ではなく `step`** |
| 教師強制の状態生成 | `reservoir/esn.py:233-262` | `ESN.run(u, x0, rng)`。入力系列が既知の区間専用 |
| 状態ノイズ | `reservoir/esn.py:182-199` | `state_noise>0` かつ `rng is None` は `ValueError` |
| 伝播器 (決定的でなければならない) | `experiment/esp.py:167-193` | `esn_propagator`。**docstring に 04 への指示がある** |
| ESN 構成 / 参照軌道 | `experiment/esp.py:239-268` / `294-339` | `build_esn_config` / `simulate_reference_trajectory` |
| 比較軌道ループ (ノイズ未対応) | `experiment/esp.py:380-432` | `simulate_condition`。`state_noise>0` で D-14 が崩れる |
| 01 の公平性機構 | `experiment/runner.py:161-191` / `295-333` | 4-A は 3-C (D-31) と同じ形で `run_task` を通す |
| 01 の行 dataclass | `experiment/runner.py:93-117` | `ResultRow` / `CSV_COLUMNS` を 4-A がそのまま使う |
| 外部生成 X → 容量行 | `experiment/capacity.py:572-771` | **4-D はこの3段をそのまま使う** |
| 確保上限の検査 | `experiment/capacity.py:156-227` | `validate_n_units_bound` / `validate_state_matrix_bounds` を再利用 |
| 設計行列 / 3手法 | `readout/design.py:29-52`, `105-126` | 4-A の対照は `FeatureSpec` 3種だけで表現する |
| リッジ / alpha 選択 | `readout/ridge.py:127-207` | 自走は学習済み係数を `predict` に流すだけ |
| MG 生成器 (**再実装しない**) | `tasks/mackey_glass.py:78-160` | `tasks/chaotic.py` は Lorenz を足すだけ |
| 診断の署名と ctx | `diagnostics/base.py:71-89` | **`DiagnosticContext.dt` は 04 のために既に在る**。足さない (D-01) |
| 04 用のモジュール名の予約 | `diagnostics/__init__.py:23-26` | `lyapunov` が予約済み |
| 設定ローダ | `config.py:695-720` | `load_config_as`。**T1 で package 化するが経路は変えない** |
| 実験レジストリ | `main.py:96-121` | `"04"` を1行。`out_dir` は一意でなければならない |
| config.py の行数照合 | `tests/test_design_doc.py:819-833` | **package 化した瞬間に赤くなる** |

### 2.2 既存の慣習で守るもの (3個)

1. **CSV の列順は行 dataclass の宣言順が単一の真実**
2. **設定 dataclass は純データ、値域検証は使う側**。未知キーは `ConfigError` (D-09)
3. **図のラベルは `labels.label(ja, en, cjk=...)` を通す** (D-10)。図は成果物 CSV の行だけを読む

### 2.3 影響範囲

- **追加**: `config/` (package 化後) / `tasks/chaotic.py` / `readout/autoregressive.py` /
  `diagnostics/lyapunov.py` / `experiment/{freerun,stability,freerun_pipeline}.py` /
  `plotting/figures_freerun.py` / `experiments/04_chaotic_freerun/` / テスト群 / ADR 2本
- **変更**: `experiment/esp.py` / `diagnostics/` (整理4件) / 各 `__init__.py` / `main.py` /
  `Makefile` / `docs/design.md` (§12 新設) / `README.md` / `.claude/decisions.yaml`
- **変更しない**: `results/` (01・02・03 はバイト不変。`capacity.csv` は `wall_time_*` 4列を除く) /
  `seeds.py` の既存4ストリームの `spawn_key` / `build_tasks` / `ExperimentConfig` (D-13) /
  `DiagnosticContext` のフィールド (D-01)

### 2.4 3b-2 完了時点で判明している事実 (**着手前に一読**)

1. **`config.py` は着手条件に到達済み**: 非空 **615行** / 総 771行。
   `test_config_py_line_count_in_the_design_doc_is_current` が `nonempty > 600` を assert するので
   **分割した瞬間に赤くなる**。テスト側の書き換えを T1 に含める
2. **ノイズを 02 経路に入れると2種類の壊れ方をする**:
   (a) `esn_propagator` は `esn.step` を rng なしで呼ぶ。**単に rng を渡すのは誤り** —
   `conditional_lyapunov` は摂動の成長率を測るので伝播器は決定的でなければならない。
   (b) `simulate_condition` の比較軌道は `state_noise>0` で「初期状態もノイズ実現値も違う」軌道になり
   **D-14 の3ストリーム分離に4本目が混ざる**。乱数消費が参照軌道に依存するため**評価順にも依存する**
3. **`diagnostics/` は 3a 完了以降1行も変更されていない**。整理4件は 3a の設計に対する未着手の負債
4. **`import rc_basics_lab.plotting` を最初に行うと `ImportError`** (循環 import)
5. **`import rc_basics_lab.diagnostics.ipc as m` はモジュールではなく関数を返す**。
   3a のレビューで実際に踏み、**変異試験が偽の緑になった**
6. **性能は 03 の枠内で余裕がある** (325〜371秒 / 予算900秒)。ただし**自走は逐次計算で
   ベクトル化できない**ため、04 の予算は「条件数 × 自走長」で守る

---

## 3. 前提・制約

### 3.1 ハード制約 (絶対に変えない)

- **D-01 / D-15**: 診断の署名は `f(X, u=None, y=None, *, ctx)`。`DiagnosticContext` に**足さない**
- **D-12 / D-23**: `diagnostics/` は `config` / `reservoir` を推移的にも import しない
- **D-13**: `ExperimentConfig` に 04 のフィールドを1個も足さない。`Chaos04Config` を新設
- **D-04 / D-05 / D-08 / D-31**: 4-A は 01 の `run_task` 経路をそのまま通す
- **D-06 / D-14**: 既存4ストリームの `spawn_key` を動かさない
- **D-02**: 誤差指標は NRMSE 主・NMSE 併記
- **D-34 の規律**: 確保軸には絶対上限を置き**確保より前に落とす**。**軸は列挙して1本ずつ確認する**
- 01・02・03 の成果物は**バイト不変**
- 型: Python 3.12+ / `Any` 禁止 / `print()` 禁止 / `# type: ignore` を理由なく使わない
- 新規依存を増やさない

### 3.2 ソフト制約 (理由があれば変えてよい)

- 出力先は **`results/04_chaotic_freerun/`** (要件書の `results/` 直下から変更。D-51)
- MG は `tasks/mackey_glass.py` を再利用し再実装しない
- 自走の read-out は `ReservoirSpec` を既定。多項式読み出しは v0.1 では入れない
- NVAR / ハイブリッド / 高次元カオス / 実データは実装しない
- 図をまたいだ絶対値比較はしない

### 3.3 技術的負債8件の配置 (**要件書の実装と混ぜない**)

| # | 負債 | 配置 | 混ぜない理由 |
|---|---|---|---|
| 1 | `config.py` の package 化 | **T1** | 04 の設定を足す前に割らないと「移動だけか」を判定できない |
| 2a | `esn_propagator` の決定性 | **T2** | 4-C がノイズを掃引軸に入れる前に決めないと `ValueError` を黙らせる形で潰される |
| 2b | 比較軌道とノイズ実現値 | **T2** | 同上 |
| 3a-3d | `diagnostics/` の整理4件 | **T3** | 実験層と混ぜると 03 成果物のバイト不変検査が切り分けられない |
| 4a | `diagnostics.ipc` の名前隠蔽 | **T2** | 変異試験が偽の緑になる経路。実験を書き始める前に消す |
| 4b | `experiment` ⇄ `plotting` の循環 | **T2** | 04 で図を足すと辺が増える。増やす前に直す |

---

## 4. タスク分解

### T1: `config.py` の package 化 (想定所要: **M**)

`config.py` を `config/` package に割る。分割単位は**実験サイクル**:
`_common.py` / `experiment01.py` / `esp02.py` / `capacity03.py` / `__init__.py`。
**`Chaos04Config` はここでは作らない** (置き場所 `chaos04.py` を決めるだけ)。

**受け入れ基準**

- `from rc_basics_lab.config import <公開シンボル>` が package 化**前と同一**に通る (D-49)。
  公開シンボル一覧を機械的に取り差分0を要求する
- **移動以外の論理変更が0**であることを `results/` 全ファイルの SHA-256 一致で示す
- 各モジュールが**非空300行以下**。分割方針と行数を design.md に記録し機械照合する
- `config/` 内の依存は `_common` への一方向のみをテストで固定
- `make ci` 緑

### T2: ノイズ・伝播器の決定 + 公開 API・レイヤ整理 (想定所要: **M**)

> **着手前に architect (`/design`) を1本通す**。ADR は `docs/adr/` に置く。

1. **伝播器は決定的にする** (D-48)。**rng を渡して黙らせる実装は禁止**
2. **比較軌道とノイズ** (D-47): `simulate_condition` は `state_noise>0` を受理しない
3. **`diagnostics.ipc` の名前隠蔽** (負債4a): 属性アクセスでモジュールが返るようにする
4. **循環 import の解消** (負債4b): `import rc_basics_lab.plotting` 単独が通る

**受け入れ基準**

- `test_plotting_can_be_imported_first` — **着手前に赤くなることを実測**してから直す
- `test_diagnostics_ipc_module_resolves_to_a_module`
- D-48 / D-47 の guard_test。**両方とも変異注入で落ちることを実測**して実装メモに残す
- 03 の 3-B' の `capacity.csv` が実測時間4列を除いて不変
- ADR 1本

### T3: `diagnostics/` の整理4件 (想定所要: **L**)

> **着手前に architect (`/design`) を1本通す**。D-24 / D-28 / D-33 / D-34 の**改訂**を伴う。

1. `orthonormal_basis` の引数設計 (F-03-1-006)。`(input_distribution, basis)` を**1つの値**に
2. `ipc.max_targets` の単位の分離。**D-34 の rule を改訂**
3. `RowAlignment` の切り出しと引数集約 (D-24 の単一基準点)
4. `chunk_size` の性能軸とメモリ軸の分離。**D-33 の rule を改訂**

**受け入れ基準**

- `results/03_capacity/` が**バイト不変**。これが「純粋な整理である」ことの唯一の証明
- 改訂後の guard_test が**1本ずつ変異注入で赤くなる**ことを実測し実装メモに残す
- **新設の抽象が空虚でない**: no-op や旧実装に差し替えると落ちるテストがそれぞれ1件以上
  (D-33 で「no-op に差し替えても1件も落ちない安全機構」を実際に作った前例がある)
- `diagnostics/` は依然 `config` / `reservoir` を import しない
- `check_decisions.py` 緑

### T4: 04 の基盤 —— カオス生成器・自走・設定層・実験4-A (想定所要: **L**)

1. **`tasks/chaotic.py`** (D-41): Lorenz (10, 28, 8/3) を RK4 で積分。**MG は委譲**。
   標準化係数は**訓練区間から推定した1組**を全区間で使う
2. **`diagnostics/lyapunov.py`** (D-42): Benettin 法。`ctx.dt` で正規化。**正本は数値推定**
3. **`readout/autoregressive.py`** (D-44 / D-50): **ESN を知らない**。状態更新器を Protocol で受ける
4. **設定層** `config/chaos04.py` の `Chaos04Config`
5. **実験4-A**: **01 の `run_task` を再利用** (D-31 と同じ形)

**受け入れ基準**

- `test_lorenz_matches_reference_trajectory` (D-41): 独立実装 (`solve_ivp`, rtol/atol 1e-10) と
  相対 1e-6 以内。**自分の実装のループ・定数を参照しない**
- `test_estimated_lyapunov_matches_the_literature_value` (D-42): 0.9056 の ±5% 以内
- `test_free_run_works_with_an_external_state_generator` (D-50、**受け入れ条件7**):
  ESN を使わない解析的な線形写像で自走が回り、閉形式の予測値と一致する
- `test_free_run_uses_the_teacher_forced_coefficients` (D-44)
- 設定の全葉にチャネル割り当て。**セクション固有の葉は scope 検査つき**
- **Δt の較正**: `sample_interval ∈ {5, 10, 25}` を測り**受け入れ条件1と3の両方が成立する値**を選ぶ。
  **落選した値の実測も design.md §12 に残す**
- 確保軸の 1・2・3・8 を実装し確保より前に検査する

### T5: 実験 4-B / 4-C / 4-D と図5枚・記録 (想定所要: **L**)

1. **4-B**: 自走。シード10本以上。有効予測時間 (D-43、**Lyapunov 時間で正規化**、
   打ち切りフラグを立てる)。長時間統計 (D-46、**2本**の指標)
2. **4-C**: 3態分類 (D-45)。**純関数 + 数値基準**で図から決めない
3. **4-D**: 4-C と**同じ states** に MC / IPC。03 の接ぎ目をそのまま使う
4. **成果物**: `freerun.csv` / `stability.csv` / `onestep.csv` / 図5枚 / `meta.json`
5. **記録**: `docs/design.md` §12 / `README.md` / `.claude/decisions.yaml` (D-41〜D-51)。
   design.md の表は**成果物から機械照合**する

---

## 5. 評価軸 (Check フェーズに渡す)

### 性能観点 (**区間ごとの予算**)

| 区間 | 予算 |
|---|---|
| `make figures-04` 合計 | **< 900 秒** |
| 真の軌道の生成 合計 | **< 60 秒** / 内訳を `meta.json` に出す |
| 4-A | < 120 秒 |
| 4-B | < 240 秒 |
| 4-C | < 300 秒 |
| 4-D | < 150 秒 |
| 図5枚 | < 20 秒 |
| 04 が追加する pytest | < 60 秒 |
| ピークメモリ | **< 4 GB** |
| `results/04_chaotic_freerun/*.csv` 合計 | **< 5 MB** |

**予算超過時に許可される調整**: `stability.n_replicates` を落とすことだけ。
格子・自走長・`n_steps` は動かさない。それでも収まらないなら**実装者判断で削らず止まる**。

### 禁止する構造

1. **自走のたびに read-out を学習し直す** (D-44)
2. **教師強制と自走で別の係数を使う**
3. **(ρ, leak, noise) ごとに真の軌道を積分し直す**
4. **条件ごとに ESN を2回作る** (4-C の分類用と 4-D の容量用)
5. **有効予測時間を生のステップ数だけで報告する**
6. **3態分類を図や目視で決める**
7. **図が診断・実験を走らせる**
8. **自走を `ESN.run` で書く** (自走は `ESN.step` の逐次ループ)

### 安全性観点 (**確保軸を列挙して1本ずつ確認する**)

| # | 軸 | 膨らむもの | 上限の置き方 | 検査位置 |
|---|---|---|---|---|
| 1 | `lorenz.length × sample_interval + burn_in` | 積分ステップ数 | 上書き不能な定数 | 生成前 |
| 2 | `lorenz.length ×` 状態次元 | 真の軌道の配列 | 同上 | 同上 |
| 3 | `freerun.free_run_steps × n_units` | 自走の状態行列 | **既存 `validate_state_matrix_bounds` を再利用** | 自走の入口 |
| 4 | `freerun.stats_steps` | 長時間統計の系列 | 上書き不能な定数 | 4-B の入口 |
| 5 | 条件数 (格子の積 × レプリケート) | 時間 | 上書き不能な定数 | 掃引の入口 |
| 6 | 位相図に載せる点数 | 描画時間・PNG サイズ | 間引きの上限 | 図の入口 |
| 7 | ビン数 / FFT 長 | ヒストグラム・スペクトル | **`stats_steps` に従属させ独立軸にしない** | 4-B の入口 |
| 8 | 4-D の `ipc.max_delay_by_degree` 等 | 目標数 | **既存 D-34 の4段を再利用。新しい上限を作らない** | `_validate_config` |

- 01・02・03 の成果物がバイト不変 / `spawn_key` を動かさない
- **T1・T2・T3 の各タスク単体で `results/` が不変**であること (整理と実験が混ざっていない証拠)

### 有効性観点 (**必須**)

- `Chaos04Config` の**全葉**にチャネル割り当て。委譲先と過不足なく一致
- **セクション固有の葉は scope 検査つき**
- `freerun.valid_time_threshold` を変えると `valid_time_lyapunov` が動く (D-43 の配線の実体)
- `lorenz.sample_interval` を変えると **Lyapunov 時間正規化の分母が動く**
- `stability.state_noise_grid` を変えると3態マップが動く (**受け入れ条件4 の核心**)
- `esn.state_noise` が学習時の状態に実際に効いている

---

## 6. 意図的な決定

> 採番は **D-41 以降**。**T3 は新規決定ではなく D-24 / D-28 / D-33 / D-34 の rule 改訂**であり、
> 経緯を rationale に残す。

- **D-41** Lorenz のパラメータと積分法。**標準化係数は訓練区間から推定した1組を全区間で使う**
  (自走中に再推定すると「当たっているように見える」壊れ方をし図でも指標でも検出できない)
- **D-42** λ_max は**数値推定を正本**とし文献値 (0.9056) は照合にのみ使う
- **D-43** 有効予測時間は **Lyapunov 時間で正規化**。**打ち切りを無かったことにしない**
- **D-44** 自走は教師強制で学習した係数をそのまま使う
- **D-45** 3態は**純関数 + 数値基準**で分類。排他かつ網羅
- **D-46** アトラクタ再現は**2本**の指標で定量化。視覚評価は結論に使わない
- **D-47** `state_noise>0` では比較軌道経路を使わない。**5本目のストリームを新設しない**
- **D-48** 伝播器は決定的でなければならない。**rng を渡して `ValueError` を黙らせない**
- **D-49** package 化しても公開シンボルの import 経路と `__all__` を変えない。依存は一方向のみ
- **D-50** `readout/autoregressive.py` は `reservoir` を import しない。Protocol で受ける
- **D-51** 成果物は `results/04_chaotic_freerun/` に出す

---

## 7. 想定リスク (起きたら止まって相談)

1. **どの条件でも自走がアトラクタを再現しない**。原因は (a) Δt (b) 学習長 / N (c) ノイズ量
   (d) 標準化のずれ で対処が正反対。**「当たるまで ESN の構造 HP を回す」は D-08 違反**
2. **package 化で 01・02・03 の成果物がバイト単位で変わる**。移動以外の変更が混ざった証拠
3. **T3 の整理で guard が空虚になる** (3a で空虚なガードを7件作った実績がある)。
   変異注入で1本ずつ確認し、1本でも落とせないものが出たら止まる

---

## 8. 確定事項 (ユーザー承認済み・2026-08-19)

| 問 | 決定 |
|---|---|
| **分割** | **3分割**。**04a = T1+T2+T3** (負債8件) / **04b-1 = T4** (基盤・自走・4-A) / **04b-2 = T5** (4-B/C/D・図5枚・記録) |
| **Mackey-Glass** | **Lorenz 主 + MG は 4-A / 4-B のみ**。**4-C / 4-D は Lorenz だけ** (2系で回すと条件数が2倍になり900秒予算を割る) |
| **ノイズのストリーム** | **現状維持** (reservoir ストリームの続き) + `simulate_condition` を `state_noise>0` で塞ぐ (**D-47**)。5本目を新設しない |

### 分割後のタスク配分

| サイクル | タスク | L | 主な成果物 | 終了条件 |
|---|---|---|---|---|
| **04a** | T1 (M) / T2 (M) / T3 (L) | 1 | `config/` package / ADR 2本 / `diagnostics/` 整理 / D-47・D-48・D-49 + D-24/28/33/34 改訂 | **負債8件が閉じ、`results/` が全件バイト不変** |
| **04b-1** | T4 (L) | 1 | `tasks/chaotic.py` / `diagnostics/lyapunov.py` / `readout/autoregressive.py` / `Chaos04Config` / `onestep.csv` | **外部生成器で自走が動き** (受け入れ条件7)、4-A の CSV が出る |
| **04b-2** | T5 (L) | 1 | `freerun.csv` / `stability.csv` / 図5枚 / design.md §12 | 受け入れ条件1〜6 |

> 3分割の根拠: 04b が1本だと **1,700行超**になり 3a (3,254行・4ラウンド) の失敗パターンを再現する。
> **負債と実験が1度も混ざらない**ので、`results/` のバイト不変検査が
> 「整理のせいか実験のせいか」を常に切り分けられる。

---

## 8-B. 分割の判断材料 (参考)

**L タスクが3本 (T3 / T4 / T5)。1サイクルで回すことは推奨しない。**

| 案 | 割り方 | diff の見込み | 評価 |
|---|---|---|---|
| A (2分割) | 04a = T1+T2+T3 / 04b = T4+T5 | 04b **1,700行超** | **非推奨**。3a の失敗パターンを再現する |
| **B (3分割)** | 04a = T1+T2+T3 / 04b-1 = T4 / 04b-2 = T5 | 400〜600 / 700〜900 / 1,000〜1,300 | **推奨**。3b-1 と同規模 |
| C (4分割) | T3 を単独サイクルに | 各 400〜900 | 次点。T3 は architect ゲート + D の改訂を伴う |
| D (T3 を 05 送り) | 04a = T1+T2 / 04b-1 = T4 / 04b-2 = T5 | 04a 250〜400 | 負債の再先送りと 04 を軽くする利得の取引 |

> **T3 が実験に依存されない**ことが判断材料として重要: 4-D は既存 API をそのまま呼ぶだけなので
> T3 を後ろへ動かしても T4・T5 は書ける。一方 **T1 と T2 は前段に置く必要がある**。

### 質問にせず前提として進めるもの

- 出力先 `results/04_chaotic_freerun/` (D-51)。図は要件書のまま5枚
- Lorenz は (10, 28, 8/3)。λ_max の照合値は 0.9056 (Viswanath 1998)、**正本は数値推定** (D-42)
- Δt の既定は **0.02**。T4 の較正で確定し落選値も記録する
- 有効予測時間の閾値は **NRMSE 比 0.4**。{0.2, 0.3, 0.4, 0.5} の感度表を残す
- ノイズ注入は**状態への加算のみ**。格子は {0, 1e-4, 1e-3, 1e-2}
- シードは **10本以上**。ESN の構造 HP は検証分割で選ばない (D-08)
- alpha 格子は 01 と共有 (D-04)。誤差指標は NRMSE 主・NMSE 併記 (D-02)
- NVAR / ハイブリッド / 高次元カオス / 実データは**実装しない**
- **サーベイに 04 の論点 (自走・有効予測時間・NVAR) の節が無い** (実測: 0件)。
  先行の参照が要るなら T5 で別途調べる。**実装の正しさは D-41〜D-46 の guard_test で担保する**

---

## 9. 受け入れ条件 → タスク対応表

| # | 受け入れ条件 | タスク | 検証手段 |
|---|---|---|---|
| 1 | 自走が蝶形アトラクタを再現する図 | T5 | **図ではなく** D-46 の距離指標 |
| 2 | 有効予測時間が Lyapunov 正規化・10シード以上 | T4 / T5 | `freerun.csv` の `valid_time_lyapunov` |
| 3 | 教師強制で差が小さく自走で対照が成立しない | T4 / T5 | 両方向を1本で測るテスト |
| 4 | 3態がハイパーパラメータ平面で分類されノイズで領域が変わる | T5 | `stability.csv` + D-45 |
| 5 | 長時間統計が真の系と定量比較されている | T5 | D-46 の2指標が CSV の列にある |
| 6 | 図5枚が1コマンド再生成 + pytest green | T4 / T5 | `make figures-04` / `make ci` |
| 7 | 自走が外部生成の状態系列生成器でも動く | **T4** | D-50 の guard_test |

---

## 10. 実装者への注意 (最も壊れやすい3点)

1. **自走は逐次計算でベクトル化できない**。予算は「条件数 × 自走長」で守るしかない。
   **速くするために自走長を削るのは受け入れ条件2 を壊す**ので条件数の側で調整する
2. **標準化係数は訓練区間から推定した1組を全区間で使う** (D-41)。自走中や評価区間で再推定すると
   「予測が当たっている」ように見える壊れ方をし、**図でも有効予測時間でも検出できない**
3. **`import rc_basics_lab.diagnostics.ipc as m` はモジュールではなく関数を返す** (T2 で解消するまで)。
   3a のレビューで実際に踏み**変異試験が偽の緑になった**

---

## T1 実装時に決めたこと

> 仕様 §4 T1 に書かれていなかった選択と、仕様の記述と食い違った点の記録。
> 次周の reviewer / fixer が読むのはこの節であり、実装のコメントではない。

### 1. `load_config` の置き場所は `_common.py` ではなく `experiment01.py` (**仕様との相違**)

§4 T1 の配置図は `_common.py : ConfigError / _coerce* / _build / load_config_as / load_config`
と書いていたが、`load_config` は `experiment01.py` に置いた。

理由: `load_config` は `ExperimentConfig` を返すので `_common` に置くと
`_common -> experiment01` の辺ができる。一方 `load_config` 本体は
`load_config_as` に委譲するため `experiment01 -> _common` の辺も要る。
**これは循環である**。実測 (変異注入): `_common.py` に
`from rc_basics_lab.config.experiment01 import ExperimentConfig` を足すと
`ImportError: cannot import name 'load_config_as' from partially initialized module`
でテスト収集そのものが落ちる。

受け入れ基準4 (「依存は `_common` への一方向のみ」) を満たすには `_common` を
package 内の**葉**にするしかなく、`load_config` は `ExperimentConfig` と同じ
モジュールに置くのが唯一の解。公開経路 (`from rc_basics_lab.config import load_config`)
は `__init__.py` の再エクスポートで変わらないので D-49 は満たす。

### 2. 許可する辺は 1 本ではなく 2 本 (**仕様との相違**)

§4 T1 は許可辺として `capacity03 -> experiment01` (`Narma10Config.base`) の1本だけを
挙げていたが、**`esp02 -> experiment01` も必要**である
(`WashoutSweepConfig.base: ExperimentConfig`、2-D が 01 の `run_experiment` を
再利用するための内包、D-19)。2本はどちらも「01 をまるごと内包する」同じ形の辺で、
向きは揃っており循環しない。`tests/test_config_package_layout.py` の
`ALLOWED_INTERNAL_EDGES` に、`experiment01 -> _common` を含めて計3本を明示した。

### 3. 「公開シンボルの差分0」の運用上の定義

`__all__` は**差分0** (36 名、リテラルのスナップショットと突き合わせ)。
`dir()` にしか出ない公開名 17 名の扱いは以下に決めた:

- 16 名 (`Mapping` / `Path` / `Protocol` / `SeedConfig` / `SeedStream` / `Sequence` /
  `UnionType` / `cast` / `dataclass` / `dataclasses` / `field` / `get_args` /
  `get_origin` / `get_type_hints` / `np` / `yaml`) は**消える**。すべて `config.py` の
  実装 import の副作用で、`__all__` に無く `import *` にも乗らない。
  これらを `__init__.py` で再エクスポートすると、numpy や yaml が config の API に
  見える状態を新たに作ることになり、整理の目的と逆行する
- `annotations` (`from __future__ import annotations`) だけは `__init__.py` にも書くので残る
- サブモジュール名 3 つ (`experiment01` / `esp02` / `capacity03`) が**増える**。
  package 化で不可避

「16 名が API でなかった」は主張ではなく**実測**にした:
`test_no_module_imported_the_dir_only_names_from_config` がリポジトリ全体の
`from rc_basics_lab.config import ...` を AST で走査し、`__all__` の外の名前の
使用が **0 件**であることを確認する。増減の**両側**を
`test_dir_only_names_changed_exactly_as_recorded` が固定するので、
`__init__.py` が実装 import を公開名に漏らしても落ちる。

### 4. `results/` の複合ハッシュ `47bcd302de7a7366...` は再現できなかった

着手前の `results/` に対し、以下の 6 通りの定義すべてで `47bcd302` にならなかった
(hex 連結 / 改行あり連結 / `shasum` 出力そのまま / ファイル内容の連結 /
シェルの `sort` と Python の `sorted` の両方 / basename ソート)。

判定は**全 24 ファイルの個別 SHA-256 の突き合わせ**で行った (そちらが本来の基準)。
以後のサイクルのために複合ハッシュの定義を1つに固定する:

```
find results -type f | LC_ALL=C sort | xargs shasum -a 256 | awk '{print $1}' \
  | tr -d '\n' | shasum -a 256
```

この定義での T1 着手前 (= 復元後) の値: **`0f7558efc418242f...`**

### 5. 実測: `results/` は 24 ファイル中 24 ファイルが不変

再生成 (`make figures-01` / `figures-02` / `figures-03`) の結果、
**PNG 9 枚はバイト一致**、CSV の差分は**実測時間列だけ**だった:

| ファイル | 差分のある列 |
|---|---|
| `comparison.csv` (30 行) | `wall_time_s` のみ |
| `esp_diagnostics.csv` (369 行) | `wall_time_s` のみ |
| `washout_sensitivity.csv` (180 行) | `wall_time_s` のみ |
| `capacity.csv` (118 行) | `wall_time_state_s` / `wall_time_mc_s` / `wall_time_ipc_s` / `wall_time_s` のみ |
| `narma10.csv` (15 行) | `wall_time_s` のみ |
| `capacity_profile.csv` (21,812 行) / `capacity_length.csv` / `comparison_summary.csv` / `esp_threshold_sensitivity.csv` / `washout_sensitivity_unpadded.csv` | **バイト一致** |

確認後、`results/` は着手前のバイト列へ戻してある (個別 SHA-256 が 24/24 一致)。

### 6. `results/02_esp_and_dynamics/meta.json` の構造差は **T1 とは無関係な既存ドリフト**

再生成した 02 の `meta.json` は `washout_sensitivity.sizes_by_washout` /
`t0_by_washout` が 6 要素から 12 要素になり `task` キーが増えた。§7 リスク2 に従い
先へ進まず原因を特定した:

- コミット済みの `results/02_esp_and_dynamics/meta.json` は `commit: 971a439a` 時点の生成物で、
  その後 `experiment/washout.py` の `to_summary` が `task` キーを持つ形に変わった際に
  **再生成されていない**
- **base-ref (8810d4e、分割前の `config.py`) を worktree に取り出して `figures-02` を回し**、
  分割後の再生成結果と突き合わせたところ、PNG 4 枚はバイト一致、CSV の差は
  `wall_time_s` のみ、`meta.json` の差は `commit` / `timestamp_utc` / `wall_time_s` のみだった。
  **`task` キーは分割前の再生成にも出る** = T1 の変更とは無関係

対応はしていない (`results/` を1バイトも変えない、が T1 の制約のため)。
02 の成果物を再生成する回で自然に解消する。

### 7. 移動だけであることのソースレベルの実測

原本 771 行のうち、移動した本体 (8 ブロック・735 行) は**すべてバイト一致**。
どのブロックにも入らない非空行は **16 行**で、内訳は import 文 15 行
(各モジュールへ配り直した) と、`# --- 実験03 (容量: MC / IPC) の設定群 ---` の
区切りコメント 1 行だけである。**この区切りコメントは削除した** ——
「どこから 03 の設定か」を示す役目はファイル境界 (`capacity03.py`) が果たすため。

### 8. その他の付随変更 (いずれも package 化で赤くなるものへの対応)

- `tests/test_public_api_reexport.py`: `PACKAGE_NAMES` に `"config"` を追加。
  package が 6 個から 7 個に増えたので、既存の完全性検査
  (`test_package_names_matches_automatic_enumeration`) がそのままでは赤くなる。
  docstring の「`config.py` 等の単一モジュールは除く」という例示も `seeds.py` に差し替えた
- `README.md` のリポジトリ構成: `config.py` を `config/` へ
- 1 モジュールあたりの上限 (非空 300 行) の**正本は
  `tests/test_config_package_layout.py::MAX_NONEMPTY_LINES_PER_MODULE`** とし、
  `docs/design.md` §11.5 はその写しとして機械照合する
  (同じ数字を 2 か所に書くと片方だけ更新されて食い違うため)

### 9. `chaos04.py` は作っていない

仕様どおり置き場所を決めただけ。`config/chaos04.py` を足すと
`test_config_package_has_exactly_the_expected_modules` と
`test_config_package_line_counts_in_the_design_doc_are_current` が赤くなるので、
T4 の担当者は design.md §11.5 の表と `EXPECTED_SUBMODULES` を同時に更新すること。
