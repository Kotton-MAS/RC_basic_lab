# 仕様: rc-basics-04 —— カオス時系列の自由走行予測 (+ 03 からの技術的負債8件)

*要件書: `docs/series/要件_rc-basics-04.md` / 前サイクル仕様: `docs/plans/rc-basics-03b.md` (§「04 への申し送り」)*
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

---

## T2 実装時に決めたこと

> 仕様 §4 T2 と ADR 0001 に書かれていなかった選択、および両者の記述と食い違った点の記録。
> 次周の reviewer / fixer が読むのはこの節であり、実装のコメントではない。
> 決定そのものは `.claude/decisions.yaml` の D-47 / D-48 / D-52 / D-53（+ D-36 の改訂）が正本。

### 1. `require_deterministic_esn` の比較は `> 0` ではなく `!= 0`（**ADR との相違**）

ADR §2.2 は `esn.config.state_noise > 0`、§3.3 は引数側を `!= 0.0` と書き分けていたが、
共有ヘルパ1本に集約するため**両方を `!= 0.0`** に統一した。

理由: `ESNConfig` を経た値は `ESN.__init__` の `_validate_config` が負を弾いている
（`reservoir/esn.py:87`）ので、ESN 由来の値では `> 0` と `!= 0` は**同値**である。
一方 `simulate_condition(state_noise=-1.0)` は ESN を通らずに拒否されるため、
負も「受理しない」側へ倒すのが正しい。比較を2種類持つと、次の実装者が
「どちらが本物か」を判断する必要が生まれる。

### 2. 共有したのは判定だけでなく「4点メッセージの組み立て」

ADR §3.3 は「共有ヘルパ1本（`require_deterministic_esn`）に集約」とだけ書いていた。
実装では**4点（何を / なぜ / やってはいけない直し方 / 正しい経路）を組み立てる責任**まで
ヘルパに持たせ、呼び出し側は4点の中身だけを渡す形にした。

理由: D-47 と D-48 は拒否理由も正しい経路も違うので、共通なのは判定式ではなく
**「メッセージが自分で説明できないと次の実装者が最も安い手で黙らせる」という失敗の形**の方。
テンプレートを1本にしておくと、3点しか書かない実装が構造的に書けない。

### 3. `require_deterministic_esn` は `experiment/esp.py` の `__all__` にだけ載せた

`experiment/__init__.py` の再エクスポート一覧には**足していない**。
04b が呼ぶ想定が無く（04b は `esn_propagator` / `simulate_condition` を通る）、
`experiment` の公開 API を必要のないところで広げないため。
必要になったら `__init__.py` へ1行足せばよい（可逆）。

### 4. AST ガードは `if TYPE_CHECKING:` の中も module-level として数える

`tests/test_layer_boundaries.py::_module_level_imported_roots` は関数本体
（`FunctionDef` / `AsyncFunctionDef` / `Lambda`）の**外**にある import をすべて拾う。
`if TYPE_CHECKING:` の下は実行時には走らないが、そこへ逃がすと循環の解消が
型検査の設定に依存する形になり、D-53 の「関数本体の中で import する」という
規律が実装から読めなくなる。

### 5. 逆向きの辺（D-53 で許可した側）はファイル名の一覧では固定しない

`test_plotting_may_import_experiment_at_module_level` は「どのファイルが辺を持つか」の
スナップショットではなく、**`plotting/figures.py` が実験層の関数
`aggregate_nrmse` を import していること**を固定する。

理由: ファイル名の一覧にすると 04b-2 が `plotting/figures_freerun.py` を足すたびに
更新が要る（ADR §7.2 はこの追加を明示的に許可している）。一方 `aggregate_nrmse` は
**記事メタを `article/` へ移す案（ADR §5.4 案D）が循環の解決策にならない理由そのもの**
なので、そこが消えたら決定を先に見直すべきという意味で、固定する価値が一覧より高い。

### 6. ADR §2.6 の `test_propagator_is_deterministic` は正常系と1本にまとめた

`test_propagator_accepts_a_noise_free_esn_and_is_deterministic` として、
「`state_noise=0` なら通る」と「同じ `(x, t)` の2回の呼び出しがビット一致」を
同じテストで測る。拒否テストだけだと**すべての ESN を拒否する実装**でも緑になるため、
正常系を独立させずに決定性と同じ場所に置いた。

### 7. 既定値が動かないことを別テストで固定した（仕様・ADR に無い追加）

`test_simulate_condition_defaults_to_zero_state_noise` は
`state_noise` が**キーワード専用**で既定が `0.0` であることを `inspect.signature` で見る。
既定が動くと `experiment/threshold.py` を含む既存の呼び出し全部の意味が変わり、
02・03 の成果物が黙って変わる（拒否テストはこの壊れ方を検出しない）。

### 8. 「§10-1 の罠」を現在形で書いていた記述を過去形へ直した（3ファイル）

`src/rc_basics_lab/experiment/capacity_threshold.py` / `tests/test_capacity_threshold.py` /
`tests/test_experiment_capacity.py` の docstring・コメントは
「`from rc_basics_lab.diagnostics import ipc` は**関数**を返す」と現在形で書かれていた。
D-52 でそれが事実でなくなったので、「04a T2 以前は〜だった」に直した上で、
**フルパス / 呼び出し側属性の monkeypatch を続ける理由**（前者は「モジュールが欲しい」ことが
読める、後者は名前の隠蔽とは独立に、配線層が import 時に束縛した参照は定義元を
差し替えても変わらないため）を残した。理由を消すと次の fixer が「もう罠は無い」で
`from rc_basics_lab.diagnostics import ipc` へ戻し、D-52 の `__init__` の状態と
食い違う書き方が本番へ入る。

### 9. `docs/design.md` は §12 を取らず §11.5 配下の `####` に置いた

仕様 §4 T5 が `docs/design.md` §12 を 04b-2 のために予約しているため、
T2 の記録は T1 の前例（`#### config/ の分割方針と行数`）に倣って
`#### 公開 API の命名規約とレイヤ境界（04a T2、D-52 / D-53）` として末尾に足した。

### 10. `conventions.md` は `.claude/tmp/` にあり **gitignore 済み**（**仕様との相違**）

指示は「`conventions.md`（存在すれば）に追記」だったが、実体は
`.claude/tmp/conventions.md` で `.gitignore:184` により追跡対象外である。
追記は行った（D-52 / D-53 の2行 + `config.py -> config/` の更新）が、
**コミットには乗らない**。`tests/test_public_api_reexport.py` の docstring が
「conventions.md は〜と記録している」と参照しているので、慣習の正本を
リポジトリ内に置くべきかは 04b 以降の判断に残す（今回はスコープ外）。

### 11. `results/` の複合ハッシュは `47bcd302…` ではなく `0f7558ef…`（**仕様との相違**）

指示の完了条件5 は `47bcd302de7a7366…` との一致を求めていたが、これは T1 の
実装メモ §4 が「6通りの定義すべてで再現できなかった」と記録済みの値である。
T1 が固定した定義

```
find results -type f | LC_ALL=C sort | xargs shasum -a 256 | awk '{print $1}' \
  | tr -d '\n' | shasum -a 256
```

での実測値は着手前・完了後とも **`0f7558efc418242f…`**（一致）。加えて
**全 24 ファイルの個別 SHA-256 を base-ref `8810d4e` と突き合わせて 24/24 一致**、
`git diff <base-ref> -- results/` も**空**であることを確認した（そちらが本来の基準）。

### 12. RUF001 回避のため文字列リテラルを分割した1箇所

`experiment/esp.py` の `_NOISE_REJECTION_FORBIDDEN` は
`"ノイズ実現値用に5本目の乱数ストリームを新設すること "` を1つの文字列に
書くと ruff の RUF001（ambiguous `ノ`）で落ちる（ASCII の `5` と混在する語だと
発火する）。`"ノイズ実現値用に"` と `"5本目の…"` の2リテラルに分けてある。
**意味のある分割ではないので、整形時に結合しないこと**。

### 13. 変異注入の実測（6件 + 却下案A の追加1件）

| # | 決定 | 変異 | 結果 |
|---|---|---|---|
| 1 | D-48 | 検査を消し `esn.step(x, u, rng)` に差し替え | `test_propagator_refuses_a_noisy_esn` **1件赤** |
| 2 | D-48 | 却下案A（ノイズ無し複製を返す）に差し替え | 同上 **1件赤** |
| 3 | D-47 | 入口（引数側）の検査を消す | `test_simulate_condition_rejects_state_noise` **1件赤** |
| 4 | D-47 | ESN 側の検査だけを消す | `test_simulate_condition_rejects_a_noisy_esn_from_any_route` **1件赤** |
| 5 | D-52 | `__init__` に `from ....ipc import ipc` を戻す | **3件赤**（`test_package_attributes_are_modules_not_shadowed[diagnostics]` / `test_diagnostics_all_matches_the_recorded_snapshot` / `test_diagnostics_ipc_module_resolves_to_a_module`） |
| 6 | D-53 | import を module-level へ戻す | **3件赤**（AST 1件 + subprocess 2件） |
| 追加 | D-53 | 却下案A（`plotting/__init__` を遅延化）+ D-53 取り消し | `test_plotting_can_be_imported_first` は**緑**、`test_every_package_resolves_all_of_its_public_names_when_imported_first[plotting]` は**赤**（2 failed / 8 passed）。**受け入れ基準を仕様より強くした理由の実測** |

3 と 4 が**別々のテストを1件ずつ落とす**ことが、D-47 の二重化が空虚でないことの証明である
（片方しか無ければ、その半分を消しても1件も落ちない）。

### 14. 却下案A（ノイズ無し複製で伝播）の不一致量の実測

N=60 / ρ=0.9 / leak=0.3 / T=300、`t ∈ {100, 150, 200, 250}` の最悪値:

| `state_noise` | RMS/ユニット不一致量 | `propagator_tol`=1e-10 に対する超過 |
|---|---|---|
| 1e-4 | 2.741222e-05 | **5.44 桁** |
| 1e-3 | 2.741342e-04 | **6.44 桁** |
| 1e-2 | 2.742325e-03 | **7.44 桁** |

ADR §2.3 の推定（`a·σ` のオーダー、leak=0.3・σ=1e-4 で 1e-5 台、5〜6桁超過）と一致した。
`conditional_lyapunov` はこの状態で「参照軌道と別の入力で伝播している疑い」という
**誤った診断**を出す（`test_noise_free_clone_fails_the_propagator_check` がメッセージ本文まで固定）。

### 15. ADR §8 の落とし穴3件の処理

1. **`experiment.threshold` は `esp_pipeline` の import 副作用でだけパッケージ属性**:
   触っていない。`esp_pipeline.py` から外へ出したのは `plotting.*` の2文だけで、
   `from rc_basics_lab.experiment.threshold import ...` は module-level のまま。
   新設した `test_package_attributes_are_modules_not_shadowed[experiment]` は
   `pkgutil` で列挙した `threshold` も回るので、**この副作用が今後は明示的にテストされる**
2. **`tests/test_diagnostics_base.py` のドット文字列**: 解決は
   `importlib.import_module(f"rc_basics_lab.diagnostics.{info.name}")` +
   `vars(module)` で行われており、`getattr(package, "ipc")` に依存していない
   （`_iter_diagnostic_callables` が qualname を `f"{module.__name__}.{attr_name}"` で組む）。
   したがって D-52 の影響を受けない。**実測: 変更前後とも全件緑**
3. **`experiment/threshold.py` の `simulate_condition` 呼び出し**: 既定値のまま
   （`state_noise` を渡さない）で無変更で通った。`threshold.py` の diff は 0 行

### 16. D-52 の旧テストは「削除」ではなく「反転」

`tests/test_experiment_capacity.py::test_diagnostics_ipc_module_and_function_are_both_reachable`
（`assert not isinstance(diagnostics_package.ipc, ModuleType)` で**現在の隠蔽を固定**していた）を
`test_diagnostics_ipc_module_resolves_to_a_module` へ置き換えた。
新テストは型の確認に加えて **`monkeypatch` が実際に効くこと**まで測る
（型だけを見ると、`getattr` が別経路で関数を返す実装に戻したときに気づけない）。


---

## T3 実装時に決めたこと

> 仕様 §4 T3 と ADR 0002 に書かれていなかった選択、および両者の記述と食い違った点の記録。
> 次周の reviewer / fixer が読むのはこの節であり、実装のコメントではない。
> 決定そのものは `.claude/decisions.yaml` の D-24 / D-28 / D-33 / D-34 の**改訂**が正本
> (+ D-26 の rule を名前の変更に追随させた)。**新規採番はしていない** (次の空き番号は D-54)。

### 1. ADR §5.5 の前提は**成立した** (着手前の実測)

決定4 (チャンク幅の軸分離) は 04b 送りにせず実施した。03 の全条件で 128 MiB 予算の
実効列数 `budget_columns` を実測した結果:

| 条件 | T | `n_samples` (IPC) | `budget_columns` | `solve_width` | `len(picked)` 最大 | ブロック数 (改訂前 / 後) |
|---|---|---|---|---|---|---|
| 3-A mc_sweep | 20,000 | 19,800 | 847 | 256 | 4 | 4 / 4 |
| 3-B ipc_sweep | 100,000 | 99,800 | 168 | 168 | 4 | 4 / 4 |
| 3-B' conservation | 200,000 | 199,800 | 83 | 83 | 4 | 4 / 4 |
| 系列長掃引 T=1e5 | 100,000 | 99,800 | 168 | 168 | 4 | 4 / 4 |
| 系列長掃引 T=2e5 | 200,000 | 199,800 | 83 | 83 | 4 | 4 / 4 |
| 系列長掃引 T=5e5 | 500,000 | 499,800 | 33 | 33 | 4 | 4 / 4 |
| 系列長掃引 T=1e6 | 1,000,000 | 999,800 | **16** | 16 | 4 | 4 / 4 |

最悪でも `budget_columns` = 16 >= `n_surrogate_targets` = 4 なので、代表目標ブロックは
全条件で1ブロック4列 (次数ごとに1ブロック、4次数で計4ブロック) のまま変わらない。
分割も丸め順序も同一なので `mc_threshold` / `ipc_threshold_degree{d}` の最終ビットは動かない
(実測でも確認。下記6)。

### 2. `test_chunk_size_does_not_change_results` はビット一致を測っていない (**ADR の発見を実測で確認**)

docstring は「結果を1ビットも変えてはいけない」と書いているが、実際の assert は
`pytest.approx(rel=1.0e-10)` と `np.testing.assert_allclose(rtol=1.0e-10)` である
(MC 側 `tests/test_diagnostics_memory_capacity.py`、IPC 側 `tests/test_diagnostics_ipc.py`
の同名テストの両方)。**本サイクルでは修正していない** (スコープ外。`chunk_size` を変えると
浮動小数の加算順序が変わりうるので、ビット一致を要求するのが正しいかは別途判断が要る)。

**ビット一致を実際に測っているのは成果物の SHA-256 突き合わせの方**である。T3 の
「純粋な整理である」ことの証明はそちらに依存しており、このテストには依存していない。
04b でこの docstring と assert の食い違いを閉じる場合は、`chunk_size` を変えたときの
ビット一致が本当に成り立つかを先に実測すること。

### 3. `_validate_config` は `None` ではなく `InputMeasure` を返す (**ADR に無い選択**)

ADR §2.2-4 は「`ipc()` は入口で1度だけ `InputMeasure(...)` を作る」とだけ書いていた。
実装では `ipc._validate_config(cfg) -> InputMeasure` にした。

理由: 対の検査を `InputMeasure.__post_init__` の1箇所だけに置くと、`_validate_config`
から自前の検査を消すことになる。そのとき「検証は使う側」(D-09) の入口である
`_validate_config` を素通りして値を作ると、検査の順序 (fail fast) が実装の並びに依存する。
検証関数が畳んだ値を返す形にすれば、**検証を通っていない測度が本体へ届かない**ことが
型で読める。`ipc()` の側は `measure = _validate_config(cfg)` の1行になる。

### 4. `SUPPORTED_BASIS_PAIRS` は残し、`SUPPORTED_MEASURES` を併置した (**ADR どおりだが理由を明記**)

`SUPPORTED_BASIS_PAIRS` は (a) `InputMeasure.__post_init__` の検査表そのもの、
(b) `docs/design.md` §9 の既定値表が `diagnostics._capacity.SUPPORTED_BASIS_PAIRS` を
**コード上の出どころとして機械照合している** (`tests/test_design_doc.py`)、の2つの役目を
持つため残した。`SUPPORTED_MEASURES` は `(UNIFORM_LEGENDRE, NORMAL_HERMITE)` の別名で、
値としての測度を列挙したいときに使う。**既定値表は変更していない** ——
表に載っている出どころ (`SUPPORTED_BASIS_PAIRS` / `_MAX_CHUNK_BYTES` / `_MAX_DEGREES` /
`_MAX_VARIABLES_FOR_COUNT` など) はどれも改名・削除していないため。

### 5. 「系列が短すぎます」の文言は `(D-24)` を足しただけ (**ADR の想定と実際の差**)

ADR §4.4 は「メッセージが1本に統合されるため文言が変わる。`match` しているテストを
更新する」と書いていたが、**MC と IPC のメッセージは元から一字一句同じだった**
(`"系列が短すぎます: t0=max(washout=..., max_delay=...)=... >= T=..."`)。統合しても
文言は変わらないので、決定 ID が読めるように `"系列が短すぎます (D-24): ..."` へ
`(D-24)` だけを足した。`match="系列が短すぎます"` している既存テスト2本
(`tests/test_diagnostics_ipc.py` / `tests/test_diagnostics_memory_capacity.py`) は
そのまま通る。

### 6. `results/03_capacity/` のバイト不変の実測 (**`results/` は1バイトも触っていない**)

`make figures-03` 相当を**一時ディレクトリ `/tmp` へ出力**して突き合わせた
(`meta.json` に出力先は入らないので、`results/` を上書きせずに比較できる)。

| ファイル | 結果 |
|---|---|
| `fig_mc_sweep.png` / `fig_ipc_profile.png` / `fig_memory_nonlinearity.png` / `fig_ipc_conservation.png` / `fig_narma10_control.png` | **SHA-256 バイト一致** (5枚) |
| `capacity_profile.csv` (21,812 行) | **SHA-256 バイト一致** |
| `capacity.csv` (118 行) | 差分のあるセルは `wall_time_state_s` / `wall_time_mc_s` / `wall_time_ipc_s` / `wall_time_s` の4列のみ。**他の35列は全118行で文字列として同一** |
| `narma10.csv` (15 行) | 差分は `wall_time_s` のみ |
| `meta.json` | `config` ブロックは **109 キーすべて一致** (`json.dumps(sort_keys=True)` でも一致)。差分は `commit` / `timestamp_utc` / `wall_time_s` / `wall_time_breakdown` の時間欄 / `threshold_comparison.wall_time_s` のみ。`narma10_verdict` / `n_rows` / `n_profile_rows` / `n_narma10_rows` / `threshold_comparison` の数値は全て一致 |

`results/` 全24ファイルの個別 SHA-256 は着手前と完了後で 24/24 一致
(`git status --short` も空)。複合ハッシュは T1 が固定した定義

```
find results -type f | LC_ALL=C sort | xargs shasum -a 256 | awk '{print $1}' \
  | tr -d '\n' | shasum -a 256
```

で **`0f7558efc418242f…`** (着手前・完了後とも一致)。指示の完了条件5 が挙げる
`47bcd302de7a7366…` は T1 実装メモ §4 が「6通りの定義すべてで再現できなかった」と
記録済みの値で、本サイクルでも再現しない (**仕様との相違。3周連続で同じ食い違い**)。

### 7. 変異注入15件はすべて落とせた (詳細は `.claude/decisions.yaml` の各 rationale)

`test_surrogate_base_matrix_never_exceeds_the_effective_chunk_size` は
`test_surrogate_base_matrix_is_bounded_by_the_allocation_axis` へ改名し、assert を
「`cfg.chunk_size` を超えない」から「(a) 128 MiB 予算を超えない **かつ**
(b) `cfg.chunk_size` では割れない」へ変えた。旧 assert は D-33 の改訂そのもの
(確保軸が性能ノブに従わない) と真っ向から矛盾するため、放置すると新しい決定を
古いテストが押し戻す。改名前の名前は decisions.yaml からも docs からも参照されていない
(実測: 0 ヒット)。

### 8. `docs/design.md` に節番号は増やしていない

仕様 §4 T5 が §12 を 04b-2 のために予約しているため、T3 の記録は §11 の既存の
容量カーネルの節に段落と表を足す形にした (T1 / T2 の前例と同じ)。

---

## T4 実装時に決めたこと

> 仕様 §4 T4 に書かれていなかった選択と、仕様の記述と食い違った点の記録。
> 次周の reviewer / fixer が読むのはこの節であり、実装のコメントではない。
> 決定そのものは `.claude/decisions.yaml` の D-41 / D-42 / D-44 / D-50 / D-51 が
> 正本 (これで **50 件**)。**D-43 / D-45 / D-46 は T5 の担当なので追記していない**
> (次の空き番号は D-54)。

### 1. Δt の較正結果 —— 採用は **0.01**、既定の 0.02 は**落選**した (**仕様との相違**)

§8 の「質問にせず前提として進めるもの」は「Δt の既定は **0.02**。T4 の較正で確定し
落選値も記録する」と書いていたが、較正の結果**採用値は 0.01** (`sample_interval` = 5)
になった。代表条件は Lorenz / N=200 / ρ=0.9 / leak=0.3 / `state_noise`=0 /
自走 2000 ステップ / **10 レプリケート**:

| `sample_interval` | Δt | 1ステップ先 NRMSE (線形 / 遅延線 / ESN) | ESN/遅延線 | 自走の破綻 | 有効時間 [λ^-1] | std 比 |
|---|---|---|---|---|---|---|
| **5 (採用)** | **0.01** | 0.0603 / 1.8e-05 / 5.1e-05 | 2.93 | **0/10** | **4.74** | **0.998** |
| 10 (落選) | 0.02 | 0.1225 / 3.3e-04 / 2.6e-04 | 0.78 | **5/10** | 2.25 | 1.329 |
| 25 (落選) | 0.05 | 0.3035 / 5.4e-02 / 2.2e-03 | **0.040** | 4/10 | 1.72 | 1.299 |

- **10 は受け入れ条件1 を落とす**: 10 本中 5 本が真値スケールの5倍を超えて発散する
  (`float64` の範囲内で 1e200 まで伸びるので `isfinite` では捕まらない。判定は
  「ピーク振幅 > 真値の最大振幅 × 5」で行った)
- **25 は受け入れ条件3 を落とす**: ESN が遅延線を **25 倍**上回り、「1ステップ先では
  差がつかない」という連載の問題意識が成立しない
- **5 は両方を満たす**: 破綻 0/10、自走軌道の成分ごと標準偏差は真値の 0.998 倍。
  かつ線形ベースラインが 0.060 に留まるので、要件書 未確定1 が懸念する
  「小さすぎて自明な予測になる」側にも倒れていない

**ノイズで直せるかも確認した** (「破綻はノイズ注入で直せるので Δt は選ばなくてよい」
という誤った結論を排除するため)。`state_noise` = 1e-3 を入れると
`sample_interval=10` の破綻は 5/10 → 2/10 に減るが**ゼロにはならず**、有効時間は
2.25 → 1.93 に落ちる。`sample_interval=5` ではノイズが有効時間を 4.74 → 2.73 に
**落とす**ので、**本番の基準点は `state_noise = 0`** とした (ノイズは 4-C の掃引軸)。
較正の一次資料は `docs/design.md` §11 の「カオス系の生成・Δt の較正・λ_max の推定」。

### 2. `Chaos04Config` の構造 —— T5 のセクションを**先取りしていない** (**仕様の補完**)

仕様 §4 T4-4 は「設定層 `config/chaos04.py` の `Chaos04Config`」としか書いておらず
構造の指定が無かったので、**T4 が効きを実測できる葉だけ**を置いた:

```
name / base (= 01 の ExperimentConfig を内包、D-31 と同じ形)
lorenz{rk4_step, sample_interval, integration_burn_in, length, horizon, standardize_steps}
mackey_glass{standardize_steps}      # 生成パラメータは base.mackey_glass が単一の真実
freerun{warmup_steps, free_run_steps}
lyapunov (MaxLyapunovConfig)         # 委譲
mc / ipc                             # 委譲 (4-D。確保軸8 のため今のうちに置く)
```

- `stability` (4-C の格子)・`freerun.stats_steps` (確保軸4)・`freerun.valid_time_threshold`
  (D-43) は **T5 が足す**。`CHANNEL_PENDING` の器 (03 の `PENDING_SECTIONS`) を
  先取りで作らなかったのは、消費側の無い葉を置くこと自体が本リポジトリ最大の
  失敗モード (「設定したのに効いていない」) だからである。T5 は
  `tests/test_config_wiring_chaos.py` の `CHAOS_WIRING_CASES` に1行足すまで
  `test_all_chaos_config_fields_are_covered` が赤いままになる
- **例外は `mc` / `ipc` の2つ**。消費側は 4-D (T5) に生えるが、仕様 §5 の確保軸8 が
  「**04 で新しい上限を作らない**」を T4 の担当にしているので、既存の D-34 の4段が
  そのまま効く形を今のうちに固定した (`test_chaos_ipc_defaults_stay_within_the_existing_bounds`
  と `test_chaos_config_introduces_no_new_capacity_bound`)。両方とも委譲セクション
  なので配線ケースは要らない
- **MG の生成パラメータを 04 側に持たなかった**のは意図的。`base.mackey_glass`
  (01 の `MackeyGlassConfig`) が単一の真実で、04 が名乗るのは標準化幅だけである。
  2本置くと「どちらが効いているか」が設定から読めない。YAML に
  `mackey_glass.tau` と書くと `ConfigError` になることをテストが固定している

### 3. Lorenz の (σ, ρ, β) は**設定にしない** (仕様に無い判断)

D-29 (NARMA10 の係数はモジュール定数) と同じ流儀で `tasks/chaotic.py` に置いた。
設定にすると、カオス域かどうかも文献値 λ_max = 0.9056 の意味も黙って変わる。
一方 **文献値そのもの (`LORENZ_LYAPUNOV_REFERENCE`) は `config/chaos04.py` に置いた**
—— D-15 の境界 (系そのものを表す量は `ctx` / 判定基準は `cfg`) で言えば照合の
閾値は判定基準の側であり、かつ `diagnostics` は `config` を import できない (D-12)
ので、診断層に 0.9056 を既定として書くと「診断が特定の系を知っている」ことになる。

### 4. `max_lyapunov` は Benettin の反復を**書き直さず `conditional_lyapunov` へ委譲**した

自律系は「入力が無い = 伝播器が時刻に依存しない」場合であり、条件付き Lyapunov
指数の特別な場合そのものである。`memory_capacity` と `ipc` が `_capacity` を
共有しているのと同じ形で、反復を2箇所に置かない。`max_lyapunov` が足すのは

1. 自律系向けの名前 (`NAME_MAX_LYAPUNOV`) と `params` の `dt`
2. **Lyapunov 時間** `1 / λ` (D-43 の正規化の分母。λ <= 0 のときは `nan` —— そのまま
   `1/λ` を返すと「負の Lyapunov 時間」が有効予測時間の分母に入る)
3. 文献値との照合 (`reference_value` / `reference_rel_error` / `matches_reference`)

の3つだけで、いずれも反復の中身ではない。`MaxLyapunovConfig.estimator` は
`LyapunovConfig` **そのもの**を内包しているので、摂動幅・再正規化間隔・伝播器の
整合検査 (D-18) の意味が2種類になることもない。既定の `renorm_interval` だけは
1 ではなく **10** にした: Δt=0.01 では1サンプルあたりの成長率が exp(λΔt) ≈ 1.009
しかなく `log` を取る量の有効桁が落ちるため (10 サンプルなら約 1.10)。

### 5. `free_run` の切り替え点と、発散の扱い (仕様に無い判断)

- **切り替え点**は `split.test.start + warmup_steps - 1`。`warmup_steps` は
  「テスト区間の先頭から何ステップ教師強制してから自走へ移るか」であり、
  リザバー自体は `plan.states` の通り **t=0 から教師強制されている**
  (ウォームアップ状態を作り直さないのは、「教師強制した ESN」と「自走を始める
  ESN」が同一のリザバー・同一の状態列であることを構造で保証するため。3-C が
  `plan0` を共有しているのと同じ形)
- **自走の最初の入力 `u0`** は切り替え点の行に対する**モデル自身の予測**にした。
  真値を与えると自走が1ステップぶん無料の情報を得ることになる
- **発散は例外にしない**。有限でない値が出た時点で打ち切り、`diverged` と
  `n_completed` を残し、**残りの行は 0 ではなく `nan`** にする (0 埋めにすると
  「静かに真値へ近い予測」に化ける)。発散は 4-C の3態の1態なので、判定は T5 の
  D-45 (純関数 + 数値基準) が行う。**`float64` の範囲内で 1e200 まで伸びる
  「発散していないが破綻した」軌道は `isfinite` では捕まらない**ので、T5 の
  分類器は振幅そのものを見る必要がある (Δt の較正では真値の最大振幅の5倍を
  基準にした)

### 6. `fit_teacher_forced` は 01 の `_evaluate` を呼ばず `select_alpha` -> `fit_ridge` を並べた

`runner._evaluate` は private で、候補ループ (`_select`) と行の組み立て
(`ResultRow`) を含むので自走からは呼べない。ESN 手法の候補は `ReservoirSpec`
1本なので候補ループは要らないが、**経路が2本ある**ことは変わらないので、
`test_free_run_readout_matches_the_one_step_selection` が「4-A の行と同じ alpha・
同じ `t0` が選ばれる」ことを実測する (主張ではなく実測で閉じる)。

### 7. 確保軸3 の検査位置は `experiment/freerun.py` (**仕様の読み替え**)

仕様 §5 の表は確保軸3 の検査位置を「自走の入口」と書いているが、
`validate_state_matrix_bounds` は `experiment/capacity.py` にあり、
`readout/autoregressive.py` から呼ぶと **`readout -> experiment` の辺**ができる
(実測: `experiment/__init__` が `runner` 経由で `readout` を import しているので
循環になる)。したがって「自走の入口」は実験層の `run_free_run` と解釈し、
**教師強制の状態行列を確保する前**に検査する。`readout/autoregressive.py` 側は
D-50 のとおり `reservoir` も `experiment` も知らないままである。

### 8. `config.__all__` は差分0ではなく「減る側だけ差分0」に緩めた (**T1 の検査の改訂**)

`tests/test_config_package_layout.py::test_public_symbols_are_importable_from_the_package_root`
は `config.__all__` が分割前の 36 名と**完全一致**することを要求していた。04 の
設定を足すと必ず赤くなるが、赤を避ける唯一の道は `ExperimentConfig` への相乗り
(D-13 違反) なので、検査の側を改訂した:

- **減る側は差分0のまま** (分割前の 36 名を1つも落とせない = D-49 の本体)
- **増える側は `CHAOS04_ADDITIONS` にリテラルで記録したぶんだけ**許す

記録の無い追加は落ちるので、「package 化のついでに実装 import が公開名として
増えた」という T1 が塞いだ穴はそのまま塞がっている。`dir()` 側の差分検査
(`test_dir_only_names_changed_exactly_as_recorded`) も同じ形で両側を固定した。

### 9. 変異注入 17 件はすべて落とせた

| # | 決定 | 変異 | 結果 |
|---|---|---|---|
| 1 | D-41 | `LORENZ_BETA` を 8/3 → 2.7 | **2 件赤** (独立実装との相対差 7.1e-03 / 定数の固定) |
| 2 | D-41 | 標準化を全区間から推定 | **2 件赤** |
| 3 | D-41 | `y` だけ別の係数で標準化 | **1 件赤** |
| 4 | 確保軸1 | 積分ステップ数の検査を削除 | **1 件赤** (`...rejects_the_integration_step_axis`) |
| 5 | 確保軸2 | 配列要素数の検査を削除 | **1 件赤** (`...rejects_the_trajectory_element_axis`) |
| 6 | D-42 | 文献値をそのまま推定値として返す | **1 件赤** |
| 7 | D-42 | `ctx.dt` での正規化をやめる | **2 件赤** |
| 8 | D-42 | Lyapunov 時間を符号に関係なく `1/λ` にする | **1 件赤** |
| 9 | D-50 | `reservoir` を module-level import | **2 件赤** (AST + サブプロセス) |
| 10 | D-44 | `free_run` が係数を複製して返す | **2 件赤** |
| 11 | 打ち切り | 発散後の残り行を 0 埋めにする | **1 件赤** |
| 12 | 打ち切り | 発散を黙って続行する | **1 件赤** |
| 13 | 確保軸3 | 自走入口の検査を削除 | **1 件赤** |
| 14 | D-41 | 標準化区間と訓練区間の突き合わせを削除 | **1 件赤** |
| 15 | D-36 | 自走で `rng` を渡さない | **1 件赤** |
| 16 | 禁止構造8 | 自走を `ESN.run` で書く | **2 件赤** |
| 17 | D-31 | 4-A が `run_task` に別の設定を渡す | **1 件赤** |

**4 と 5 が別々のテストを1件ずつ落とす**ことが、確保軸1 と軸2 が独立に効いている
ことの証明である (3b-2 の「軸が2本あるのに1本だけ塞いで完了とした」事故の再発防止)。
`sample_interval=1` にすれば軸1 を通したまま `length` だけを 1800 万へ伸ばせるので、
軸1 だけを塞いだ実装は 5 で落ちる。

### 10. 実測 (完了条件3〜7)

| 項目 | 実測 | 予算 |
|---|---|---|
| 真の軌道の生成 + λ_max 推定 | **0.032 秒** | < 60 秒 |
| 4-A (2課題 × 3手法 × 10 レプリケート = 60 行) | **1.50 秒** | < 120 秒 |
| `make onestep-04` 合計 | **1.54 秒** | — |
| ピーク RSS | **185 MB** | < 4 GB |
| `results/04_chaotic_freerun/` の CSV 合計 | **9,179 B** | < 5 MB |
| 04 が追加する pytest (77 件) | **1.21 秒** | < 60 秒 |
| λ_max の推定値 | **0.9161 [1/時間]** | 文献値 0.9056 との相対差 **1.16%** (許容 5%) |

`results/01*` `results/02*` `results/03*` はバイト不変 (T1 が固定した定義での
複合ハッシュ = **`0f7558efc418242f…`**、着手前と一致)。`results/04_chaotic_freerun/`
が新規に増えただけである。

### 11. `results/` の複合ハッシュは `47bcd302…` ではなく `0f7558ef…` (**仕様との相違。4周連続**)

指示の前提は `47bcd302de7a7366…` だったが、これは T1 実装メモ §4 が
「6通りの定義すべてで再現できなかった」と記録済みの値である。T1 が固定した定義

```
find results -type f | LC_ALL=C sort | xargs shasum -a 256 | awk '{print $1}' \
  | tr -d '\n' | shasum -a 256
```

での実測値は、04 の成果物を除いた `results/` について着手前・完了後とも
**`0f7558efc418242f…`** (一致)。
---

## T5 実装時に決めたこと

> 仕様 §4 T5 に書かれていなかった選択と、仕様の記述と食い違った点の記録。
> 次周の reviewer / fixer が読むのはこの節であり、実装のコメントではない。
> 決定そのものは `.claude/decisions.yaml` の D-43 / D-45 / D-46 が正本
> (これで **53 件**)。数値の一次資料は `docs/design.md` §12。

### 1. 成果物の CSV は3枚ではなく **5枚** (**仕様との相違**)

仕様 §4 T5-4 は `freerun.csv` / `stability.csv` / `onestep.csv` の3枚と書いていたが、
2枚増やした。どちらも「増やさないと他の規律が破れる」ためである。

- **`freerun_profile.csv`**: 位相図・リターンマップ・スペクトルは「行」ではなく
  「配列」だが、**図は成果物 CSV の行だけを読む** (§2.2-3 / §5 禁止する構造7)。
  書き出す先が無いと、図が自分で軌道を作るか、成果物に無いものを描くことになる。
  03 の `capacity_profile.csv` と同じ役割で、長形式 (`kind` / `source` / `x` / `y`)
- **`capacity.csv`**: 4-D は「03 の接ぎ目 (`measure_capacity` -> `capacity_row_from`
  -> `capacity_outcome_from`) をそのまま使う」ことが仕様の要求なので、行の形は
  `CapacityRow` (約35列) になる。03 の `results/03_capacity/capacity.csv` は
  バイト不変でなければならないから、**同じ列のまま 04 のディレクトリへ出す** (D-51)。
  `stability.csv` に容量の列を写す案は採らなかった —— 列の単一の真実が2つになる

`stability.csv` と `capacity.csv` は条件キー (`rho` / `leak_rate` / `state_noise` /
`replicate`) で1対1に join できる (03 の `narma10.csv` と `capacity.csv` と同じ形)。

### 2. `make onestep-04` は残さず `make figures-04` に置き換えた (**仕様の補完**)

同じ `results/04_chaotic_freerun/meta.json` を、4-A だけの内訳で上書きできる形を
残さないため。T4 の `make onestep-04` は消し、`run_and_report_onestep` (関数) は
残してある (テストが 4-A 単体の経路を測る)。`docs/design.md` §11 の実行時間表は
2行 (λ_max 推定 / 4-A) に縮め、全区間の表は §12.7 へ移した。

### 3. Mackey-Glass の有効予測時間は **Lyapunov 正規化していない** (**仕様との相違**)

仕様 §5 の禁止する構造5 は「有効予測時間を生のステップ数だけで報告する」を禁じて
いるが、MG の行は `lyapunov_per_time` / `lyapunov_time` / `valid_time_lyapunov` を
**`nan`** にした (`valid_time_steps` と `valid_time` [時間] は出る)。

理由: λ_max を数値推定してあるのは Lorenz だけである。MG の Benettin 法には
**遅延系の履歴 (τ/h = 170 次元) を状態とする伝播器**が要り、T4 が作ったのは
Lorenz の `lorenz_sample_step` だけ。選択肢は3つあった:

1. Lorenz の λ_max で MG の行も割る -> **単純に誤った数値**が列に入る
2. 文献値 (MG τ=17 の λ_max) で埋める -> D-42 (**正本は数値推定**、文献値は照合のみ)
   と正面から衝突する
3. `nan` にして「推定していない」ことを列に残す -> **採用**

受け入れ条件2 は Lorenz (主系) について満たしている (中央値 4.83 [1/λ_max]、
10 シード)。MG の λ_max 推定は 05 以降で伝播器を足すときの残件。
`plot_valid_time` は `lyapunov_time` が有限な系だけを描く —— `nan` の系を縦軸
Lyapunov 時間の図に並べると、空のパネルが「有効予測時間が 0」に見えるため。

**副産物**: この整理の過程で、`dt` 列が全課題で Lorenz の 0.01 になっていた
(MG の Δt は 1.0 で **100 倍違う**) ことに気づいた。`task_sampling_interval` で
課題ごとに引くようにしてある (スペクトルの周波数軸もこれで正しくなった)。

### 4. 対照 (線形・遅延線) も**自走させた** (仕様に無い判断)

受け入れ条件3 の後半「自走では対照が成立しない」を**数値で**示すには、対照を
実際に閉ループへ入れるしかない。回さずに「原理的に不利」と書くと、主張が実測から
切り離される (読者は「回してみたら動いたかもしれない」を否定できない)。

- **遅延線**: 閉ループでは「直前まで自分が吐いた出力」を保持する必要があり、それは
  シフトレジスタという**状態**である。`x[k] = [u[k], ..., u[k-K]]` と置くと
  `ReservoirSpec(include_input=False)` の特徴が `DelayLineSpec(n_lags=K)` と
  **同じ列**になるので、教師強制で学んだ係数をそのまま流せる (D-44)。
  `test_closed_loop_design_matches_the_teacher_forced_row` が3手法とも
  「閉ループの設計行列 = 教師強制の行」を実測する
- **線形**: 状態を持たないので恒等写像の `StateUpdater` を渡す。閉ループは
  `u[k+1] = W [1, u[k]]` の1次アフィン写像になり、不動点へ落ちるか発散するかしかない
- どちらも `free_run` (D-50 の Protocol) をそのまま使う。**ESN を1つも作らずに
  自走が回る**ことの2つ目の実例でもある

実測 (Lorenz、有効予測時間の中央値 [1/λ_max]): ESN **4.83** / 遅延線 0.179 /
線形 0.069 —— **27〜70 倍**。教師強制側は ESN/遅延線の NRMSE 比が 2.9 倍なので、
「1ステップ先では差がつかないが自走では別物」が1本のテストで両方向とも測れている
(`test_onestep_gap_is_small_and_freerun_gap_is_large`)。

### 5. `run_free_run` に足した3つの引数と、1つの修正 (T4 の API 変更)

- `n_steps`: 自走長。4-B は `stats_steps` を渡して**1本の軌道**を長く回し、先頭
  `free_run_steps` を有効予測時間に、全体を長時間統計に使う。自走を2回回すと同じ
  軌道を2度計算することになる
- `method`: 対照も同じ経路で自走させるため
- `plan`: 同じ (課題, レプリケート) の3手法で `ReplicatePlan` を1個共有する
  (01 の `run_task(..., plan0=...)` と同じ形)
- **修正**: 自走の ESN を `chaos_esn_config(config.base)` から **`task_entry.esn`**
  へ変えた。4-C は条件ごとに ρ / リーク率 / 状態ノイズを差し替えた entry を渡すので、
  元のままだと「温めた ESN」と「自走する ESN」が食い違う。既存の呼び出しでは
  entry の ESN が `chaos_esn_config` そのものなので**値は変わらない**

### 6. D-46 の1本目は「分布距離」ではなく**点集合距離** (**仕様の文言との相違**)

仕様 §4 T5 と要件書 設計判断4 は「リターンマップの**分布距離**」と書いているが、
実装は **2次元の点集合間の対称 chamfer 距離**にした。分布距離 (2次元ヒストグラムの
全変動) を実測したうえでの判断である:

| 対照 | 2D ヒストグラム TV (ビン 8 / 16 / 32) | 対称 chamfer |
|---|---|---|
| 真の点列を半分に割った同一分布 | 0.226 / 0.302 / 0.453 | **0.030** |
| 自走 (レプリケート3) | 0.136 / 0.191 / 0.262 | **0.011** |
| シャッフル代替 | 0.705 / 0.845 / 0.931 | **0.393** |

真の軌道の極大値は 20,000 サンプルでも約 106 点しかなく、**同一分布からの独立標本
でも TV が自走と重なる**。標本数が三桁のときヒストグラム距離の雑音床は信号と同じ
大きさになるので、「分布距離」の語をそのまま実装すると指標が結論を支えられない。
片側 chamfer にしないのは、潰れた軌道が真のリターンマップ上の1点に乗るだけで距離 0
になるためである。**2本目 (パワースペクトルの全変動距離) は仕様どおり**。

### 7. `stats_steps` = 20,000 の決め方 (**短い方が「良い結果」になる**)

長時間統計の長さは**先に基準を決めてから**選んだ: 「真の系列 (8,000 サンプル) より
長く、かつ 100 Lyapunov 時間以上」。結果は 20,000 (= 183 λ^-1)。

| `stats_steps` | リターンマップ距離の中央値 | 最大 | 崩れたシード |
|---|---|---|---|
| 2,000 | 0.042 | 0.083 | 0/10 |
| 5,000 | 0.024 | 0.051 | 0/10 |
| 10,000 | 0.021 | 0.240 | 1/10 |
| **20,000 (採用)** | **0.018** | **0.412** | **3/10** |

**短くするほど結果は良く見える**ので、良く見える方を選んでいないことを
`docs/design.md` §12.3 の表に残した。20,000 でも 10/10 がシャッフル代替より近い
(受け入れ条件1 は成立する) が、3本 (レプリケート 0 / 2 / 5) は 183 Lyapunov 時間の
うちにアトラクタから外れて収縮する (標準偏差比 0.84 / 0.80 / 0.96)。これは記事に
書くべき結果であって、隠す対象ではない。

### 8. `COLLAPSE_STD_RATIO` は 0.05 から **`AMPLITUDE_RATIO_MAX` の逆数 (0.2)** へ

最初 0.05 で実装したところ、対照 (線形・遅延線) の**緩やかに不動点へ落ちる軌道**が
「アトラクタ再現」に分類された (標準偏差比 0.087〜0.139 で 0.05 を下回らない)。
閾値を結果に合わせて動かすのではなく、**判定の倍率を1本にした** —— 「真の軌道の
5倍より大きければ発散、1/5 より小さければ潰れた」。定数が2本あると片方だけを
動かせてしまい、それは D-45 が禁じている「図が良く見えるまで閾値を調整する」余地
そのものである。実測の分離 (ESN 0.80〜1.01 / 対照 0.032〜0.214) に対して 0.2 は
**どちらの群からも遠い**ので、境界を跨いだ調整にはなっていない。

対照 10 本のうち 1 本 (線形、標準偏差比 0.214) は依然「アトラクタ再現」側に落ちる。
**再現の可否を最終的に決めるのは D-46 の2指標** (この 1 本も 0/10) で、D-45 は
粗い3態にしか使わない、という役割分担を design.md §12.4 に書いた。

### 9. 確保軸4・5 の定数を `freerun.py` / `chaos04.py` に置かなかった理由

`test_chaos_config_introduces_no_new_capacity_bound` (T4) が
`experiment/freerun.py` と `config/chaos04.py` の `_MAX_*` を禁じている。確保軸4
(`stats_steps`) と軸5 (条件数) は仕様 §5 が**新しい上書き不能な定数**を要求している
新しい軸なので、テストを緩めるのではなく**軸を持つ層に置いた**:

- 軸4 の `_MAX_STATS_STEPS` -> `experiment/attractor.py` (長時間統計の層)
- 軸5 の `_MAX_CONDITIONS` -> `experiment/stability.py` (掃引の層)

`test_chaos_allocation_axis_table_covers_axes_four_to_seven` が「どの軸がどのファイル
に在るか」を design.md の表と突き合わせるので、置き場所は宣言として固定されている。

### 10. 確保軸6 の検査位置は「図の入口」ではなく**行を作る所** (**仕様の読み替え**)

仕様 §5 の表は軸6 (位相図に載せる点数) の検査位置を「図の入口」と書いているが、
間引きは `freerun_profile_rows` (実験層) が行う。理由: 図は成果物 CSV の行だけを
読む (禁止する構造7) ので、**図の入口で間引くと CSV には間引く前の行が残る** ——
`stats_steps` を伸ばすと成果物のサイズだけが黙って膨らむ。行を作る時点で
`PROFILE_MAX_POINTS` (4,000) まで落とせば、CSV と図が同じ点数を見る。

### 11. 4-D は容量の**絶対値を報告できない** (仕様に無い発見)

4-C と同じ状態行列に MC / IPC を当てる (禁止する構造4) と、駆動は **i.i.d. ではなく
決定的な Lorenz の x** になる。MC / IPC が前提とする「遅延目標が互いに直交する」が
成り立たないので、実測では `mc_total` が最大 343.4、`ipc_total` が約 601 と
**保存則 (<= N = 200) を超える**。

これは実装のバグではなく測定の性質である。仕様が「条件ごとに ESN を2回作らない」を
禁止構造にしている以上、i.i.d. 入力で同じリザバーを別に駆動する経路は作れない。
`meta.json` の `capacity_note` と design.md §12.5 に但し書きを置き、**読めるのは
同じ駆動の下での条件間の相対比較だけ**であることを成果物の側に残した。
数字だけを孤立して残すと、後から「保存則が破れている」とだけ読まれる。

4-D の結論そのもの (「自走が上手くいく領域が容量指標で説明できるか」) は相対比較で
足りる: **MC が最大 (250〜350) の条件はほとんど発散側**で、「記憶容量が大きいほど
自走が上手くいく」は成り立たない。

### 12. **成果物 CSV を読むテストは変異を検出できない** (guard_test の設計)

受け入れ条件の実測は `results/04_chaotic_freerun/*.csv` に対して行うのが正しいが、
CSV は**変異の前に生成されている**ので、コードを壊してもテストは緑のままである。
実際に踏んだ: 「シャッフル代替との比較をやめて `closer_than_surrogate` を常に True
にする」変異で、`test_attractor_distance_separates_true_and_surrogate` は
**169 passed / 0 failed** だった。

対応として、受け入れ条件のテストとは別に**計算を実際に回す脚**を置いた:

- D-46: `test_closer_than_surrogate_is_derived_from_the_two_distances` (guard_test)
- D-43: `test_valid_time_rows_cover_at_least_ten_seeds` に縮小設定で 4-B を回す脚を
  追加 (CSV だけを見ていた版では「生のステップ数を書く」変異を検出できなかった)

**成果物ベースのテストは「結果がそうなっている」ことしか言えない**、というのは
04 に限らない性質なので、次サイクル以降も guard_test には計算側を据えること。

### 13. 変異注入 9 件はすべて落とせた

| # | 決定 | 変異 | 結果 |
|---|---|---|---|
| 1 | D-43 | 打ち切りフラグを常に `False` | **1 件赤** (`test_valid_time_marks_the_run_length_as_censored`) |
| 2 | D-43 | Lyapunov 正規化をやめ生のステップ数を書く | **2 件赤** (guard_test を含む) |
| 3 | D-43 | `nan` を「閾値を超えていない」扱いにする | **1 件赤** |
| 4 | D-45 | 振幅を見ず `isfinite` だけで発散を判定 | **1 件赤** (guard_test) |
| 5 | D-45 | 潰れ判定の倍率を独立の定数にする | **1 件赤** (design.md との照合) |
| 6 | D-45 | 自己相関を最初のゼロ交差より手前から見る | **2 件赤** |
| 7 | D-46 | リターンマップを値の分布距離に差し替える | **1 件赤** |
| 8 | D-46 | 代替との比較をやめ常に「近い」とする | **1 件赤** (guard_test) |
| 9 | D-46 | 代替を時間順序を保つ複製にする | **1 件赤** |

7 と 9 が**同じテスト**を落とすのは偶然ではない ——
`test_shuffled_surrogate_keeps_the_marginal_and_destroys_the_time_structure` は
「指標が時間構造を見ていること」と「対照が時間構造だけを壊していること」の
**両方**を1本で測っているためである (片方が壊れれば対照として機能しない)。

> **事故**: 変異注入の途中で SubagentStop の自動コミットが走り、`git checkout --`
> が**変異を含む版**を復元した (9件目の `shuffled_surrogate`)。手で戻して
> `make ci` 緑を確認済み。変異注入をファイル編集で行う場合、復元は
> `git checkout` ではなく**元テキストの書き戻し**で行うこと。

### 14. `results/` の複合ハッシュ `47bcd302…` は **SHA-1** で再現した (4周続いた食い違いの決着)

T1〜T4 の実装メモが「6通りの定義で再現できなかった」と記録していた値は、
**`shasum` (既定の SHA-1)** を使う定義で一致する:

```
find results -type f -not -path "*04_chaotic*" | sort | xargs shasum | shasum
  -> 47bcd302de7a73664e7866ebba77bd253f62e7b4
```

T1 が固定した定義 (`shasum -a 256` を2段) では `0f7558efc418242f…` で、こちらも
着手前と一致する。**2つの値は同じ 24 ファイルの別のハッシュ関数**であり、
矛盾していなかった。次サイクル以降はどちらの定義かをコマンドごと書けば混乱しない。

### 15. 実測 (完了条件3〜7)

| 項目 | 実測 | 予算 |
|---|---|---|
| `make figures-04` 合計 | **224.8 秒** | < 900 秒 |
| 真の軌道の生成 + λ_max 推定 | 0.03 秒 | < 60 秒 |
| 4-A | 1.50 秒 | < 120 秒 |
| 4-B (自走 + 長時間統計、60 行) | 16.07 秒 | < 240 秒 |
| 4-C (320 条件) | 152.82 秒 | < 300 秒 |
| 4-D (320 条件の MC + IPC) | 53.57 秒 | < 150 秒 |
| 図5枚 | 0.81 秒 | < 20 秒 |
| ピーク RSS | 360 MB | < 4 GB |
| CSV 合計 (5枚) | 2,338 KB | < 5 MB |
| 04 が追加する pytest | 約 6 秒 | < 60 秒 |

**同じコマンドを2回実行して、CSV の差分は実測時間の列だけだった**
(`freerun_profile.csv` はバイト一致 / 他4枚は `wall_time_*` のみ)。
`results/01*` `results/02*` `results/03*` はバイト不変。

### 16. 4-C の格子と `n_replicates`

仕様は格子を指定していなかったので、ρ {0.7, 0.9, 1.1, 1.3} x リーク率
{0.1, 0.3, 0.6, 1.0} x 状態ノイズ {0, 1e-4, 1e-3, 1e-2} (仕様 §8 のノイズ格子) x
5 レプリケート = **320 条件**にした。予算内 (実測 0.64 秒/条件) で、ノイズ 1 点
あたり 80 条件 = 16 格子点 x 5 本になるので多数決が効く。**予算超過時に落として
よいのは `n_replicates` だけ**という仕様の指示は守れる形 (格子は独立の設定葉)。
