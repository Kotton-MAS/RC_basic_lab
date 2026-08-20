# 仕様: rc-basics-05 実験実装 — センサー時系列の異常検知

*入力: `docs/要件_rc-basics-05.md` / `docs/survey_異常検知データセット_05.md` / `.claude/tmp/conventions.md`*
*planner 出力 (2026-08-20)。⚠ 分割不足の警告あり (L タスク3本) — §8 Q3 参照*

## 1. ゴール

実データの単変量時系列に対し、**同一前処理・同一閾値方針**でリザバー予測残差と3系統の対照
(+ 一様乱数・入力ノルムの常置対照) を AUPRC で比較し、
図5枚を `make figures-05` 1コマンドで再生成できるようにする。

## 2. 現状認識

### 関連箇所 (実測)

| 用途 | 場所 | 05 での使い方 |
|---|---|---|
| 課題の共通形式 | `tasks/base.py:24-77` (`TaskData`, `TaskGenerator`) | `u`/`y` は `(T,D)` 2次元必須。異常検知はラベル列が要るので `TaskData` を**継承・拡張しない**別 dataclass を新設 |
| 前処理の共通化 | `tasks/chaotic.py:155-211` (`Standardizer.from_training_prefix`, D-41) | **そのまま転用**。「係数を作れる場所を1箇所に閉じる」パターンが 05 の「前処理を手法間で完全共通化」の実体 |
| 手法切替の単一入口 | `readout/design.py:51-75, 78-101` (`FeatureSpec`) | ESN / 遅延線 / 線形は既存の3 spec で表現できる。**新しい手法分岐を作らない** |
| 1ステップ先予測の前例 | `readout/autoregressive.py:50-135` (`StateUpdater`, D-50) | 残差スコアは教師強制の1ステップ先予測なので自走は不要。`StateUpdater` の「reservoir 非依存」の書き方を踏襲 |
| 分割 | `experiment/split.py:67-131` (`compute_t0` / `make_split`) | 訓練 / 較正 / テストの連続3分割に転用。`compute_t0` は D-05 (全手法同一行) の実体 |
| 指標 | `metrics.py:1-67` | PR 系は0件。`_as_pair` の「形状検証 → 計算」パターンを踏襲 |
| 設定ローダ | `config/_common.py:39-128` | 受理できる型は **scalar / `tuple[X, ...]` / dataclass / `X|None` のみ**。**`dict` は不可** |
| 実験別設定 | `config/chaos04.py:53-213` (D-13) | `Anomaly05Config` を新設。`ExperimentConfig` に1フィールドも足さない |
| 1コマンド成果物 | `experiment/freerun_pipeline.py:1-101` (`FREERUN_ARTIFACTS`) | 「成果物一覧の単一の真実」タプルを 05 でも作る |
| CLI 配線 | `main.py:99-142`, `Makefile:75-83` | `_run_05` + `EXPERIMENTS["05"]` + `figures-05` / `data-05` |
| 文書の機械照合 | `tests/test_readme_summary.py:23-55`, `tests/test_design_doc.py:36-66` | README の数値表と `docs/design.md` §9 の既定値表は**機械検査されている** |
| 配線テスト機構 | `tests/wiring.py:36-131`, `tests/test_config_wiring.py:212-279` | `WiringCase` / `leaf_paths` を 05 用に複製 |
| レイヤ境界の機械検査 | `tests/test_layer_boundaries.py:98-153` (AST 走査) | 05 の「tasks は I/O しない」検査をここに足す |
| 反面教師 | `experiment/freerun.py` = **1620行** (実測)、`attractor.py` 715 / `stability.py` 631 | 05 は最初から5モジュールに割る |

### 既存の慣習で守るもの

1. **設定は実験ごとに独立した dataclass** (D-13)。全葉フィールドに「値を変えたら出力が変わる」テストを付ける
2. **CSV は `XXX_CSV_COLUMNS` 定数 + `csv.DictWriter` + `dataclasses.asdict(row)`** (11箇所が同一パターン。共通化はスコープ外)
3. **`experiment` → `plotting` は関数内 import** (D-53)。図は成果物 CSV の行だけを読む

### 影響範囲

新規: `metrics_detection.py` / `datasets/` (新パッケージ) / `tasks/anomaly.py` /
`experiment/anomaly*.py` ×5 / `plotting/figures_anomaly.py` / `config/anomaly05.py` /
`experiments/05_anomaly_detection/`。

既存への変更: `main.py` (+1エントリ) / `Makefile` (+2ターゲット) /
`README.md` (+1セクション +ライセンス節) / `docs/design.md` (+既定値表の行) /
`pyproject.toml` (dev グループのみ) / `tests/test_layer_boundaries.py` (+1テスト)。

**既存の `results/01..04` の成果物と既存 API シグネチャは1バイトも変えない。**

## 3. 前提・制約

### ハード制約

- サーベイ §4 の4決定 (データセット / 予測残差のみ / PA は併記のみ / 較正区間分位点) は確定。再議論しない
- **データ本体をリポジトリに含めない**。`data/` は `.gitignore:169-171` で除外済み
- 既存 API シグネチャ変更禁止。`ExperimentConfig` / `TaskData` / `metrics.py` の既存関数に手を入れない
- `results/01..04/` の成果物はバイト不変
- **pytest はネットワークに触れない**。CI は `make ci` = lock-check + ruff + mypy strict + pytest
- CPU のみ。`make figures-05` の予算 < 900 秒 (04 は実測 220 秒 / 予算 900 秒)
- Python 3.12+ / 型注釈必須 / `Any` 禁止 / `print()` 禁止 (ruff T20)

### ソフト制約

- 出力先は `results/05_anomaly_detection/`
  (D-51 が 04 で同じ変更を済ませており、`test_experiment_registry_has_unique_default_out_dirs` が一意性を機械的に要求する)
- 実データ源は MGAB を既定、UCR は**サブセット (5〜10系列)**。250系列全部は予算に入らない
- 系列は `dataset.max_length` で打ち切ってよい
- `n_replicates` は予算超過時に落としてよい唯一の値

### 設計判断への回答

**(1) AUPRC は自前実装。scikit-learn は dev グループのテストオラクルとしてのみ追加する。**

| 案 | 利点 | 欠点 |
|---|---|---|
| A. 自前のみ | 実行時依存ゼロ (D-10 の規律)。**指標の定義そのものが記事の主題**なので、階段和の式がリポジトリ内に読める意味が大きい | 正しさの担保がテスト設計に全依存 |
| B. sklearn を実行時依存に | `average_precision_score` が正本 | scikit-learn + joblib + threadpoolctl が実行時に増える。連載の「依存最小」の主張が崩れる |
| **C. 自前 + sklearn を dev のみ (推奨)** | 実行時依存は増えない。guard_test が**ランダム入力に対する sklearn との厳密一致**という最強のオラクルになる | dev の CI インストール時間が数十秒増える。`uv.lock` が動く |

→ **C を推奨**。実行時 `dependencies` に scikit-learn が入っていないことを検査するテストを guard に置く (D-62)。
C 却下時の退避は A + 手計算リテラル3ケース + 単調変換不変性 + 「台形則なら値がずれる凸ケース」。

**(2) 外部データ取得は `tasks/` に置かない。新パッケージ `datasets/` に閉じる。**

```
datasets/            ネットワーク・キャッシュ・SHA256・ライセンス表記 (I/O を持つ唯一の場所)
  fetch.py           URL -> data/ へのダウンロードと検証
  manifests/*.csv    ファイル名 + SHA256 + 系列メタ (リポジトリにコミット)
  mgab.py / ucr.py   ファイル -> AnomalySeries への読み取り
tasks/anomaly.py     純関数のみ。合成データ生成 + AnomalySeries の定義 + 前処理
```

依存の向きは **`datasets` → `tasks` の一方向**。これは `config` → `diagnostics` (D-12) と同じ形。

**(3) 前処理の共通化は `Standardizer` の形をそのまま踏襲する。**

`tasks/anomaly.py` に `AnomalyPreprocessor` (frozen dataclass) を置き、
係数を作れる場所を `from_training_prefix(series, n_steps, normalize)` **1本**に閉じる。
実験層は**1レプリケート = 1インスタンス**を作り、全手法・全区間に配る。
手法ごとに再推定する経路が構造上書けない。
`normalize` は `"zscore" | "minmax" | "robust" | "none"` の4値で、未対応値は `ValueError`。

**(4) 実験モジュールは最初から5本に割る** (`freerun.py` 1620行の再発防止):

| モジュール | 責務 | 行数予算 |
|---|---|---|
| `experiment/anomaly_score.py` | スコア構成器6種。入口は1つ (`build_score`) | 350 |
| `experiment/anomaly_threshold.py` | 較正区間分位点 + テスト側最適化 (参考値) + 閾値掃引 (5-B) | 300 |
| `experiment/anomaly.py` | 5-A の1条件 = (系列 × 手法 × レプリケート) → 行 | 450 |
| `experiment/anomaly_sweep.py` | 5-C (プロトコル掃引 + 順位入替) と 5-D (N 掃引) | 400 |
| `experiment/anomaly_pipeline.py` | `ANOMALY_ARTIFACTS` / `run_and_report_anomaly` / meta.json / 図 | 350 |

**上限 600行/ファイルをテストで固定する** (D-63)。

**(5) ライセンス表記**はトップレベル `README.md` の「実験05」セクション +
専用の「データセットのライセンスと取得手順」小節に書く。
選定基準 (要件書 設計判断5) は `docs/design.md` に §13 として書く。
`datasets/manifests/` の各 CSV 先頭に出典 URL・ライセンス・引用要求をコメント行で持たせ、
README とマニフェストの**ライセンス文字列が一致すること**をテストで固定する。

### 設定ローダの制約から来る帰結

`_common.py` は `dict` を受理しない。したがって **SHA256 表を YAML に書けない**。
マニフェストは `datasets/manifests/{mgab,ucr}.csv` (リポジトリにコミット) とし、
YAML には `dataset.series: tuple[str, ...]` だけを置く。
キャッシュ先も YAML に置かない (値を変えても出力が変わらない死んだフィールドになる) —
モジュール定数 `DEFAULT_DATA_DIR = Path("data/05_anomaly")` + `run_05.py --data-dir` にする。

## 4. タスク分解

- [ ] **T1: 検知指標層 `metrics_detection.py`** — 想定所要 **M**
  - 何をするか: AP (階段和) / PR 曲線点列 / 点単位 P・R・F1 / PA-F1 / PA%K /
    固定誤報率の分位点閾値 / `is_ignored` マスク適用を numpy だけで実装。`metrics.py` は触らない
  - 触るファイル: `src/rc_basics_lab/metrics_detection.py` (新規) / `tests/test_metrics_detection.py` (新規) /
    `pyproject.toml` (dev に scikit-learn) / `.claude/decisions.yaml` (D-54, D-62)
  - 受け入れ基準:
    1. `average_precision(labels, scores)` が **ランダム入力1000ケース (n=50〜500、異常率 1〜20%、同順位あり) で
       `sklearn.metrics.average_precision_score` と `rtol=1e-12` で一致**
       (`::test_matches_scikit_learn_average_precision_on_random_inputs`)
    2. **台形則との差が出る具体例で、台形則の値の方が大きいことを実測して固定**
       (`::test_average_precision_is_the_step_sum_not_the_trapezoid`)
    3. 一様乱数スコアの AP が異常率に収束する (n=200000、|AP − 異常率| < 0.01)
    4. **`pa_f1` は一様乱数スコアの PA-F1 を同時に返す型でしか取得できない**
       (`PointAdjustReport(pa_f1, pa_f1_random, k)` を返し、`pa_f1` 単独を返す公開関数を作らない)。
       Kim et al. の SWaT 相当の合成条件 (異常区間長 100、異常率 5%) で `pa_f1_random > 0.9` を実測
    5. `mypy strict` / `ruff` green、ネットワーク・ファイル I/O を1行も持たない
  - 実装メモ: 同順位の扱い (同スコアは1つの閾値に畳む) を sklearn と一致させること

- [ ] **T2: データ層 `datasets/` + `tasks/anomaly.py` + ライセンス文書** — 想定所要 **L**
  - 何をするか: (a) `AnomalySeries` dataclass (`values (T,1)` / `labels (T,)` bool / `ignore (T,)` bool /
    `train_end: int` / `name` / `params`) と `AnomalyPreprocessor` を `tasks/anomaly.py` に。
    (b) 合成源 `generate_synthetic_anomalies(cfg, rng)` — 既存 `generate_mackey_glass` に委譲し、
    MGAB と同じ「値と微分が一致する2点でセグメントを切って縫合する」手続きで異常を挿入 (再実装しない)。
    (c) `datasets/fetch.py` (DL + SHA256 照合 + キャッシュ) と `datasets/mgab.py` / `datasets/ucr.py`
    (UCR はファイル名 `NNN_UCR_Anomaly_{name}_{train_end}_{start}_{end}.txt` から区間ラベルを復元)。
    (d) `datasets/manifests/*.csv`。(e) README のライセンス節 + `make data-05`
  - 受け入れ基準:
    1. **SHA256 不一致のファイルを掴ませると例外**になり、キャッシュに残らない
       (`::test_download_is_rejected_when_the_sha256_does_not_match`。HTTP 部分を差し替えたローカル fixture、ネットワーク不使用)
    2. **`tasks/` と `metrics_detection.py` が `urllib` / `requests` / `socket` / `open` / `pathlib` の I/O を
       module-level でも関数内でも持たない**ことを AST 走査で機械検査
       (`tests/test_layer_boundaries.py::test_tasks_and_metrics_never_perform_io`)
    3. **全テストがネットワーク断でも green**。実データ源のテストはキャッシュが無ければ skip
       (`::test_default_source_needs_no_network`)
    4. `AnomalyPreprocessor.from_training_prefix` **以外に係数を作る経路が存在しない**ことを
       `normalize` 4値 × 「係数がテスト区間から再推定されていない」の両方で実測
       (`::test_all_methods_share_one_preprocessor_fitted_on_training_prefix`)
    5. 合成源が MGAB と同じ構造の系列を返す: 異常が指定個数、`ignore` が異常の前後に付き、
       `values` が有限、`train_end` より手前に異常が1つも無い
    6. `README.md` の「データセットのライセンスと取得手順」に MGAB (CC0-1.0 / GitHub / DOI) と
       UCR (**ライセンス未指定・再配布可否不明・本体は同梱しない**) が書かれ、
       `datasets/manifests/*.csv` のライセンス文字列と一致することをテストで固定
  - 実装メモ: UCR ZIP は 184 MB。マニフェストには**サブセットのファイル名と個別 SHA256 のみ**を書き、
    ZIP 全体の SHA256 も併記する

- [ ] **T3: 実験5-A / 5-B (対照込みの検知性能比較と閾値)** — 想定所要 **L**
  - 何をするか: `config/anomaly05.py` (`Anomaly05Config`)、`experiment/anomaly_score.py` (6スコア構成器)、
    `experiment/anomaly_threshold.py`、`experiment/anomaly.py` (行 dataclass + CSV 列定数 + 1レプリケートの実行)。
    3分割は `experiment/split.py` を使い、`compute_t0` で全手法の行を揃える (D-05)
  - 受け入れ基準:
    1. **6系統が、同一の `AnomalyPreprocessor` インスタンスと同一の行 index で評価される**
       (`::test_all_methods_share_identical_rows_and_preprocessor`)
    2. **一様乱数と入力ノルムを成果物から外せない**: `ANOMALY_METHODS` に必ず含まれ、設定から除外できない
       (`::test_random_and_input_norm_controls_are_always_present`)
    3. **閾値がテストラベルを参照せずに決まる**: 較正区間のラベルを全反転させても運用閾値が1ビットも変わらず、
       テスト区間のラベルを全反転させると `f1_calibrated` は変わるが閾値は変わらない
       (`::test_operating_threshold_is_calibrated_without_test_labels`)
    4. **PA-F1 の列は `pa_f1_random` 列と同時にしか CSV に現れない**
       (`::test_point_adjust_is_never_reported_without_the_random_control`)
    5. `f1_test_optimal` は `f1_calibrated` と**別列**で、既定設定において全行で `f1_test_optimal >= f1_calibrated`
    6. 主指標列 `auprc` が **point-adjust を一切通していない**ことを、乱数スコアで実測
       (`auprc ≈ 異常率` かつ `pa_f1 > 0.9`)
    7. モジュールいずれも 600行以下 (`::test_anomaly_modules_stay_under_the_line_budget`)

- [ ] **T4: 実験5-C (プロトコル感度) / 5-D (N と性能)** — 想定所要 **M**
  - 何をするか: `experiment/anomaly_sweep.py`。5-C は `normalize × input_window × score_smoothing` の格子で
    手法順位を出し、**順位入替の有無を数値化** (Kendall tau と基準条件からの順位変化件数)。
    5-D は `n_units_grid` を掃引し、**基準 N の AUPRC の 90% を初めて割る N** を特定
  - 受け入れ基準:
    1. **5-C の格子上で `preprocess` の既定値と一致する点の行が、5-A の対応行と厳密一致する**
       (`::test_sweep_reproduces_the_headline_condition_exactly`)。前処理が2実装に割れる経路を塞ぐ
    2. 順位入替の指標が**行として**存在する (`rank_changed: bool` / `kendall_tau: float`)。図は CSV の行だけを読む
    3. 5-D の劣化点 `n_units_at_90pct` が meta.json に記録され、N 格子の端が選ばれた場合は
       **そのことが分かる値** (`nan` ではなく端の値 + `saturated: bool`) になる
    4. `size_sweep.n_units_grid` を変えると 5-D の行数と `n_units_at_90pct` が変わる
    5. 掃引の全条件が単一の `AnomalyPreprocessor` 生成経路を通る

- [ ] **T5: パイプライン・図5枚・配線・文書** — 想定所要 **L**
  - 何をするか: `experiment/anomaly_pipeline.py` (`ANOMALY_ARTIFACTS` / `run_and_report_anomaly` /
    meta.json に `wall_time_breakdown`)、`plotting/figures_anomaly.py` (図5枚)、
    `experiments/05_anomaly_detection/{run_05.py,config.yaml}`、`main.py` の `EXPERIMENTS["05"]`、
    `Makefile` の `figures-05` / `data-05`、`README.md` の実験05セクション、
    `docs/design.md` §13 (選定基準) + §9 既定値表への追記、`tests/test_config_wiring_anomaly.py` の全フィールド被覆
  - 受け入れ基準:
    1. `make figures-05` が **CSV 5枚 + 図5枚 + meta.json** を `results/05_anomaly_detection/` に出し、
       `ANOMALY_ARTIFACTS` の宣言と実体が一致する
    2. **`experiment/anomaly*.py` が `plotting` を module-level import しない** (D-53)
    3. **`Anomaly05Config` の全葉フィールドが `test_each_parameter_changes_output` 同型のテストで被覆される**。
       `test_all_anomaly_config_fields_are_covered` が未登録フィールドで赤になる
    4. 図5枚が CJK フォント無しでも英語ラベルで生成される (D-10)
    5. `make figures-05` の実測 wall time が meta.json に区間別で残り、**合計 < 900 秒**
    6. README の数値表が `results/05_anomaly_detection/anomaly_summary.csv` と一致することを機械検査
    7. `docs/design.md` §9 の既定値表に 05 の行が**コード上の出どころ付き**で追加され、
       既存の `test_design_doc.py` の照合を通る
    8. `uv run pytest -q` が green (ベースライン 964 passed から**減らない**)

## 5. 評価軸 (Check フェーズへ)

### 機能観点

`make data-05 && make figures-05` を新規クローン相当 (`data/` を空にした状態) で1回通し、成果物11個が出る。
要件書の受け入れ条件1〜7 を成果物だけで判定する:

- 条件1 → `anomaly.csv` の `preprocessor_id` 列と `t0` 列が (系列, レプリケート) 内で単一値
- 条件3 → `f1_test_optimal - f1_calibrated` の分布が meta.json に
- 条件4 → `anomaly_protocol.csv` の `rank_changed` の件数が meta.json に
- 条件5 → meta.json の `n_units_at_90pct`

### 性能観点

| 区間 | 予算 | 計測 |
|---|---|---|
| 5-A (系列 × 6手法 × レプリケート) | < 400 s | meta.json `wall_time_breakdown.headline_s` |
| 5-B (閾値掃引) | < 60 s | `.threshold_s` |
| 5-C (プロトコル掃引) | < 250 s | `.protocol_s` |
| 5-D (N 掃引) | < 150 s | `.size_s` |
| 図5枚 | < 20 s | `.figures_s` |
| 合計 | **< 900 s** | `wall_time_s` |
| pytest 全体 | ベースライン +30 s 以内 (現状 51 s) | `pytest --durations=10` |
| ピーク RSS | < 4 GB | 確保軸の事前検査 |

**確保軸** (確保する前に検査する): `max_length × n_units`、`sweep 条件数 × n_replicates`、
`PR 曲線の点数 × 手法数`。

### 安全性観点

- 既存 `results/01..04/` がバイト不変
- 既存テスト964件が1件も落ちない
- ダウンロード: HTTPS のみ / リダイレクト追随の上限 / **サイズ上限** (ZIP 200 MB) /
  展開先のパストラバーサル検査 (`zipfile` の member 名検証) / タイムアウト。`data/` 以外に書かない
- ライセンス: UCR のデータ本体がリポジトリに入っていないこと
- 機密情報なし (公開データのみ、ログに URL 以外を出さない)

### テスト観点

新規: `test_metrics_detection.py` (約25) / `test_tasks_anomaly.py` (約15) /
`test_datasets_anomaly.py` (約12) / `test_experiment_anomaly.py` (約20) /
`test_experiment_anomaly_sweep.py` (約10) / `test_config_wiring_anomaly.py` (葉フィールド数 + 3) /
`test_anomaly_pipeline.py` (約8) / `test_plotting_anomaly.py` (約6)。

既存へ追加: `test_layer_boundaries.py` +2 / `test_main.py` +1。

### 有効性観点 — 設定 YAML の全フィールドと「値を変えたら変わる出力」

`test_config_wiring_anomaly.py::test_each_parameter_changes_output` の parametrize 一覧
(これが `leaf_paths(Anomaly05Config)` と完全一致することを `test_all_anomaly_config_fields_are_covered` が要求する):

| フィールド | channel | 変わる出力 (判定の観測点) |
|---|---|---|
| `name` | meta | meta.json の `config.name` のみ。結果行は不変 |
| `dataset.source` | rows | 系列そのもの (`anomaly.csv` の `dataset` 列と全指標) |
| `dataset.series` | rows | 行数 (系列数) |
| `dataset.max_length` | rows | `n_train` / `n_calibration` / `n_test` |
| `dataset.train_ratio` | rows | `n_train` |
| `dataset.calibration_ratio` | rows | `n_calibration` (かつ `n_train` 不変を別テストで切り分け) |
| `synthetic.length` | rows | 合成源の系列長 → `n_test` |
| `synthetic.n_anomalies` | rows | `anomaly_rate` / `auprc_random` |
| `synthetic.segment_length` | rows | `ignore` の点数 → 評価点数 |
| `preprocess.normalize` | rows | 全指標 (かつ 5-C の対応格子点と一致) |
| `preprocess.standardize_steps` | rows | 前処理係数 → 全指標 |
| `preprocess.input_window` | rows | `t0` (D-05) と全手法の指標 |
| `preprocess.score_smoothing` | rows | `auprc` (平滑化窓1と16で必ず変わる) |
| `reservoir.n_units` | rows | ESN 系統の `auprc` のみ (対照は不変を要求) |
| `reservoir.spectral_radius` | rows | ESN 系統の `auprc` |
| `reservoir.leak_rate` | rows | ESN 系統の `auprc` |
| `reservoir.input_scale` | rows | ESN 系統の `auprc` |
| `reservoir.density` | rows | ESN 系統の `auprc` |
| `reservoir.washout` | rows | `t0` → 全手法の評価行 |
| `reservoir.n_replicates` | rows | 行数 |
| `ridge.alpha_grid` | rows | `selected_alpha` 列 (D-04: 全手法が同一格子) |
| `threshold.target_false_alarm_rate` | rows | `threshold` / `far_test` / `f1_calibrated` (`auprc` は**不変**を要求 = 閾値非依存指標であることの実測) |
| `threshold.report_test_optimal` | rows | `f1_test_optimal` 列の有無 |
| `threshold.sweep_points` | rows | `anomaly_threshold.csv` の行数 |
| `evaluation.report_point_adjust` | rows | `pa_f1` と `pa_f1_random` の**2列同時**の有無 |
| `evaluation.pa_k_grid` | rows | PA%K の行数 |
| `evaluation.ignore_transition` | rows | 評価点数 → `auprc` |
| `protocol_sweep.normalize_grid` | rows | `anomaly_protocol.csv` の行数 |
| `protocol_sweep.input_window_grid` | rows | 同上 |
| `protocol_sweep.score_smoothing_grid` | rows | 同上 |
| `size_sweep.n_units_grid` | rows | `anomaly_size.csv` の行数 + `n_units_at_90pct` |
| `seeds.reservoir` | rows | ESN 系統のみ変わる (対照は不変) |
| `seeds.task` | rows | 合成源の系列 |
| `seeds.split` | rows | `split_offset` |
| `seeds.control` | rows | **一様乱数対照の `auprc` のみ**変わる (他手法は不変) |

**特に落としてはいけない4つ** (この実験を丸ごと無意味にする配線漏れ):
`preprocess.score_smoothing` (平滑化が効いていない) / `threshold.target_false_alarm_rate` (閾値が効いていない) /
`seeds.control` (乱数対照が固定値になっている) / `size_sweep.n_units_grid` (5-D が同じ N を回している)。

## 6. 意図的な決定 (`.claude/decisions.yaml` に追記)

id は既存の最大 D-53 の続き。guard_test は各タスクの受け入れ基準で追加する。

- **D-54**: AUPRC は average precision (階段和) で計算する。台形則・線形補間を禁止する
  - 根拠: sklearn 公式ドキュメント「not interpolated ... can be too optimistic」。
    台形則は補間の分だけ楽観バイアスが乗るが、値は 0〜1 に収まり曲線も滑らかなので**図でも CSV でも壊れて見えない**
  - guard_test: `tests/test_metrics_detection.py::test_average_precision_is_the_step_sum_not_the_trapezoid`
- **D-55**: point-adjust を既定にしない。PA-F1 を報告する場合は一様乱数の PA-F1 を必ず並べる。
  `pa_f1` 単独を返す公開関数・単独の CSV 列を作らない
  - 根拠: Kim et al. AAAI 2022 —— 一様乱数の F1_PA が5データセット中4つで SOTA を上回る (SWaT 0.969 vs GDN 0.935)。
    PA-F1 は「高い値が出る」形でしか壊れないため、単独報告された瞬間に読者も実装者も検出できない
  - guard_test: `tests/test_experiment_anomaly.py::test_point_adjust_is_never_reported_without_the_random_control`
- **D-56**: 運用閾値は較正区間のスコア分位点で決め、テスト区間では固定する。
  テスト側最適化の結果は `f1_test_optimal` という別列にのみ出す。**閾値決定関数はテスト区間のラベルを引数に取らない**
  - 根拠: 要件書 設計判断3 / 実験5-B の主題。テスト側で閾値を選ぶと誰でも高性能に見えるが、
    これは「良い結果」として出るのでレビューで止まらない。引数から外せば混入が型検査で書けなくなる
  - guard_test: `tests/test_experiment_anomaly.py::test_operating_threshold_is_calibrated_without_test_labels`
- **D-57**: 前処理は手法間で完全に共通化する。係数は `AnomalyPreprocessor.from_training_prefix` で
  訓練区間の先頭から推定した1組だけを作り、値として全手法・全区間へ配る
  - 根拠: 要件書 設計判断4。D-41 と同型。区間ごとに推定し直すと「当てられていない区間でも
    平均・分散が揃うため予測が当たって見える」壊れ方をする。**異常検知ではさらに悪く、
    テスト区間から推定した分散にはその区間の異常が入っているため、異常が「正常な分散」として吸収される**
  - guard_test: `tests/test_tasks_anomaly.py::test_all_methods_share_one_preprocessor_fitted_on_training_prefix`
- **D-58**: 外部データセット本体をリポジトリに含めない。取得スクリプト・ファイル名リスト・SHA256 マニフェストのみ。
  SHA256 不一致は例外にしてキャッシュにも残さない
  - 根拠: UCR は公式ページにライセンス表記が一切なく再配布可否が法的に不明。Figshare ミラーは
    寄託者が原著者でなくサイズも異なるため根拠にならない。SHA256 照合が無いと、URL 先が差し替わったとき
    「違うデータで実験して同じ数値が出ない」という形でしか気づけない
  - guard_test: `tests/test_datasets_anomaly.py::test_download_is_rejected_when_the_sha256_does_not_match`
- **D-59**: ネットワーク・ファイル I/O・ライセンス確認は `datasets/` にのみ置く。
  `tasks/` と `metrics_detection.py` は純関数層に保ち AST で機械検査する。依存は `datasets -> tasks` の一方向
  - 根拠: `tasks/__init__.py` の規律の延長。課題層に HTTP とキャッシュを足すと純関数層が
    ステートフルな I/O 層に化け、memristor-rc-lab への移植性 (D-12 が守る性質) が失われる
  - guard_test: `tests/test_layer_boundaries.py::test_tasks_and_metrics_never_perform_io`
- **D-60**: pytest はネットワークに一切触れない。既定データ源は合成とし、実データはキャッシュが無ければ skip
  - 根拠: CI がネットワーク可用性に依存すると、UCR の URL が死んだ日にリポジトリ全体が赤になり、
    実装の正しさと外部の可用性が区別できなくなる (サーベイ §2 が実測したとおり論文記載 URL の多くは既に死んでいる)
  - guard_test: `tests/test_datasets_anomaly.py::test_default_source_needs_no_network`
- **D-61**: 一様乱数スコアと入力ノルムを対照として常置する。`ANOMALY_METHODS` に必ず含み、設定から除外できない
  - 根拠: Kim et al. に倣う。一様乱数の AUPRC は異常率付近に張り付くため「PA を使っていないこと」の証拠として機能し、
    入力ノルムは「学習していないものがどこまで届くか」の下限になる。
    **対照を設定で外せるようにすると、予算が厳しい日に真っ先に外され、外した図が記事に載る**
  - guard_test: `tests/test_experiment_anomaly.py::test_random_and_input_norm_controls_are_always_present`
- **D-62**: 指標は自前実装し、scikit-learn を実行時依存に追加しない。dev グループにのみ置きテストのオラクルとして使う
  - 根拠: 実行時依存を4つに保つ (D-10 と同じ規律)。一方 AUPRC の自前実装は同順位の扱いで容易に間違え、
    間違いは「少し違う値」としてしか現れないため、独立オラクルなしの自前実装は危険
  - guard_test: `tests/test_metrics_detection.py::test_runtime_dependencies_do_not_include_scikit_learn`
- **D-63**: 05 の実験層は着手時点で5モジュールに分け、1ファイル 600 行を上限とする。上限そのものを緩めない
  - 根拠: 実測 `freerun.py` 1620行 / `capacity.py` 1204 / `attractor.py` 715 / `stability.py` 631。
    行数は「後で割る」と必ず割られない。乱暴な代理指標だが、決定論的に落ちるという一点で散文より強い
  - guard_test: `tests/test_experiment_anomaly.py::test_anomaly_modules_stay_under_the_line_budget`

**決定にしなかったもの** (ソフト制約に落とす):

- 「異常スコアは v0.1 では予測残差のみ」— **やらないことの宣言**であり、guard_test が
  「登録されていないことを確認する」という否定形にしかならず、決定として弱い。スコープに書くだけにする
- 「出力先を `results/05_anomaly_detection/` にする」— 既存の
  `test_experiment_registry_has_unique_default_out_dirs` が既に一意性を守っている

## 7. 想定リスク (これが起きたら止まって相談)

1. **UCR のライセンスが法的に不明なまま記事に載る**。取得スクリプトとハッシュしか置かない方針でも
   「引用のお願いのみ・ライセンス未指定」の状態は変わらない。README の記述が「再配布していない」以上の主張を
   しそうになったら止める。回避不能なら UCR を落として MGAB + 合成のみで記事を書く判断が要る
   (実データ軸が消えるので記事の性格が変わる = ユーザー判断)
2. **6手法すべての AUPRC が乱数対照とほぼ同じ (= 誰も検知できていない)**。
   MGAB はカオス的で人間の目にも区別できない設計なので十分あり得る。
   この場合「否定的結果も書く」規律に沿って記事は成立するが、
   5-C の順位入替も 5-D の劣化点も**測る対象が消える** (差が無いものの順位は雑音)。
   **T3 完了時に必ずチェックポイントを置く**
3. **予算超過**。ESN で 10万点 × 系列数 × レプリケート × (5-C 格子 27点 + 5-D 5点) は
   素直に書くと数時間になる。`max_length` と `n_replicates` で削れる範囲を超えたら、
   5-C の格子を全組合せから「軸ごとに1つずつ振る」(3+3+3=9点) に落とす判断が要る。
   これは実験の意味を変える (交互作用が見えなくなる) のでユーザーに相談する

## 8. 不明点

**Q1. scikit-learn を dev グループに追加してよいか。**
- (a) **推奨: 追加する** — 自前 AUPRC の guard_test を `average_precision_score` との厳密一致にできる。
  実行時依存は増えない (D-62)
- (b) 追加しない — guard は手計算リテラル3ケース + 台形則との差 + 単調変換不変性 + 乱数の期待値収束の4本立て。
  実装は通るが、同順位の扱いの誤りを見逃す確率が残る

**Q2. `make figures-05` の既定データ源とネットワーク前提。**
- (a) **推奨: `figures-05` を `data-05` に依存させ、既定は MGAB** — 「1コマンドで再生成」を満たしつつ実データ軸が立つ。
  2回目以降はキャッシュを使うのでオフライン
- (b) 既定を合成にし、実データは `figures-05-real` を別ターゲットに — 完全にオフラインで1コマンドだが、
  記事の主図が合成データになる (記事タイトル「実データで使ってみる」と食い違う)

**Q3. このサイクルを分割するか (L が3本あるため)。**
- (a) **推奨: 05a (T1 + T2 = 指標層 + データ層) / 05b (T3〜T5 = 実験と図)** — 03a/03b・04a/04b と同じ運用。
  05a の完了時点で「AUPRC が sklearn と一致し、MGAB が取得できる」という検証可能な区切りが立ち、
  リスク2 の判断も 05b の冒頭で行える
- (b) 1サイクルで通す — 文脈の連続性は保たれるが、reviewer の findings が5タスク分まとまって出るため
  fixer の一巡が長くなる
