# 05a からの申し送り (次のPR候補と 05b 着手前の決定事項)

*作成: 2026-08-20 / ブランチ `feat/rc-basics-05` / 基準 ref `a293343`*
*対象: 05a = T1 (検知指標層) + T2 (データ層)。T3〜T5 は 05b で実施する*
*出典: `.claude/tmp/findings/round-{1,2,3,4}/triage.json` (レビュー4周分、全件がファイルに残っている)*

---

## 0. 05a の到達点

| 指標 | 05 着手前 | 05a 完了時 |
|---|---|---|
| `uv run pytest -q` | 964 passed | **1132 passed** |
| `.claude/decisions.yaml` | 53 件 | **64 件** (D-54〜D-68) |
| `mypy` strict / `ruff` | green | green |
| `results/01..04/` | — | **バイト不変** |
| 実行時依存 | 4件 | **4件のまま** (scikit-learn は dev のみ) |

レビューは4周。BLOCKER は全周ゼロ。HIGH は **5 → 5 → 3 → 2** と単調減少し、すべて修正済み。

---

## 1. 05b 着手前の決定事項 (**ユーザー承認済み・2026-08-20 / 実装済み**)

**これらは「次のPR候補」ではなく、T3 の planner に渡す入力である。**

**ユーザーの決定 (3点とも確定。再議論しないこと)**:

1. **設定として死んでいるものは除去する** —— 死葉リストで固定するのではなく、
   `length` / `horizon` を持たない絞った dataclass を受ける形へ config を変える
2. **実験モジュールは `SeriesSource` Protocol にだけ依存する** —— 源の具象名で分岐しない
3. **循環 import は避ける** —— `datasets/__init__.py` から `cli` を外す (下記 (a) を採用)

### 1.0 実装済み (05b 着手前の準備リファクタ、2026-08-20)

**3点とも実装した。以下は T3 の planner にとって「決めるべきこと」ではなく
「決まっていること」である。** 決定は `.claude/decisions.yaml` の D-69〜D-72
(1決定 = 1約束 = 1 guard_test)。

| 決定 | 実装 | guard_test |
|---|---|---|
| D-69 | `SyntheticMackeyGlassConfig` (7葉、`length`/`horizon` なし) を新設し `SyntheticAnomalyConfig.mackey_glass` が受ける。既定値は `MackeyGlassConfig()` から引く | `tests/test_config_wiring_anomaly.py::test_each_synthetic_leaf_changes_the_generated_series` |
| D-70 | `MackeyGlassConfig` への変換は `SyntheticMackeyGlassConfig.to_mackey_glass(length=...)` 1箇所 | `tests/test_config_wiring_anomaly.py::test_only_to_mackey_glass_builds_the_generation_parameters` |
| D-71 | `SeriesSource` Protocol を `tasks/anomaly.py` に新設。3実装 (`SyntheticSeriesSource` / `mgab.MgabSeriesSource` / `ucr.UcrSeriesSource`) | `tests/test_series_source_protocol.py::test_each_source_satisfies_the_series_source_protocol` |
| D-72 | `datasets/__init__.py` の import と `__all__` から `cli` を外した | `tests/test_public_api_reexport.py::test_the_datasets_facade_does_not_import_the_cli` |

**`SeriesSource` の確定した形** (T3 はこれを前提に書く):

```python
@runtime_checkable
class SeriesSource(Protocol):
    def is_available(self) -> bool: ...
    def __call__(self, rng: np.random.Generator) -> AnomalySeries: ...
```

- 系列名は持たない (`Mapping` の鍵と `AnomalySeries.name` が持つ)
- 実データ源は `rng` を使わないが、呼び出し口を1つに保つために受け取る
- `is_available()` は**ネットワークに触れない** (D-60)。T3 の実験ループは
  `{key: source(rng) for key, source in sources.items() if source.is_available()}`
  の形で書ける (源の具象名で分岐しない)
- 束縛の構築: `SyntheticSeriesSource(cfg=...)` / `mgab.MgabSeriesSource(series="1")` /
  `ucr.UcrSeriesSource(filename=ucr.subset()[0])` (いずれも `data_dir` は既定値あり)

**T3 の全葉被覆について**: `leaf_paths(SyntheticAnomalyConfig)` は 11 葉
(`length` / `n_anomalies` / `segment_length` / `ignore_margin` +
`mackey_glass.{tau,beta,gamma,exponent,rk4_step,sample_interval,integration_burn_in}`)
で、**全葉が `tests/test_config_wiring_anomaly.py` で被覆済み**。
`DELEGATED_SECTIONS` による免除も死葉リストも要らない。T3 が `Anomaly05Config`
を足すときは、`synthetic.*` 節をこのテストへ委譲する形にできる
(委譲先が実在する = 04 の `DELEGATED_SECTIONS` の前提を満たす)。

実測 (2026-08-20、この準備リファクタ完了時): `uv run pytest -q` = **1157 passed
/ 39.92 秒** (着手前 1132 passed)、`mypy` strict / `ruff check` / `ruff format
--check` green、`results/01..04/` バイト不変、`make data-05` はキャッシュ有効時に
ネットワークを開かず 10 (MGAB) + 8 (UCR) 系列を照合。

以下は決定に至った背景と、実装時の具体的な注意点である (記録として残す)。

### 優先度1: `SyntheticAnomalyConfig.mackey_glass` の死んだ葉

`tasks/anomaly.py` の `generate_synthetic_anomalies` が
`dataclasses.replace(cfg.mackey_glass, length=raw_samples, horizon=1)` で
`length` と `horizon` を**必ず上書き**するため、この2葉は設定として死んでいる。

仕様 §5 は「`leaf_paths(Anomaly05Config)` の全葉が `test_each_parameter_changes_output` で
被覆される」ことを T3 に要求しているので、**この2葉は構造的にテストを赤にする**。

04 の前例 (`DELEGATED_SECTIONS` による免除) は
**「委譲先の別テストが同じ葉を被覆している」ことを前提とした免除**であり、05 には使えない
—— 01 の `test_config_wiring` が「`length` は効く」と証明している葉が、05 の経路では効かないため。
`mackey_glass.*` をまるごと免除にすると、`tau`/`beta`/`gamma` など**実際に効く7葉まで一緒に抜ける**。

**T3 の planner への指示 (解決済み)**:

> ~~`mackey_glass.*` を `DELEGATED_SECTIONS` で一括免除しないこと。…死葉リストとして固定する。~~
> **後者 (config を変える) を採用して実装済み (D-69)。** `mackey_glass.length` /
> `mackey_glass.horizon` は葉として存在しないので、免除も死葉リストも要らない。
> 7葉はすべて `tests/test_config_wiring_anomaly.py` が個別に被覆している。

### 優先度2: 系列源の共通型 (`SeriesSource` Protocol) が無い

T3 は合成源・MGAB・UCR の3つを同じ実験ループに流すため、**この時点で実装が3つ揃う**
= 早すぎる抽象化にはならない。今決めるべきは形ではなく**向き**である。

共通型を純関数層 `tasks/anomaly.py` に置いて `datasets/` が実装する
(依存 `datasets -> tasks` を維持、D-59) か、逆に `datasets` 側に置くかで、
T3 の実験モジュールがどちらを import するかが決まる。
決めずに着手すると「実験モジュールが `datasets` と `tasks` の両方の具象を if 分岐で捌く」形になり、
**D-59 の一方向依存が実験層で崩れる**。

**T3 の planner への指示 (解決済み)**:

> **実装済み (D-71)。** `SeriesSource` は `tasks/anomaly.py` にあり、3実装
> (`SyntheticSeriesSource` / `MgabSeriesSource` / `UcrSeriesSource`) が
> 引数名・引数種別・戻り値の型まで含めて満たすことを
> `tests/test_series_source_protocol.py` が実行時に固定している (mypy strict も
> `Mapping[str, SeriesSource]` として構造的に検査する)。確定した面は §1.0 を参照。
> 実験モジュールは Protocol にだけ依存し、源の具象名で分岐しないこと。

### 優先度3: `datasets/__init__.py` → `cli` → `datasets` の循環 import

`__init__.py` が `cli` を import し、`cli.py` がパッケージへ戻る辺が存在する。
現状は submodule import 機構のおかげで動くが、
**`cli` が `__init__` の再エクスポートを1つでも使い始めた瞬間に `ImportError`** になる。
T5 の CLI 配線はまさに再エクスポートを使いたくなる作業であり、
そこで初めて壊れると原因が CLI 変更に見えて循環に見えない。

**T5 着手前にどちらかを決める** → **(a) を採用して実装済み (D-72)**:
- **(a) `__init__.py` の import と `__all__` から `cli` を外す (推奨、構造で閉じる)**
- (b) `cli.py` は必ず葉モジュールを直接 import する規律をテストで固定する

(a) を採ったのは、(b) が「`cli.py` の書き方」という守り続ける規律であるのに対し、
(a) は辺そのものを消すため守る対象が残らないため。除外は
`tests/test_public_api_reexport.py` の `NOT_ON_THE_FACADE` (現在 `datasets.cli`
の1件) に明示してある。**T5 は `rc_basics_lab.datasets.cli` / `__main__` を
直接呼ぶこと** —— `from rc_basics_lab.datasets import cli` は解決できない。

---

## 2. 次のPR候補 (MEDIUM)

いずれも 05a のスコープ外として意図的に未対応。**実験結果には影響しない。**

| # | 場所 | 内容 |
|---|---|---|
| M1 | `tasks/anomaly.py` の `_find_cut_search_cells` docstring | `_find_cut` 本体を**行番号で参照**しており、3周連続でずれ続けている (297-304行 → 実際は347-350行)。**行番号参照をやめてシンボル参照** (`min_span`/`max_span`/`half_width`) にする |
| M2 | `datasets/fetch.py` | `_extract_member` と `_stream_to_file` が「chunk 読む → digest 更新 → sink.write」の同型ループを個別実装。共通ヘルパーへ抽出 |
| M3 | `tests/test_datasets_anomaly.py` | guard-of-guard テスト2本が約28行の AST 走査ロジックを重複。モジュールレベル関数へ抽出 |
| M4 | `datasets/fetch.py` | `dir_fd` / 一時 fd が例外経路で確実に閉じることの**回帰テストが無い** (実装は正しいことを実測済み)。`mock.patch("os.close")` で呼び出し回数を固定する |
| M5 | `datasets/fetch.py` | `_staged_write` の外に残る `mkdir` の事前条件が非局所。3つ目の呼び出し元が忘れると安全性と無関係な `FileNotFoundError` が出る |
| M6 | `datasets/fetch.py` の `commit` | `error_cls` 注入により「digest 不一致」と「実体差し替え」が呼び出し元によって別の公開例外型になる。層固有の型を送出し、呼び出し側で翻訳する |
| M7 | `datasets/mgab.py` | CSV 読み取りが1点ずつの Python ループ (10万行で実測 0.078秒 → `csv.reader` 版 0.036秒)。ホットパスではない |
| M8 | `docs/plans/rc-basics-05.md` | T2 の実測記録が round-1 以前のまま stale (テスト件数が古い) |
| M9 | `tasks/anomaly.py` の `_MAX_FIND_CUT_CELLS` docstring | 「100MB未満」と書いているが閾値付近の実測ピークは **160MB**。根拠を実測値に更新する |
| M10 | `README.md` | UCR の箇条書きに**引用文字列が無い** (MGAB はフルサイテーションを載せており非対称) |

## 3. 次のPR候補 (INFO)

- `metrics_detection.py:34` の `BoolArray` が `types.py` と二重定義 (T1 のファイルを触らない制約のため残った)
- `metrics_detection.py:102` の `PrecisionRecallCurve.__post_init__` の分岐が未被覆 (カバレッジ99%の残り1行)
- `datasets/ucr.py` の `is_available` が `RemoteFile` を組み立てるが `url` を使わない
- `_StagedSink._committed` を read-only property にして `_staged_write` から private 属性を読まない
- `manifest()` のメモ化なし (実測 27.5μs/回 × 最大25回 = 0.7ms/fetch。無視できる)
- `secrets.token_hex(16)` の `16` と `os.open(..., 0o600, ...)` を名前付き定数/キーワード引数に
- `_stream_to_file` が `_StagedSink` を前方参照している (読み順の問題のみ)
- `pyproject.toml` に `Operating System ::` classifiers を追加するか (CI は ubuntu-latest のみ、緊急性なし)

## 4. 04 からの申し送り (05a でも未着手、スコープ外)

- **CSV 書き出しの11箇所共通化**。`report.py` 2 / `capacity_pipeline.py` 2 / `freerun.py` 3 /
  `stability.py` 1 / `esp_pipeline.py` 3 が同一パターンを個別実装している。
  05 も同じパターンを踏襲する予定なので、共通化するなら 05b 完了後がよい
- `experiment/freerun.py` の3分割 (1620行)

---

## 5. セキュリティの打ち切り判断 (記録)

レビュー3周にわたり reviewer-security がより深い TOCTOU 変種を出し続けた
(ファイル差し替え → 親ディレクトリ差し替え → 中間成分差し替え)。
4周目に **reviewer-security 自身が打ち切りを推奨**している:

> 脅威モデル (`data/` に書ける同一ユーザーのプロセス = ソースも書き換えられる相手) を踏まえると
> 防御水準はこの層として妥当であり、これ以上の深追いは投資対効果が急速に落ちる —— 打ち切りを推奨する。

最後に残った「`O_NOFOLLOW` はパスの最終成分しか守らない」という穴は、
reviewer 推奨の **(a) `relative_path` を1階層に制限する**方式で塞いだ
(`resolve_under` が2階層以上を `UnsafeArchiveMemberError` にし、
manifest の実データが全行1階層であることもテストで固定)。
openat 連鎖による一般解は「攻撃者能力に対して複雑さが釣り合わない」として採らなかった。

**05b でこの層に手を入れる場合も、この判断を再議論しないこと。**

---

## 6. 今サイクルで学んだこと (運用面、`/retro` の入力候補)

1. **HIGH の大半は「実装の誤り」ではなく「rule が束ねた約束のうち測られていないもの」だった。**
   D-64 / D-65 がいずれも複数の約束を1つの rule に書き、guard_test が一部しか測っていなかった。
   D-66 / D-67 / D-68 への分割で解消。**1決定 = 1約束 = 1 guard_test** を最初から守ると周回が減る
   (`check_decisions.py` の `guard_test` が単一 node id しか解決しないという制約とも整合する)
2. **各周の修正が次の周の HIGH を生んだ。** 1周目の mkstemp 化が2周目の HIGH を、
   2周目の AST ガードが3周目の HIGH を生んでいる。
   **修正を入れたら、次周は「新しい観点」ではなく「直したところの周辺」を疑うよう
   reviewer に明示する**と収束が早い (3周目以降はこれを round-state に書いた)
3. **reviewer の報告は検証が要る。** 1周目の reviewer-performance は**存在しない関数**を指し、
   実測値が **62倍**ずれていた (100k点で「26.7秒」→ 実測 0.428秒)。
   一方2周目以降の同 reviewer は正確だった。**抜き取り検証は毎周1件では足りない可能性がある**
4. **静的 AST ガードの完全性は追い切れない。** Python でバイトを書く方法は無限にあり、
   3周連続で「まだ抜けがある」と指摘された。
   **「原理的限界を docstring に明記して追わない」線引きを決定に書く**のが正解だった
