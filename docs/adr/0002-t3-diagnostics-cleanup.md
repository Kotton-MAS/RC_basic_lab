# ADR 0002: `diagnostics/` の整理4件 —— 行合わせ・入力測度・チャンク幅の軸分離と `max_targets` の扱い (04a T3)

- **Status**: **Accepted** (2026-08-20 ユーザー確認。§9 の却下2案はいずれも**却下のまま**)
- **Date**: 2026-08-20
- **Cycle / Task**: rc-basics-04a / T3 (`docs/plans/rc-basics-04.md` §4 T3)
- **決定 ID**: **新規採番なし**。**D-24 / D-28 / D-33 / D-34 の rule 改訂**として扱う (次の空き番号は D-54)
- **関連決定**: D-01 / D-09 / D-12 / D-15 / D-23 / D-25 / D-26 / D-27 / D-32 / D-35 / D-37 / D-38 / D-49 / D-52

> **本 ADR の事実の出所**: すべて**ソース読取・成果物ファイルの読取**と、そこからの**推論**である。
> 実測 (実行) は1件も含まない。数値の見積りには「推定」と明記した。
> 実装者は §8 の変異注入リストで**実測に置き換えてから** `.claude/decisions.yaml` を更新すること。

---

## 1. Context

### 1.1 4件に共通する性質

`diagnostics/` は **3a 完了以降1行も変更されていない**。整理4件は 3a の設計に対する未着手の負債であり、
04b の実験層はこれに依存しない。したがって T3 は**純粋な整理**として切り分けられ、
その証明が `results/03_capacity/` のバイト不変である。

4件は「同じ規律を2箇所に書いた / 単位の違う2量を1つの名前で縛った」という**同型の負債**であり、
3a が F-03-1-001 (窓計算の複製) で1度潰した形が別の軸で再発している。

### 1.2 ソース読取で確認した事実 (**設計判断の土台**)

1. **`meta.json` は `dataclasses.asdict(Capacity03Config)` をそのまま持つ**。`config.ipc` ブロックに
   `basis` / `input_distribution` / `chunk_size` / `max_targets` / `max_degrees` が出ている。
   すなわち **`IpcConfig` のフィールド構成を変えると `results/` に差分が出る**
2. **`capacity.csv` は `chunk_size_mc_effective` / `chunk_size_ipc_effective` を列に持つ**。
   本番 3-B' は **84 / 83** で、**128 MiB キャップは本番で実際に発動している**
3. `IpcConfig.basis` / `input_distribution` は **YAML の葉**。設定ローダが受理するスカラ型は
   `bool` / `int` / `float` / `str` **のみ**で `Enum` は落ちる
4. `orthonormal_basis(u_lagged, degree, distribution=UNIFORM, *, basis=LEGENDRE)` は
   D-28 が「対で意味を持つ」と宣言した2値のうち**片方だけを位置引数の既定値つきで**受ける
5. **`t0 = max(ctx.washout, 最大遅延)` と「系列が短すぎます」の拒否は MC と IPC に2箇所複製されている**。
   F-03-1-001 で潰したのは `lagged` の窓式だけで、**基準点の算出そのものは複製が残っている**
6. `CapacityProblem.lagged` は `t0` と `n_samples` しか読まず `x` / `gram` に触れない。
   このため窓計算だけを検査したいテストが `_dummy_problem` で **特異な Gram ごと**構築している。
   **D-24 / D-28 の guard_test は両方ともこのダミーに依存している**
7. `chunk_size` の実効値は**3つの用途**に同じ値で使われている: (A) 実目標の solve 幅、
   (B) サロゲートの solve 幅、(C) **代表目標ブロックの確保幅**。
   (A)(B) は性能軸だが **(C) には性能上の意味が一切ない**
8. 本番の `n_surrogate_targets` は **4**、3-B' の予算列数は **83**。
   したがって代表目標ブロックは本番で**常に1ブロック4列**で `cfg.chunk_size` に依存していない
9. `heatmap_cells > cfg.max_targets` の検査は、**目標数 (本) と heatmap セル数 (セル) という
   単位の違う2量を同じ 200,000 で縛って**いる。本番最深設定は目標 **4,075本** / heatmap **800セル**で
   **49倍・250倍の余裕**がある
10. D-34 の rationale (F-03-4-010) は既に「**`max_targets` は運用者が明示的に宣言する予算の終端であり
    絶対上限は置かない**」「絶対上限を追加する判断はありうるが、それは別決定として扱う」と明文で決めている
11. **仕様 §5 表 #8 が「4-D の目標数の上限は既存 D-34 の4段を再利用。新しい上限を作らない」と指定**
12. `test_ipc_config_fields_change_output` は `dataclasses.fields(IpcConfig)` を機械列挙し、
    **全フィールドが「出力を変える」か「除外リストに載る」かのどちらか**であることを要求する。
    新しい設定フィールドは必ず除外リスト行き = **空虚な設定フィールド**
13. `_capacity.py` は**先頭 `_` の非公開モジュール**で `diagnostics/__init__.py` から再エクスポートされない。
    `CapacityProblem` / `orthonormal_basis` の署名変更は**公開 API の変更ではない**
14. 04b-D が呼ぶ接ぎ目 (`measure_capacity` / `capacity_row_from`) は
    **`_capacity` の内部構造にも `chunk_size` の導出にも触れない**

### 1.3 制約

- **`results/03_capacity/` の CSV 4本・図5枚がバイト不変** —— 唯一の証明
- T4 / T5 に踏み込まない。`DiagnosticContext` に足さない (D-01)
- `diagnostics/` は `config` / `reservoir` を import しない (D-12 / D-23)
- D-52 の命名規約。新規依存を増やさない。`Any` 禁止

---

## 2. 決定1 (D-28 の改訂): `(input_distribution, basis)` を `InputMeasure` 1値にまとめる

### 2.1 選択肢

| 観点 | **案A: 関数の引数だけを1値に** | 案B: `IpcConfig` も1値に | 案C: `Enum` 化 | 案D: 現状維持 |
|---|---|---|---|---|
| YAML スキーマ | **不変** | 変わる | 不変 | 不変 |
| **`meta.json`** | **不変** | **`config.ipc` が変わる** | 不変 | 不変 |
| 設定ローダ | 触らない | 触らない | **改修が要る** | — |
| 不正な組の落ち方 | 構築時 `ValueError` | 同左 | **構築不能** (最も強い) | 現状 |
| 主なリスク | 小 | **バイト不変の証明が崩れる** | ローダ改修が T1 に波及 | **04b が同じ罠を踏む** |

### 2.2 決定

**案A を採用する。**

1. `_capacity.py` に **`InputMeasure`** (frozen / slots) を新設。`distribution` / `basis` を持ち、
   `__post_init__` で未対応の組を **`ValueError` (メッセージに `D-28`)** にする
2. `UNIFORM_LEGENDRE` / `NORMAL_HERMITE` / `SUPPORTED_MEASURES` を置く。
   既存の `UNIFORM` 等は**そのまま残す** (`IpcConfig` の値が str のままのため)
3. `orthonormal_basis(u_lagged, degree, measure)` —— 第3引数は **既定値なし**。
   **既定値を持たせないことが本決定の実体**で、「片方だけ渡す」呼び方を**型検査で書けなくする**
4. `ipc()` は**入口で1度だけ** `InputMeasure(...)` を作る
5. `memory_capacity()` は `UNIFORM_LEGENDRE` を渡す (次数1は分布に依存しない旨を docstring に)

**`InputMeasure` の値域検証は D-09 に反しない。** D-09 は「**設定 dataclass** は純データ」であり、
`InputMeasure` は YAML から構築されない**カーネル内の値オブジェクト**である。

### 2.3 却下理由

- **案B**: `meta.json` の `config.ipc` が**構造ごと変わる**。T3 の唯一の証明手段を、整理そのものが壊す。
  **04b-1 の `Chaos04Config` の書き方にも波及**する → §9 に分離
- **案C**: 設計としては最も強いが、`_coerce_scalar` が Enum を受理しないためローダ改修が要り、
  T1 (D-49) の直後に同じ層へ手を入れることになる。加えて **D-28 の guard_test が固定している
  `ValueError` 経路が消え**、YAML の誤りが「ローダの型エラー」に化けて D-28 の意味が読めなくなる
  → **見直し条件に格上げ**
- **案D**: F-03-1-006 は 3a → 3b → 04 と**2回先送りされている**

### 2.4 バイト不変

**保てる。** `IpcConfig` のフィールド構成・既定値・YAML は不変 → `meta.json` 不変。
`psi_table` の生成は引数の**渡し方**が変わるだけで値も分岐も同一。
`capacity.csv` に `basis` 由来の列は無い。

### 2.5 見直し条件

- 設定ローダが `Enum` を受理するようになったとき → **案C** へ
- `SUPPORTED_BASIS_PAIRS` が3組以上になったとき
- 入力分布を「実測から推定する」方向へ変えたくなったとき

### 2.6 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_basis_is_orthonormal_and_mismatched_pair_raises` (**D-28 guard。既存維持**) | 正規直交 + 未対応の組が `ValueError` |
| `test_input_measure_rejects_unsupported_pairs` (新設) | **構築時点**で `ValueError` |
| `test_orthonormal_basis_requires_an_explicit_measure` (新設) | 第3引数に**既定値が無い**ことを署名で固定 |

---

## 3. 決定2 (D-34 の改訂): `max_targets` の単位は**分離しない**。縛る軸を列挙して正本にする

### 3.1 選択肢

| 観点 | **案A: 分離しない + 軸を列挙** | 案B: `max_heatmap_cells` を足す | 案C: `_MAX_HEATMAP_CELLS` を5段目に |
|---|---|---|---|
| 機能的な必要 | **無い** (49倍 / 250倍の余裕) | 無い | 無い |
| `meta.json` | **不変** | **新キー。変わる** | 不変 |
| 空虚さのリスク | 低 | **高**: 除外リスト行き確定 = 空虚な設定フィールドを1本増やす | 中 |
| 既存決定との整合 | **D-34 rationale・仕様 §5 表 #8 と一致** | 正面から矛盾 | 「別決定として扱う」と留保された案 |
| 運用者への影響 | 変わらない | **ノブが1本増える** | `max_targets` を小さくしても緩む (**現状より弱い**) |

### 3.2 決定

**案A を採用する ——「やらない」を明示的に選ぶ。** ただし**何もしない**のではなく:

1. **D-34 の rule に定義を書き直す**: `max_targets` は「目標数の上限」ではなく
   「**単位の違う複数の確保軸を1本で縛る、運用者が宣言する共有予算の終端**」である
2. **縛っている軸を列挙表として正本化する**: (a) 目標数 (b) heatmap セル数。
   軸を足すときは列挙表とパラメトライズしたテストの**両方**に足す

**「単位の食い違い」の実体は、ノブが足りないことではなく、`max_targets` が何を縛っているかが
どこにも書かれていないこと**である。reviewer が指摘に到達できたのは `ipc.py` のコメントを
読んだからで、**rule からは読めない**。列挙表 + 完全性テストはこの欠落を直接埋める。

### 3.3 バイト不変

**保てる (自明)。** 本番の検査経路は1行も変わらない。

### 3.4 見直し条件 (**数値で書く**)

- 目標数 / `max_targets` の余裕が**10倍を切ったとき** (現在 49倍)
- heatmap セル数 / `max_targets` の余裕が**10倍を切ったとき** (現在 250倍)
- `max_targets` が縛る軸が **3本以上**になったとき
- `ipc_heatmap` が成果物の配列として `results/` に直接出るようになったとき

### 3.5 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_out_of_range_config_raises` (**D-34 guard。既存維持 + heatmap 軸のケース追加**) | 1つの node id が4段の絶対上限 **+ `max_targets` が縛る2軸**をすべて固定 |
| `test_max_targets_bounded_axes_are_enumerated` (新設) | 列挙表の各軸が**独立に** `ValueError` へ到達できる |
| `test_max_targets_also_bounds_the_heatmap_cell_count` (既存維持) | 別角度からの固定 |

---

## 4. 決定3 (D-24 の改訂): `RowAlignment` を切り出し、**基準点の算出も**そこへ集約する

### 4.1 選択肢

| 観点 | 案A: 引数だけ集約 | 案B: `RowAlignment` を値として切り出す | **案C: 案B + 基準点の算出も含める** | 案D: 現状維持 |
|---|---|---|---|---|
| ダミー状態行列 | 消える | 消える | 消える | 残る |
| **`t0` の複製** | **残る** | **残る** | **消える** | 残る |
| 空虚でないことの証明 | 構造ガードのみ | 構造ガードのみ | **構造 + 挙動ガード** | — |

### 4.2 決定

**案C を採用する。** `RowAlignment` (frozen / slots) は **`t0` と `n_samples` の2つだけ**を持ち、
状態行列にも Gram にも触れない。

1. **`RowAlignment.from_series(*, n_steps, washout, max_delay)`**: 基準点を算出し
   `t0 >= n_steps` を拒否する。**MC と IPC はこの1本を呼ぶ** (現在の2箇所の複製を消す)
2. **`RowAlignment.lagged(series, delay)`**: 現 `CapacityProblem.lagged` を移す
3. `CapacityProblem` は `rows: RowAlignment` を持ち、`from_states` は
   **自分が切り出した行数と `rows.n_samples` の一致を検査する**

**この抽象が「それが無いと書けないテスト」を連れてくること**が採用の条件。
`RowAlignment(t0=20, n_samples=480)` を直接構築して窓計算を検査するテストは
`CapacityProblem` 経由では**書けない**。`_dummy_problem` は削除する。

### 4.3 却下理由

- **案A**: 「`t0` と `n_samples` は対で意味を持つ」関係が型に出ない。D-28 で対をまとめておきながら
  こちらだけ散らすのは規律として一貫しない
- **案B (基準点を集約しない)**: `RowAlignment` が「窓計算だけを持つ器」になり、
  **no-op に差し替えても挙動が変わらない** (呼び出し側で同じ式を書き直せる) ——
  **D-33 で作った「no-op に差し替えても1件も落ちない安全機構」と同じ形の抽象**になる
- **案D**: guard_test 2本が**ダミーの状態行列に依存し続ける**

### 4.4 バイト不変

**保てる。** 算出式が同一 (`max(washout, max_delay)` / `n_steps - t0`)。
窓式を移すだけで内容は同一。`RowAlignment` は成果物のどこにも出ない。
**注意**: 「系列が短すぎます」のメッセージは1本に統合されるため文言が変わる。
`match` しているテストがあれば更新する。

### 4.5 見直し条件

- 目標ごとに異なる行集合を使う診断が要るようになったとき (D-24 そのものの見直し)
- `RowAlignment` に3つ目のフィールドを足したくなったとき (責務が混ざったサイン)

### 4.6 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_all_targets_share_identical_rows` (**D-24 guard。既存維持**) | **`_dummy_problem` ではなく `RowAlignment` を直接構築するよう書き換える** |
| `test_row_alignment_needs_no_state_matrix` (新設・**空虚でないことの証明**) | **このテストは `RowAlignment` が無いと書けない** (案B/案D では収集時に落ちる) |
| `test_row_alignment_is_the_only_base_point_calculation` (新設) | `from_series` を monkeypatch して**両診断で呼ばれる**ことを確認 (複製が復活したら赤) |
| `test_capacity_problem_rejects_inconsistent_row_alignment` (新設) | 行数の合わない `RowAlignment` で `ValueError` |

---

## 5. 決定4 (D-33 の改訂): チャンク幅を**性能軸**と**確保軸**に分ける

### 5.1 3つの用途のうち1つだけが性能軸でない

| 用途 | 何を決めるか | 性能上の意味 |
|---|---|---|
| (A) 実目標の solve 幅 | 1回の solve に畳む目標列数 | **有る** (D-26) |
| (B) サロゲートの solve 幅 | 同上 | **有る** |
| (C) 代表目標ブロックの確保幅 | `picked` を一度に何列実体化するか | **無い** |

(C) が `cfg.chunk_size` に従うのが指摘の実体 —— **運用者の性能ノブが確保上限を動かしている**。

### 5.2 選択肢

| 観点 | 案A: rule に3用途を明記 | 案B: 名前だけ分ける | **案C: (C) を `cfg.chunk_size` から切り離す** |
|---|---|---|---|
| 「軸を兼ねている」は解消するか | しない | **名前だけ** | **する** |
| 空虚さのリスク | — | **高** (D-33 で1度作った形そのもの) | 低 |
| 本番の値 | 不変 | 不変 | **不変** (`min(4, 83) = 4` は現行と同じ1ブロック4列) |

### 5.3 決定

**案C を採用する。**

1. `RowAlignment` に2つのメソッドを置く (どちらも `n_samples` だけの関数):
   - **`solve_width(configured)`** —— 性能軸。`cfg.chunk_size` を上限とし 128 MiB で下げる
   - **`block_width(n_columns)`** —— 確保軸。**`cfg.chunk_size` を読まない**
2. 両者は `bounded_chunk_size` **1本の純関数**へ委譲する (128 MiB 予算を2箇所に持たない)
3. `params` に記録する `chunk_size_effective` は引き続き (A) の値とする

### 5.4 却下理由

- **案A**: 「兼ねていると書く」で答えるのは**負債を仕様化しているだけ**
- **案B**: 値が同一なら差し戻し変異で**1件も落ちない**。D-33 の rationale が記録している
  「no-op に差し替えても既存58テストが1件も落ちない空虚な安全機構」(F-03-2-018 BLOCKER) と**同じ形**

### 5.5 バイト不変 (**前提を1つ実測で確認すること**)

- (A)(B) の導出は1行も変わらない → `chunk_size_*_effective` は不変
- (C) は本番で **`min(4, 83) = 4`**、現行も1ブロック4列 → 分割が同一で丸め順序も変わらない
- **実装者が着手前に実測する前提**: 03 の全条件で **`n_surrogate_targets` (=4) <= `budget_columns`**。
  成立しない条件が1つでもあれば分割が変わり **`mc_threshold` / `ipc_threshold_degree{d}` の
  最終ビットが動きうる**
- **`test_chunk_size_does_not_change_results` は `rel=1e-10` の近似比較であり、
  ビット一致を測っていない** —— docstring の主張と assert が食い違っている (本 ADR の発見)
- 成立しない場合は**この項目だけを 04b 送り**にする

### 5.6 見直し条件

- `n_surrogate_targets` に上限を置く決定が入ったとき
- `_MAX_CHUNK_BYTES` を動かすとき (**そのときこそ2軸に分けた効果が出る**)
- 代表目標のブロック化に性能上の意味が生まれたとき (分離の理由が消える)

### 5.7 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_params_record_configured_and_effective_chunk_size_when_capped` (**D-33 guard。既存維持**) | (A) が下げられ `params` に両方が残る |
| `test_representative_blocks_do_not_follow_chunk_size` (新設・**空虚でないことの証明**) | `chunk_size=1` / `n_surrogate_targets=4` でブロック数が **1** (旧実装なら 4) |
| `test_block_width_is_capped_by_the_memory_budget` (新設) | キャップを外す変異で赤 |
| `test_solve_and_block_widths_share_one_budget_function` (新設・任意) | 128 MiB 予算が2箇所に増えていない |

---

## 6. 改訂後の rule 本文 (案)

### D-24 (改訂)

> MC / IPC は全目標を同一の行集合で回帰する。行合わせの担い手は `RowAlignment` (`t0` と `n_samples`
> **だけ**を持つ値) **1つ**であり、(i) `t0 = max(ctx.washout, その診断の最大遅延)` の算出、
> (ii) `t0 >= T` の拒否、(iii) 遅延窓の切り出しの3つを、診断ごとに複製せずここ1箇所に置く。
> `CapacityProblem` は `RowAlignment` を内包し、構築時に行数の一致を検査する。
> 行合わせの検査は**状態行列を経由せずに書けること**。

### D-28 (改訂)

> `(input_distribution, basis)` は**対でのみ意味を持つ**ので、対を1つの値 `InputMeasure` にまとめ、
> `orthonormal_basis` は**既定値なしの第3引数**としてこれを受け取る (片方だけを渡す呼び方を
> 型検査で書けなくする)。未対応の組は `InputMeasure` の**構築時点**で `ValueError` にする。
> 設定層は YAML と `meta.json` の面を変えないため**2つの文字列フィールドのまま保ち**、
> `ipc()` が入口で1度だけ畳む。

### D-33 (改訂)

> チャンク幅は**2つの軸**を持ち、名前と導出元を分ける。**(i) 性能軸 `solve_width(configured)`**:
> `cfg.chunk_size` を上限とし 128 MiB を超える場合に限り下げてよい。結果を1ビットも変えず、
> `params` に設定値と実効値の両方を記録する。**(ii) 確保軸 `block_width(n_columns)`**:
> **`cfg.chunk_size` を読まない** —— 運用者の性能ノブが確保上限を動かしてはならない。
> 両軸とも `bounded_chunk_size` **1本の純関数**へ委譲する。

### D-34 (改訂)

> IPC の確保・組合せ計算量は、**上書き不能な絶対上限4段**と、**運用者が宣言する共有予算
> `max_targets` 1本**で縛る。**`max_targets` は「目標数の上限」ではなく、単位の違う複数の確保軸を
> 1本で縛る共有予算の終端である。** これは意図的であり、現在縛る軸は **(a) 目標数
> (b) `ipc_heatmap` のセル数**の2本で、**この列挙が正本**である。軸ごとに別の設定フィールドへ
> 分けることは行わない —— 本番で目標が上限の 1/49・heatmap が 1/250 と機能的な必要が無く、
> 運用者が正しく設定すべきノブを増やすため。**どちらかの余裕が10倍を切ったら分離を再検討する。**

---

## 7. 新設する抽象が空虚でないことをどう保証するか (**§7 リスク3 への回答**)

3a では **D-33 で「no-op に差し替えても1件も落ちない安全機構」を実際に作った**。
同じ失敗を繰り返さないため、新設する抽象それぞれに**3種類の変異**を課す。

| 抽象 | (a) no-op 変異 | (b) 旧実装差し戻し | (c) 「それが無いと書けないテスト」 |
|---|---|---|---|
| `InputMeasure` | 検査を消す → D-28 guard が赤 | 既定値を戻す → 署名テストが赤 (+ mypy) | 対を1値として受けることを署名で固定 |
| `RowAlignment` | `max()` を `washout` に → D-24 guard と MC 側が赤 | `lagged` へ戻す → **収集時**に赤 | **状態行列を作らずに窓計算を検査する** |
| `block_width` | 予算キャップを外す → 赤 | `solve_width` に戻す → 赤 | **確保幅が `chunk_size` に依存しないことをブロック数で数える** |
| `max_targets` の軸列挙 | 列挙表から1軸削る → 該当ケースが赤 | heatmap 検査を消す → 赤 | どの軸が縛られているかを機械で数える |

**規律**: 新しい抽象は「**それが無いと書けないテストを1本連れてくる**」こと。
連れてこられない抽象は名前を付け替えただけであり導入しない。
**(a)(b) の変異を1本ずつ実際に注入し、落ちたテスト名と件数を実装メモに残す**。
**1件でも落とせないものが出たら止まって相談する。**

---

## 8. 変異注入リスト

| # | 対象 | 変異 | 期待 |
|---|---|---|---|
| 1 | `InputMeasure` | 対の検査を `pass` に | D-28 guard / 構築時テスト |
| 2 | `orthonormal_basis` | 第3引数に既定値を戻す | 署名テスト (+ mypy) |
| 3 | D-28 | 分岐を `basis` から `distribution` に | 直交性側 |
| 4 | `RowAlignment` | `max(washout, max_delay)` を `washout` に | D-24 guard / MC 側 |
| 5 | D-24 | `start = t0 - delay` を `t0 + delay` に | **2件が別々に落ちること** |
| 6 | D-24 | `RowAlignment` を消し `lagged` へ戻す | **収集時**に赤 |
| 7 | D-24 | `from_states` の整合検査を消す | 整合テスト |
| 8 | D-24 | MC 側だけ `from_series` を経由せず手書きに戻す | **複製の復活を直接検出** |
| 9 | D-33 | `solve_width` を `return configured` に | D-33 guard / 純関数テスト |
| 10 | D-33 | `block_width` を `solve_width` に戻す | ブロック数テスト |
| 11 | D-33 | `block_width` のキャップを外す | 予算テスト |
| 12 | D-33 | `bounded_chunk_size` を2本に複製し片方の予算を変える | 共有テスト |
| 13 | D-34 | heatmap 検査を消す | 2件 |
| 14 | D-34 | 列挙表から heatmap 軸を削る | 列挙テスト |
| 15 | 参考 | `IpcConfig` にフィールドを1本足す | `test_ipc_config_fields_change_output` が赤 + `meta.json` に新キー (**バイト不変が崩れることの実測記録として1度だけ確認し revert**) |

**バイト不変の確認**: `make figures-03` を回し CSV 4本・図5枚を個別 SHA-256 で突き合わせる。
`meta.json` は `commit` / `timestamp_utc` / `wall_time_*` が必ず動くので、
**`config` ブロックの構造差が0であること**を別途 JSON レベルで確認する。

---

## 9. 確認済み (2026-08-20)

> **9-1 / 9-2 とも却下のままで確定。** `IpcConfig` のフィールド構成・YAML スキーマ・
> `meta.json` の `config.ipc` ブロックは**一切変更しない**。

### 確認時の判断材料 (記録)

> **本 ADR の推奨案には、承認が必要な変更は1件も含まれない。**
> `_capacity.py` は非公開モジュールで再エクスポートされないため、署名変更は公開 API の変更ではない。
> `IpcConfig` / YAML / `measure_capacity` / `capacity_row_from` はいずれも**不変**。

以下の2件は**却下した案**であり、**「却下でよい」ことの確認**をお願いしたい。

| # | 変更 | 却下した理由 |
|---|---|---|
| **9-1** | `IpcConfig.basis` / `input_distribution` を `measure: InputMeasure` に置換 | T3 の唯一の証明手段 (`results/` が動かない) を整理そのものが壊す。04b-1 の書き方にも波及 |
| **9-2** | `IpcConfig` に `max_heatmap_cells` を追加 | 本番の余裕 49倍 / 250倍で機能的必要が無く、**空虚な設定フィールドを1本増やす**。仕様 §5 表 #8 と D-34 rationale の両方に反する |

---

## 10. 04b への影響

### 10.1 04b-1 (T4)

1. **`measure_capacity` / `capacity_row_from` の署名は1文字も変わらない**
2. **`IpcConfig` のフィールド構成が不変**なので `Chaos04Config` の書き方も 03 と同一でよい
3. `diagnostics/lyapunov.py` は `_capacity` を使わないので影響を受けない。**D-52 の命名規約は適用**
4. **新しい抽象を写経しないこと**: `RowAlignment` は遅延目標を作る診断のための値であり、
   自走の窓計算に転用しない (自走は逐次計算で行合わせの問題を持たない)

### 10.2 04b-2 (T5)

1. **4-D は仕様 §5 表 #8 のとおり「既存 D-34 の4段を再利用。新しい上限を作らない」**
2. `n_surrogate_targets` を 83 より大きくすると確保軸が実際に効き始める。結果は変わらないが
   **確保軸の実効値は `params` に記録されない**
3. **`results/03_capacity/` は 04b でも1バイトも変えない**

---

## 11. 実装順序と実装者への注意

**推奨順序**: (1) `RowAlignment` → (2) `block_width` → (3) `InputMeasure` → (4) D-34 の rule と列挙

**歩くスケルトン**: **`RowAlignment` を新設し `test_row_alignment_needs_no_state_matrix` を1本書いて
`_dummy_problem` を消す**。ここまでで「状態行列を作らずに行合わせを検査できる」が成立する。

**落とし穴**

- **`test_chunk_size_does_not_change_results` はビット一致を測っていない** (`rel=1e-10`)。
  docstring の主張と assert が食い違っている。**着手前に §5.5 の前提を本番全条件で実測**すること。
  成立しない条件があれば**決定4 だけを 04b 送りにする**
- **`meta.json` の `config` ブロックは `IpcConfig` の `asdict` そのもの**。触らないこと
- `docs/design.md` の更新箇所: `CapacityProblem.lagged` → `RowAlignment` / 既定値表 / D-33 の記述
- **`.claude/decisions.yaml` の D-26 の rule が `chunk_size_effective` を名指ししている**。
  名前が変わるなら D-26 も追随させる (F-03-3-009 で1度 rule 間の自己矛盾を起こしている)
- D-24 / D-28 の guard_test は**両方とも `_dummy_problem` に依存している**

---

## 12. Consequences

**良くなること**

- **窓計算だけを検査するテストがダミーの状態行列と特異な Gram を作らなくなる**
- **`t0` の複製 (MC / IPC の2箇所) が消える**。F-03-1-001 が潰し損ねた最後の複製
- **運用者の性能ノブが確保上限を動かす経路が消える**
- 「対で意味を持つ」という D-28 の宣言が**型で表現される**
- `max_targets` が何を縛っているかが rule から読めるようになる
- 4件とも `results/` に触れないので「負債と実験が1度も混ざらない」が保たれる

**悪くなること / 引き受けるコスト**

- `_capacity.py` に型が2つ増える。`CapacityProblem` は `t0` を直接持たなくなる
- **「単位の食い違い」に構造ではなく列挙とテストで答えている** (決定2)。
  将来余裕が縮んだら分離の設計をやり直すコストが発生する
- `IpcConfig` は依然 `basis` と `input_distribution` を別々の str として持つ (§9-1 が承認されるまで)
- `RowAlignment` の非空虚性の一部は**構造ガード**に依存する (挙動ガードより弱いことを認めた上で、
  基準点集約によって挙動ガードも付けている)

**中立**: 新しい決定 ID を1つも採番しない。次の空き番号は **D-54**
