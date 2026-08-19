# ADR 0001: 伝播器の決定性・ノイズ経路の封鎖・公開 API とレイヤ境界 (04a T2)

- **Status**: **Accepted** (2026-08-20 ユーザー承認。§9 の D-52 を含め4決定すべて採用)
- **Date**: 2026-08-19
- **Cycle / Task**: rc-basics-04a / T2 (`docs/plans/rc-basics-04.md` §4 T2)
- **決定 ID**: D-47 / D-48 (仕様 §6 で予約済み) / **D-52 / D-53 (本 ADR で新規採番)**
- **改訂を伴う既存決定**: D-36 (rule に境界を1文追加。意味は変えない)
- **関連決定**: D-12 / D-14 / D-16 / D-18 / D-23 / D-33 / D-34 / D-35 / D-49

> **本 ADR の事実の出所**: すべて**ソース読取**と、そこからの**推論**である。実測は1件も含まない
> (architect はコードを実行しない)。数値の見積りには「推定」と明記した。
> 実装者は §8 の変異注入リストで**実測に置き換えてから** `.claude/decisions.yaml` に書くこと。

---

## 1. Context

04a T2 は「負債8件」のうち4件を扱う。4件はいずれも **04b が触れる前に決めないと、04b の実装が
既存のガードを黙らせる形で潰す**という共通の性質を持つ。

| # | 負債 | 04b で潰される形 |
|---|---|---|
| 2a | `esn_propagator` が `rng` なしで `esn.step` を呼ぶ | 4-C がノイズ軸を振ると `ValueError`。**`rng` を渡して黙らせる**のが最短の直し方に見える |
| 2b | `simulate_condition` の比較軌道がノイズ実現値を制御しない | 5本目のストリーム新設か、評価順依存の結果か、どちらかが黙って入る |
| 4a | `diagnostics.ipc` が関数に隠蔽され `import ... as m` がモジュールを返さない | 04b-1 が `diagnostics/lyapunov.py` を足すとき、関数名 `lyapunov` で**3件目の衝突**が生まれる |
| 4b | `import rc_basics_lab.plotting` 単独が `ImportError` | 04b-2 が `plotting/figures_freerun.py` を足すと同型の辺が1本増える |

### 1.1 ソース読取で確認した事実

1. `ESN._update` は `x_{t+1} = (1-a)x + a·tanh(drive + Wx + σξ)`。`σ>0` かつ `rng is None` で
   `ValueError`。**ノイズは tanh の内側**に入る。
2. `conditional_lyapunov` は `cfg.check_propagator=True` (既定) で
   `propagator(X[t], t) == X[t+1]` を `propagator_tol=1e-10` (RMS/ユニット) で検査する。
   不一致時のメッセージは「**参照軌道と別の入力で伝播している疑い**」である。
3. `simulate_condition` には現在 `state_noise` 引数が**無い**。`Esp02Config` にもノイズのフィールドは無い。
4. 03 の 3-B' は `simulate_reference_trajectory(..., state_noise=...)` を通り、
   **`simulate_condition` を経由しない**。
5. `experiment/threshold.py:42` が `simulate_condition` を**直接** import している。
6. `diagnostics/__init__.py` の再エクスポートにより、**`ipc` と `memory_capacity` の2件**で
   モジュール名と関数名が衝突している。他の5診断と `tasks` / `readout` は衝突しない。**衝突は例外側**である。
7. **本番コードで `from rc_basics_lab.diagnostics import ipc` を使っている箇所は 0 件**
   (`capacity_threshold.py:39` のヒットは罠を説明するコメント)。
8. `tests/test_experiment_capacity.py:205` の
   `assert not isinstance(diagnostics_package.ipc, ModuleType)` は、
   **現在の隠蔽そのものを固定している**。
9. `tests/test_public_api_reexport.py` は `hasattr` しか見ないため、
   **関数が同名モジュールを隠していても緑になる**。この慣習テスト自身が隠蔽の共犯である。
10. 循環の実体: `plotting/__init__` → `plotting.figures` → `experiment.runner` →
    **`experiment/__init__`** → `experiment.pipeline` → `plotting.figures` (部分初期化) → `ImportError`。
    循環を起こしているのは **`experiment/__init__` が合成層を eager import している1点**である。
11. `plotting` が `experiment` から取っているのは行 dataclass だけではない。
    `mean_nrmse_by_washout` / `aggregate_nrmse` (関数) と記事メタ定数を含む。
12. `experiment.threshold` は `__init__` の import 一覧に**無く**、`esp_pipeline` が
    import する副作用でだけパッケージ属性になっている。

### 1.2 制約 (変えられないもの)

- 01・02・03 の `results/` は**バイト不変**。特に 3-B'。
- **5本目の乱数ストリームを新設しない** (ユーザー確定)。
- `DiagnosticContext` にフィールドを足さない (D-01)。診断層は `config` / `reservoir` を import しない。
- T3 には踏み込まない。新規依存を増やさない。

---

## 2. 決定1 (D-48): 伝播器は決定的にする —— ノイズ ESN は**受け付けない**

### 2.1 選択肢

| 観点 | 案A: ノイズ無しの複製で伝播 | **案B: `ValueError` で拒否** | 案C: ノイズ実現値を凍結 | 案D: `disable_noise=True` 引数 |
|---|---|---|---|---|
| 測る量の正しさ | **不正** | 正 | **厳密に正** | 不正 |
| D-18 との整合 | **破綻** (§2.3) | 整合 | 整合 | 破綻 |
| 実装量 | 小 | **最小** | 大 (`reservoir` の公開 API 変更) | 小 |
| 失敗の見え方 | **誤った診断**で落ちる | 正しい診断で落ちる | 落ちない | 誤診 |
| 04 での必要性 | — | — | **現時点で needs 0** | — |

### 2.2 決定

**案B を採用する。** `esn_propagator(esn, u)` は `esn.config.state_noise > 0` を検出したら
**伝播器を作らずに `ValueError`**。`esn.step` に `rng` を渡す実装は禁止する。

判定は **`esn_propagator` の入口**で行う。伝播器は `conditional_lyapunov` の深部で初めて
呼ばれるため、生成時に落とさないと D-18 の検査メッセージとして出る (= 誤診)。

**エラーメッセージが伝えること (4点すべてを含めること)**

1. 何を拒否したか
2. なぜか (摂動の成長率 vs 摂動 + ノイズ実現値の差の成長率)
3. **やってはいけない直し方2つ**: `rng` を渡す / **ノイズ無しの複製で伝播する**
4. 正しい経路 (`state_noise=0` で構成する。ノイズ下の λ が要るなら §2.5 の見直し条件)

### 2.3 案A を却下した理由 (最も重要)

案A は「安全に見えて、実際には成立しない」。ノイズ有りの参照軌道 `X` に対してノイズ無しの
複製で伝播すると `propagator(X[t], t) != X[t+1]`。差は `a·[tanh(pre) − tanh(pre + σξ)]` で
RMS/ユニット距離は概ね `a·σ` のオーダー (**推定**: `leak=0.3`, `σ=1e-4` で 1e-5 台)。
`propagator_tol=1e-10` を**5〜6桁超過**する。したがって:

- `check_propagator=True` では**必ず** `ValueError`。しかもメッセージは
  「参照軌道と別の入力で伝播している疑い」であり、次の実装者を**存在しないバグの捜索**へ送り込む。
- `check_propagator=False` にすると、D-18 が守る防衛線を**ノイズ条件でだけ全部外す**ことになる。
  3a で7件作った「空虚なガード」と同じ壊れ方の、より悪い版。

「ノイズ無しの複製」は**呼び出し側が明示的に作るなら正しい**。誤りなのは**伝播器の中でこっそり
作ること**である。案D も同じ破綻に到達するため却下。

### 2.4 案C を今やらない理由

理論的に正しい唯一の「ノイズ下の λ」だが、`reservoir/esn.py` の**公開 API 変更**が要り、
**04 に需要が無い** (§3.4)。案B のエラーメッセージが案C を名指しで指すので迷子にならない。

### 2.5 見直し条件

- 「ノイズ下の条件付き Lyapunov 指数」が**成果物の列として**必要になったとき
- `ESN` のノイズが tanh の**外側**の加算に変わったとき (そのとき初めて案A が成立する)

### 2.6 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_propagator_refuses_a_noisy_esn` | `ValueError` になり `ESN.step` が**1回も呼ばれない** |
| `test_noise_free_clone_fails_the_propagator_check` | **却下案Aが成立しないことの実測記録**。不一致量が `propagator_tol` を桁で超える |
| `test_propagator_is_deterministic` | 同じ `(x, t)` の2回の呼び出しがビット一致 |

---

## 3. 決定2 (D-47): `simulate_condition` は `state_noise>0` を受理しない

### 3.1 何が壊れるか

1. 各軌道が「初期状態も**ノイズ実現値も**違う」ものになり、D-14 の3ストリーム分離に
   **4本目の未制御な変動**が混ざる
2. 各軌道が引く乱数の**個数と位置**が参照軌道の消費量に依存するため、**結果が評価順に依存する**

### 3.2 選択肢

| 観点 | **案A: `simulate_condition` の入口** | 案B: `evaluate_condition` | 案C: 診断層 | 案D: 現状維持 |
|---|---|---|---|---|
| 抜け道 | 無い | **在る** (`threshold.py:42` が直接 import) | — | 全部 |
| 実現可能性 | 可 | 可 | **不可** (D-12/D-23 で `state_noise` を知れない) | — |
| guard の非空虚性 | 引数で到達可能 | 同左 | — | **ガードが存在しない** |

### 3.3 決定

**案A を採用。塞ぐ場所は `simulate_condition` の入口**とし、二重で置く。

1. **署名で受けて拒否する (主)**: `state_noise: float = 0.0` を既定値つきキーワードで足し、
   `!= 0.0` なら即 `ValueError`。D-36 が `build_esn_config` に行ったのと**同じ形**にする。
   次の実装者は必ず「上の2つと同じように流せばいい」と考えて手を伸ばす。
   **その手が触れる場所に停止標識を置く**のがこの決定の実体。
   引数が無いと guard_test が monkeypatch 依存の間接的なものになる (D-33 の弱さと同じ)。
2. **ESN 側も検査する (副・経路非依存)**: 比較軌道ループの直前で
   `reference.esn.config.state_noise > 0` を検査する。

実装は共有ヘルパ1本 (`require_deterministic_esn`) に集約し、**D-48 側からも同じヘルパを呼ぶ**。

**エラーメッセージが伝えること (4点)**: 何を拒否したか / なぜか (2つとも) /
やってはいけない直し方 (**5本目のストリーム新設**) / 正しい経路 (`simulate_reference_trajectory`)

### 3.4 04 の 4-C がノイズ条件で何を使うか

**4-C は 02 の ESP 判定経路を使わない。**

| 要るもの | 使う経路 |
|---|---|
| 3態分類 (D-45) | **自走軌道の統計から決める純関数**。02 経路に触れない |
| 4-D の状態行列 | `simulate_reference_trajectory(..., state_noise=...)` (03 の 3-B' と同一) |
| ESP 的な性質 | 同じ条件を **`state_noise=0` で別に評価**し「写像の性質」として報告 |

- 受け入れ条件4 は **3態マップが担う**ので、ノイズ条件で ESP 判定を諦めても条件は満たせる
- ノイズ下で「2軌道の分離」を測りたくなったら、それは ESP ではなく**別の量**。
  別の名前・別の列・別の決定として扱い `converged` 列の意味を変えない
- **禁止**: `state_noise` を落として ESP を測り、その行を**ノイズ条件の行として**書くこと

### 3.5 見直し条件

- 「ノイズ実現値を制御した多軌道」が必要になったとき (5本目のストリームが正面から議論の対象になる)
- 03 の 3-B' が比較軌道を必要とするようになったとき

### 3.6 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_simulate_condition_rejects_state_noise` | `ValueError` になり **`ESN.run` が1回も呼ばれない**。メッセージに代替経路が含まれる |
| `test_simulate_condition_rejects_a_noisy_esn_from_any_route` | monkeypatch 経由でも比較軌道ループ前に `ValueError` (二重化が空虚でない証明) |
| 既存 `test_reference_states_match_esp_simulate_condition` | ゼロノイズ経路が**1ビットも変わっていない** |

---

## 4. 決定3 (D-52): モジュール名と公開関数名を衝突させない

### 4.1 問題の正確な形

`from ....ipc import ipc` により、パッケージ属性 `diagnostics.ipc`(=**モジュール**) が
**関数**で上書きされる。`import a.b.c as m` は `getattr` を先に見るため `m` は関数になる。

- `monkeypatch.setattr(m, "...")` が**関数オブジェクトの属性設定**として成功し、何も差し替わらない
  → 3a のレビューで**変異試験が偽の緑になった**
- 同じ衝突が **`memory_capacity` にも存在する**。`ipc` 固有ではなく**命名規約の欠如**である
- **既存の慣習テストは `hasattr` しか見ないのでこの隠蔽を緑で通す。テスト自身が共犯**

### 4.2 選択肢

| 観点 | **案A: 関数の再エクスポートを外す** | 案B: 関数名を変える | 案C: モジュール名を変える | 案D: 現状維持 |
|---|---|---|---|---|
| 本番コードの変更 | **0箇所** | 約80箇所 | import 文 + docs | 0 |
| T3 との衝突 | 無し | **有り** (同じファイル群) | 中 | 無し |
| 04b-1 への効き | 命名を規約で縛れる | 同左 | モジュール名側 | **効かない。3件目を招く** |
| 記事の語彙 | `ipc` を関数名として保てる | `compute_ipc` は本文と乖離 | 冗長 | — |

### 4.3 決定

**案A を採用する。**

1. `diagnostics/__init__.py` から**関数** `ipc` / `memory_capacity` の再エクスポートを外し、
   代わりに**モジュール**として属性に載せる。`__all__` からもこの2名を外す
2. **命名規約を決定として立てる**: どのパッケージでも**公開サブモジュール名と同名の公開シンボルを
   `__init__` で再エクスポートしない**
3. 関数の**正規の入手経路はフルパス**。本番3ファイルは既にこの形なので、
   **変更はテスト側と `__init__` だけ**

### 4.4 却下理由

- **案B**: 記事・docs・decisions.yaml の rationale と語彙がずれる。80箇所の機械置換を T3 と
  同じサイクルで行うと「純粋な整理である」ことの証明が切り分けにくくなる。得るものは案A と同じ
- **案C**: `__init__` が関数を再エクスポートし続ける限り**同じ事故が別の名前で再発しうる**
- **案D**: 04b-1 で**3件目**になる。「テストで固定する」対象が**バグの側**になるのは
  決定として弱いどころか逆向き

### 4.5 「旧経路を残す」の扱い

- **残す**: `from rc_basics_lab.diagnostics.ipc import ipc` (本番3ファイルが現に使う経路)。テストで固定
- **残さない**: `from rc_basics_lab.diagnostics import ipc` が**関数**を返すこと。
  `__all__` のスナップショットテストで「2名が消えたこと」と「他が1つも動いていないこと」の**両側**を固定

### 4.6 見直し条件

- 外部利用者から root から関数として使いたいという要求が出たとき → 案B へ移る

### 4.7 guard_test 候補

| テスト名 | 何を測るか |
|---|---|
| `test_package_attributes_are_modules_not_shadowed` | **全7パッケージ × 全公開サブモジュール**が `ModuleType`。**04b-1 の `lyapunov.py` も自動で被覆** |
| `test_diagnostics_ipc_module_resolves_to_a_module` | `m` がモジュールで monkeypatch が実際に効く。**既存テストを置き換える** (今の assert は隠蔽を固定している) |
| `test_diagnostics_all_matches_the_recorded_snapshot` | `__all__` を増減の両側で固定 |
| `test_diagnostic_functions_are_importable_from_their_modules` | 旧経路 (フルパス) が全診断で通る |

---

## 5. 決定4 (D-53): 合成層 → 作図層の依存を**呼び出し時**に落とす

### 5.1 依存の実態を層で読み直す

- `plotting → experiment` は**静的**な依存 (行 dataclass の型・記事メタの文言)。
  3b-2 の reviewer が「記事メタを単一の真実にする目的は妥当」と評価した辺であり、**残す**
- `pipeline → plotting` は**動的**な依存 (図を描く関数を呼ぶだけ)。型注釈にも定数にも現れない
- 循環は「静的な辺」と「動的な辺」を**同じ module-level import 文**で書いていることと、
  下位パッケージの `__init__` が合成層を eager import していることの積で起きている

### 5.2 選択肢

| 観点 | 案A: `plotting/__init__` を遅延 | **案B: 合成層の import を関数内へ** | 案C: `experiment/__init__` を遅延 | 案D: `article/` へ | 案E: package 分離 |
|---|---|---|---|---|---|
| 変更量 | 小 | **小 (6行の移動、3ファイル)** | 中 | 大 | 大 |
| 循環は消えるか | **消えない (§5.4)** | 消える | 消える | **消えない (§5.4)** | 消える |
| 公開 import 経路 | 不変 | **不変** | 不変 | 変わる | **変わる** |
| 04b-2 の新しい辺 | 手当てが要る | **AST ガードが自動で被覆** | 一覧への追記が要る | — | 04 の配置と衝突 |

### 5.3 決定

**案B を採用する。** 合成層3ファイルの `plotting` import を**関数本体の中へ移す**。

規律の言い方 (D-53 の rule):

> **`rc_basics_lab.experiment` 配下のモジュールは `rc_basics_lab.plotting` を
> module-level で import しない。作図の呼び出しは関数本体の中で import する。**
> 作図層が実験層の行 dataclass・記事メタを module-level で import することは**許可する**。

### 5.4 却下理由

- **案A は偽の緑を作る (最大の却下理由)。** 遅延化すると仕様が指定した受け入れ基準
  `test_plotting_can_be_imported_first` は**緑になる**。しかし直後の属性アクセスで
  **同じ `ImportError` が1回ぶん先に移動するだけ**。受け入れ基準を満たしながら負債が残る形であり、
  このリポジトリが繰り返し踏んでいる「テストは緑だが直っていない」パターンそのもの。
  **受け入れ基準の方を強くする** (§5.6)
- **案C** は `__getattr__` の名前一覧と `TYPE_CHECKING` の**二重管理**を持ち込む。
  加えて `experiment.threshold` は import 副作用でしか属性になっていないため、
  遅延化すると既存テストが赤くなる
- **案D は循環を消せない。** `plotting` は行 dataclass だけでなく**関数**
  (`mean_nrmse_by_washout` / `aggregate_nrmse`) も import している。
  記事メタの `article/` 化自体は妥当な将来案だが**この循環の解決策ではない**
- **案E** は最も正直だが公開 import 経路を変える (D-49 の逆) 上に 04b の計画を書き換えることになる。
  **可逆な決定なので必要になってからでよい**

### 5.5 見直し条件

- 作図層が実験層の**関数**に依存しなくなったとき
- 合成層の関数内 import が10か所を超えたとき (案E の方が安くなる)

### 5.6 guard_test 候補 (**受け入れ基準を仕様より強くする**)

| テスト名 | 何を測るか |
|---|---|
| `test_plotting_can_be_imported_first` (仕様指定名) | **subprocess** で単独 import が成功。着手前に赤くなることを実測してから直す |
| `test_every_package_resolves_all_of_its_public_names_when_imported_first` | 7パッケージそれぞれで**単独 import 後に `__all__` の全名前を解決**。§5.4 の「1回の属性アクセス先へ逃げた ImportError」を捕まえる。**案A ではこれが赤いままになる** |
| `test_experiment_never_imports_plotting_at_module_level` | AST 走査で module-level の `plotting` import が0件。**04b-2 の `freerun_pipeline.py` を自動で被覆** |
| `test_plotting_may_import_experiment_at_module_level` | 逆向きの辺が**許可されている**ことを明示的に固定 (暗黙にすると次の fixer が「一貫性のため」両方向を消しに来る) |

---

## 6. 決定の要約と既存決定との関係

| ID | rule (要約) | 新規/改訂 |
|---|---|---|
| **D-47** | `simulate_condition` は `state_noise` を既定値つきキーワードで受け `!= 0` なら `ValueError`。ESN 側も検査。5本目のストリームは新設しない | 新規 (予約済み) |
| **D-48** | `esn_propagator` は `state_noise>0` を受理せず `ValueError`。`rng` を渡さない。**ノイズ無しの複製で伝播することも禁止** | 新規 (予約済み) |
| **D-52** | `__init__` は公開サブモジュール名と同名の公開シンボルを再エクスポートしない | **新規採番** |
| **D-53** | `experiment` 配下は `plotting` を module-level で import しない。逆向きは許可する | **新規採番** |
| **D-36** | 「常に rng を渡す」に**境界を1文追加**: **軌道を作る呼び出し**に限る。伝播器は D-48 が支配する | **改訂 (意味は変えない)** |

### 6.1 矛盾しないことの確認

| 決定 | 関係 |
|---|---|
| D-01 / D-15 | 触れない。**影響なし** |
| D-12 / D-23 | 再エクスポート**だけ**を変える。import 先は1つも増えない。**影響なし** |
| D-14 | D-47 が「5本目を作らない」を明文化して**補完**。改訂不要 |
| D-16 | 定義は不変。「測る条件は `state_noise=0` に限る」が加わる |
| D-18 | D-48 は**この検査が誤診として発火する唯一の経路を上流で消す**。改訂不要 |
| D-33 の教訓 | 4件すべてに変異注入を要求する (§8) |
| D-34 | 「設定側の検査 + 経路非依存の検査」の二重化を D-47 が踏襲 |
| D-49 | D-53 はこれを延長 (経路不変)。D-52 は**意図的にこの規律の例外**を1件作る (§9) |

---

## 7. 04b-1 / 04b-2 への影響

### 7.1 04b-1 (T4)

1. **`diagnostics/lyapunov.py` の公開関数名を `lyapunov` にしてはいけない** (D-52)。
   `max_lyapunov` / `lyapunov_exponent` など。guard が自動で赤くする
2. **自走は D-48 の対象外**。自走は**軌道を作る**呼び出しなので `state_noise>0` なら
   `ESN.step(..., rng=...)` に rng を渡すのが正しい (D-36 側)。この境界を docstring に明示する
3. 4-A は 01 の `run_task` 経路。`runner.py` は既に rng を常に渡している。影響なし

### 7.2 04b-2 (T5)

1. **4-C は `state_noise` を掃引軸にするが `simulate_condition` を呼ばない**。
   **呼びたくなったらそれは設計の逸脱**であり D-47 の `ValueError` がその場で止める
2. **`plotting/figures_freerun.py` は experiment を module-level で import してよい**。
   一方 **`freerun_pipeline.py` の `plotting` import は関数内に置く**。AST ガードが自動で検査する
3. `stability.csv` の列設計: ノイズ条件の行に ESP 判定由来の列を**そのまま持ち込まない**

---

## 8. 実装順序と実装者への注意

**推奨順序**: (1) D-53 → (2) D-52 → (3) D-48 → (4) D-47

**変異注入 (4件すべて。実装メモに実測を残す)**

| 決定 | 潰す変異 | 期待 |
|---|---|---|
| D-48 | 検査を消し `esn.step(x, u, rng)` に差し替える | `test_propagator_refuses_a_noisy_esn` が赤 |
| D-48 | ノイズ無し複製を返す実装 (却下案A) に差し替える | 不一致量が実測値として得られる |
| D-47 | 入口の検査を消す | `test_simulate_condition_rejects_state_noise` が赤 |
| D-47 | ESN 側の検査だけ消す | 経路非依存のテストが赤 (二重化が空虚でない証明) |
| D-52 | `__init__` に `from ....ipc import ipc` を戻す | `test_package_attributes_are_modules_not_shadowed` が赤 |
| D-53 | import を module-level へ戻す | AST テストと subprocess テストの**両方**が赤 |

**落とし穴**

- `experiment.threshold` は `esp_pipeline` の import 副作用でだけパッケージ属性になっている
- `tests/test_diagnostics_base.py` は `"rc_basics_lab.diagnostics.ipc.ipc"` という**ドット文字列**で
  診断を指している。解決方法が `getattr(package, "ipc")` に依存していないかを確認すること
- **`results/` は4件のどれでも1バイトも変わらないはず**。変わったら止まること

---

## 9. 承認された項目 (2026-08-20)

> **D-52 は承認された。** `diagnostics.__all__` から関数 `ipc` / `memory_capacity` を外し、
> `from rc_basics_lab.diagnostics import ipc` は**モジュール**を返す。命名規約も決定として立てる。

### 承認時の記録 (元の判断材料)

「公開 API シグネチャ / 公開経路の変更」に該当するのは **1件だけ**:

- **決定3 (D-52)**: `diagnostics.__all__` から `ipc` / `memory_capacity` の2名を外し、
  `from rc_basics_lab.diagnostics import ipc` が**関数ではなくモジュール**を返すようにする
  - 本番コードの影響: **0箇所**
  - 破壊の見え方: mypy で「Module not callable」、実行時 `TypeError`。**静かには壊れない**
  - 承認しない場合の代替: 案D (現状維持) になるが、04b-1 で3件目の衝突が起きうるため、
    **少なくとも命名規約 (D-52 の第2項) だけは 04b-1 の着手前に決める必要がある**

以下は**追加であり破壊ではない**ので承認不要と判断 (事後報告):

- D-47 の既定値つきキーワード追加 (D-36 と同じ形)
- D-48 の新しい `ValueError` (現在この条件で呼ぶ本番コードは存在しない)
- D-53 の import 文の移動 (公開経路は1つも変わらない)

---

## 10. Consequences

**良くなること**

- 「ノイズを入れたら `ValueError` が出たので `rng` を渡した」という**最も安い間違い**が
  正しい診断メッセージ付きで塞がる。却下案が成立しないことがテストとして残るので、
  次の実装者が同じ検討を最初からやり直さずに済む
- 変異試験の偽の緑を生む経路が消え、**慣習テスト自身が共犯だった**状態も直る
- どのパッケージも単独 import で完結し `__all__` の全名前が解決する
- 4件とも `results/` に触れないので「負債と実験が1度も混ざらない」が保たれる

**悪くなること / 引き受けるコスト**

- `from rc_basics_lab.diagnostics import ipc` の意味が変わる (§9)
- 合成層に関数内 import が3ファイル・6行ぶん生まれる。**なぜそこに在るのか**を
  コメントで D-53 に紐づける (紐づけないと次の fixer が先頭へ戻して循環が復活する)
- `plotting → experiment` の辺は**残る**。真のレイヤ分離は 05 以降の課題
- ノイズ条件での ESP 判定・条件付き Lyapunov 指数は 04 では**測れない**

**中立**: 決定 ID が D-52 / D-53 と飛ぶ (`check_decisions.py` は連番を要求しない)
