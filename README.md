# rc-basics-lab

リザバー計算（Reservoir Computing）の基礎を、**同一 API・同一分割・同一探索予算**で
比較しながら確かめるための実験ラボ。連載記事「RC 基礎編」01〜05 の実装基盤であり、
本リポジトリは**記事01〜05 の全範囲**を実装している
（各実験の内容と再生成コマンドは下の「実験0N」の節を参照）。

- 3ベースライン（**線形 / 遅延線 / ESN**）を `FeatureSpec` の差だけで切り替える
- 誤差指標は **NRMSE = RMSE / std(y_true)**（`NRMSE = 1` が「平均予測と同等」）
- 診断層（PCA・ESP 判定・条件付き Lyapunov 指数・実効時定数）は `X` だけを
  入力に取り、**ESN に一切依存しない**（他のリザバー実装へそのまま移植できる）
- 意図的な設計判断は `.claude/decisions.yaml` に **guard_test 付き**で記録している
  （散文ではなく、決定ごとに「破れたら落ちるテスト」を1本ずつ持たせている）

## 3コマンドで再現する

```bash
# 1. 依存をロックどおりに入れる
uv sync --locked

# 2. テスト (約 3 秒)
uv run pytest -q

# 3. 実験を回して results/ を再生成する (約 1 秒)
uv run python experiments/01_what_is_rc/run.py --config experiments/01_what_is_rc/config.yaml
```

### Claude Code のフックを使う場合のみ: `settings.local.json`

上の3コマンドには不要。**Claude Code のフック (claude-pdca-kit) を動かす場合だけ**必要になる。

キットのフックは `PY=$(command -v python3)` で処理系を解決する。`pyenv` などが
古い Python を先に解決すると、このリポジトリのコード (PEP 695 の型エイリアスを含む)
を**構文解析できず、フックが無言で死ぬ**。実際に8サイクルこの状態が続いた (D-73)。

処理系の解決はマシンごとに違うので、gitignore された `.claude/settings.local.json` に置く:

```jsonc
{
  "env": {
    // このリポジトリの .venv/bin を PATH の先頭に置く。
    // ${PATH} の展開は効かないので、既存の PATH を全部列挙すること。
    "PATH": "<このリポジトリの絶対パス>/.venv/bin:<既存の PATH をそのまま>"
  }
}
```

`printenv PATH` の出力をそのまま後ろに繋げばよい。設定後、
`uv run pytest tests/test_hook_interpreter.py` が緑になれば正しく解決できている
(`.venv` を作り直したときに先頭要素が dangling になっていないかも、このテストが見る)。

3 番目は `make figures-01` でも同じ（`python main.py --experiment 01` も同じ経路）。
`results/` に次の4点が出る:

| ファイル | 内容 |
|---|---|
| `results/comparison.csv` | 2課題 × 3手法 × 5レプリケート = 30行の長形式の結果 |
| `results/comparison_summary.csv` | (課題, 手法) ごとの NRMSE 平均±標準偏差・符号正解率平均（集計版） |
| `results/fig_comparison.png` | 課題別の NRMSE（点+誤差棒、`NRMSE=1` の基準線つき） |
| `results/fig_state_space.png` | 入力空間とリザバー状態空間の PCA |
| `results/meta.json` | commit / 時刻 / ライブラリ版 / 設定全体 / 実測 wall time / PCA の要約 |

図は 200 dpi（retina 相当）。日本語フォント（Hiragino Sans / Noto Sans CJK JP /
IPAexGothic / Yu Gothic）が見つかる環境では日本語ラベル、見つからない環境では
**ラベル文字列ごと英語**に切り替わる（豆腐文字を出さないため。D-10）。
新しい依存は追加していない。

## 何が分かるか（実測値）

テスト区間・5レプリケートの NRMSE 平均 ± 標準偏差
（**この表は `results/comparison_summary.csv` の値**。集計ロジックは
`experiment.summary.aggregate_nrmse` にあり、乖離したら
`tests/test_readme_summary.py` が落ちる）:

| 課題 | 線形 | 遅延線 | ESN |
|---|---|---|---|
| Mackey-Glass (1ステップ先予測) | 0.1454 ± 0.0002 | **0.0005 ± 0.0000** | 0.0007 ± 0.0001 |
| 遅延パリティ `y[t]=u[t-1]u[t-2]` | 1.0004 ± 0.0003 | 1.0007 ± 0.0007 | **0.0894 ± 0.0166** |

符号正解率（遅延パリティ）: 線形 0.500 / 遅延線 0.522 / **ESN 1.000**。

- **遅延パリティでは線形も遅延線も解けない**。目標が `{1, u[t-k]}` の張る線形空間に
  厳密に直交するため、失敗は経験則ではなく解析的な帰結である（`docs/design.md` §2.2）
- **Mackey-Glass の1ステップ先予測では遅延線が ESN を上回る**。
  Δt=1.0 の MG は1ステップ先ならほぼ線形予測できるため。
  「MG だけを見れば遅延線で足りる。差が出るのは非線形性を要求するパリティの側」
  という読み方をする（隠さずそのまま記録している）
- リザバー状態の PCA は、生の入力（1次元）より高次元に広がる一方で、
  **64ラグの遅延埋め込み（65次元）より `n_components_95` は小さい**。
  分散で数えた次元数はリザバーの効用と別物であるという実測結果を
  `docs/design.md` §7.2 に数値付きで残してある
- **MG では alpha 格子の下端が選ばれ続けるが、これは探索の失敗ではない**。
  検証 NRMSE は alpha に対して単調増加で内点解が存在せず、下端 1e-10 は
  `cond(Phi^T Phi) ≈ 2e16` に由来する数値限界である（`docs/design.md` §8 / D-11）

## 実験02: ESP はスペクトル半径だけでは決まらない

```bash
# 7成果物 (CSV2枚 + 図4枚 + meta.json) を再生成する (実測 wall_time_s = 88.0 秒)
make figures-02

# ESP 判定の閾値感度 CSV だけを再生成する (実測 60.70 秒)
make threshold-02
```

`make figures-02` は `python main.py --experiment 02` と同じ経路。
出力先は `results/02_esp_and_dynamics/`（01 と `meta.json` の名前が衝突するため分けている）:

| ファイル | 内容 |
|---|---|
| `esp_diagnostics.csv` | 369行（2-A 15 + 2-B 18 + 2-C 336）。ESP 判定・λ・実効時定数を条件ごとに |
| `washout_sensitivity.csv` | 180行（6 washout × 2課題 × 3手法 × 5レプリケート） |
| `fig_esp_decay.png` | 2-A: 無入力での状態距離の減衰（ρ 別） |
| `fig_leak_timescale.png` | 2-B: リーク率と実効時定数（理論線 `-1/log(1-a)` を重ねる） |
| `fig_esp_map.png` | 2-C: ρ × 入力強度 の ESP 成立領域（記事の目玉） |
| `fig_washout_sensitivity.png` | 2-D: washout 長への性能感度 |
| `meta.json` | commit / 設定全体 / `esp_defaults` / λ と判定の整合の内訳 / 2-D の変動幅 |
| `esp_threshold_sensitivity.csv` | 判定閾値 9通りの感度（`make threshold-02` でのみ更新） |

### 何が分かるか（実測値）

**入力を強くすると、ESP が成立する ρ の上限が上がる**
（この行は `results/02_esp_and_dynamics/esp_threshold_sensitivity.csv` の
既定値の行そのもの。乖離したら `tests/test_readme_summary.py` が落ちる）:

| 入力強度 σ_u | 0 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| ESP が壊れる最小の ρ | 1.0 | 1.1 | 1.1 | 1.3 | 1.5 | 1.7 | 格子外 (>1.9) |

- 無入力（σ_u=0）では通説どおり **ρ=1.0 が境界**だが、σ_u を上げると境界は単調に
  上がり、σ_u=2.0 では **ρ=1.9 でも ESP が成立する**。
  「ESP を ρ<1 と同一視するのは広く流布した誤り」（Scholarpedia / Jaeger）の再実演
- **この境界は判定閾値の選び方に依存しない**。`abs_tol` を 1e-4〜1e-8、
  `window` を 100〜400 で振った9通りすべてで臨界 ρ は1点も動かない
  （`docs/design.md` §9.2）
- **条件付き Lyapunov 指数（局所量）と ESP 判定（大域量）は完全には一致しない**。
  `meta.json` の実測で「λ>0 なのに収束（偽の ESP）」は `n_false_esp = 0` 件、
  一方「λ<0 なのに非収束」が `n_local_but_not_global = 27` 件ある。後者は実装バグ
  ではなく**多安定性**（tanh が奇関数なので `x*` と `-x*` が対の吸引子になる）で、
  4軌道を直接観測して確認した（`docs/design.md` §9.5 / D-20）
- **washout 長は性能をほとんど動かさない**。訓練データ量との交絡を除く補償
  （D-19）を入れると、MG × ESN の NRMSE の (最大/最小) 比は
  `washout_sensitivity.headline.ratio = 1.00763` にとどまり、変動幅は
  **レプリケート間のばらつきの 1/17**（6組すべてで
  `exceeds_replicate_noise = false`）。補償を入れないと同じ曲線が
  **完全に単調増加**（比 1.01151）に見えるが、それは訓練データ量の効果である

## 実験03: 記憶容量 (MC / IPC) と情報処理容量

```bash
# 9成果物 (CSV3枚 + 図5枚 + meta.json) を results/03_capacity/ に再生成する
# (実測 wall_time_s = 326.98 秒 / 予算 900 秒)
make figures-03

# 系列長 T の掃引 CSV だけを再生成する (予算外・手動、実測 174.58 秒)
make saturation-03
```

`make figures-03` は `python main.py --experiment 03` と同じ経路。
出力先は `results/03_capacity/`:

| ファイル | 内容 |
|---|---|
| `capacity.csv` | 118行。条件 (3-A/3-B/3-B'/3-C) ごとの MC/IPC 総容量・実効遅延など |
| `capacity_profile.csv` | 21,812行。次数・遅延ごとの長形式プロファイル (D-38) |
| `narma10.csv` | 20行。3-C の成績 (4手法 × 5レプリケート。列は 01 と同一) |
| `fig_mc_sweep.png` | 3-A: 記憶容量プロファイルの ρ 依存性 (上限線 y=N つき) |
| `fig_ipc_profile.png` | 3-B: IPC の (次数, 遅延) ヒートマップ |
| `fig_memory_nonlinearity.png` | 3-B: 線形/非線形容量の分解 |
| `fig_ipc_conservation.png` | 3-B': 状態ノイズ下の保存則 (対角線 y=N) |
| `fig_narma10_control.png` | 3-C: 公平な対照での NARMA10 (参照線つき) |
| `meta.json` | commit / 設定全体 / 実測 wall time / `threshold_comparison` / `narma10_verdict` |
| `capacity_length.csv` | 系列長 T の掃引 (`make saturation-03` でのみ更新) |

### 何が分かるか（実測値）

**記憶容量は上限 N には遠く届かず、ρ とともに伸びる**。本番設定 (N=200) で
`mc_total / N` が到達する最大は 18.9%。図の上限線 y=N は**スケールの参照**で
あって到達目標ではない (`docs/design.md` §11.5)。

**ρ を上げると非線形容量が減り、線形容量が増える**。IPC の次数分解
(`fig_ipc_profile.png` / `fig_memory_nonlinearity.png`) で配分の移動が読める。
状態ノイズを入れると総容量は N を厳密に下回る (`fig_ipc_conservation.png`)。

**NARMA10 では遅延線が ESN を上回った** (テスト NMSE のレプリケート平均、
`narma10.csv` から機械照合):

| 手法 | 線形 | 遅延線 (リッジ) | 遅延線 (OLS) | ESN |
|---|---|---|---|---|
| NMSE | 1.0181 | 0.1538 | **0.1534** | 0.2673 |

- 結果の向きは問わない設計にしてあり (`meta.json` の `narma10_verdict` は勝敗を
  両方向とも同じ形で書く)、**このまま記録している**
- **正則化は遅延線の優位の理由ではない** (D-90)。先行 (Goudarzi et al. 2014) の
  対照は正則化なしの遅延線だったので、同じ特徴・同じ分割で alpha だけを 0 に
  した第4水準を足した。結果は 0.1534 で、リッジ版 (0.1538) ともほぼ同値のまま
  ESN を下回る。`meta.json` の `regularisation_changes_the_verdict` は `false`
- **容量がこの成績を説明する**。同じ ESN (N=50) の線形メモリ容量は
  `mc_total = 11.15` しかないのに対し、遅延線は選ばれた k=30 ぶんの
  **完全なタップ**を持つ。さらに ESN は総容量 `ipc_total = 30.25` の大半
  (`ipc_nonlinear = 19.79`) を、この課題があまり必要としない非線形性に使っている
- ただし遅延線は 5回中4回で格子の**上端 k=30** を選んでおり真の最適は測れていない。
  探索予算も非対称 (遅延線は alpha × k、ESN は alpha のみ)。詳しくは
  `docs/design.md` §11.5
- しきい値法 (`none` / `surrogate` / `chi2`) の比較と既定の根拠は
  `docs/design.md` §11.2 (一次資料は `meta.json` の `threshold_comparison`)

## 実験04: カオス時系列の自由走行予測

```bash
# 04 の成果物 (CSV5枚 + 図5枚 + meta.json) を results/04_chaotic_freerun/ に
# 再生成する (実測 wall_time_s = 220.3 秒 / 予算 900 秒)
make figures-04
```

`make figures-04` は `python main.py --experiment 04` と同じ経路。
4-A (教師強制の1ステップ先予測) / 4-B (自走) / 4-C (3態マップ) /
4-D (同じ状態行列への MC・IPC) を1回の実行でそろえる。

- **Lorenz (10, 28, 8/3) を RK4 (刻み 0.002) で積分し 5 ステップごとにサンプル**
  する (Delta t = 0.01)。この Delta t は較正で選んだ値で、落選値の実測は
  `docs/design.md` §11 にある (D-41)
- **最大 Lyapunov 指数は数値推定が正本** (Benettin 法)。実測 **0.9161 [1/時間]**
  で、文献値 0.9056 (Viswanath 1998) との相対差は **1.16%** (D-42)。
  文献値は照合にしか使わない
- **教師強制の1ステップ先予測では ESN と遅延線の差が小さい** (要件書 受け入れ
  条件3 の片側)。10 レプリケートの NRMSE 平均は Lorenz で
  遅延線 1.8e-05 / ESN 5.1e-05、Mackey-Glass で 遅延線 6.2e-04 / ESN 4.3e-04。
  対して線形 (`[1, u[t]]`) は 0.060 / 0.145 で、課題自体は自明ではない
- **自走にすると対照が成立しない** (受け入れ条件3 のもう片側)。有効予測時間
  (誤差の NRMSE 比が 0.4 を超えるまで、**Lyapunov 時間で正規化**) の中央値は
  Lorenz で **ESN 4.83 / 遅延線 0.179 / 線形 0.069 [1/lambda_max]** ——
  **27〜70 倍**の差がつく (D-43)。対照も同じ経路で自走させて測っている
  (遅延線の閉ループはシフトレジスタ、線形は状態を持たない恒等写像)
- **アトラクタ再現は視覚評価で結論しない** (D-46)。リターンマップの点集合距離と
  パワースペクトルの全変動距離の**2本**を、**真の軌道のシャッフル代替**と
  比べる。ESN は 2課題とも **10/10 のシードで代替より近い** (片側符号検定
  p = 0.00098)。対照は 0/10
- **自走の3態 (発散 / 周期軌道 / アトラクタ再現) は純関数 + 数値基準で分類する**
  (D-45)。`float64` の範囲内で 1e200 まで伸びる破綻は `isfinite` では捕まらない
  ので、分類器は振幅そのものを見る。**状態ノイズの効きは非単調**で、発散は
  80 条件中 31 -> 32 (中程度のノイズでは 28 / 24 まで減るが最大のノイズでは
  無ノイズより増える)。「ノイズ注入を入れずに『自走が不安定』と結論すると誤り」
  (要件書 設計判断3) は支持されるが、単純な単調安定化は成り立たない
- **容量 (MC / IPC) は自走の成否を単調には説明しない** (4-D)。MC が最大の条件は
  ほとんど発散側にある。駆動が i.i.d. でないため容量の絶対値は 03 の掃引と
  比較できない (`meta.json` の `capacity_note`、`docs/design.md` §12.5)
- **自走は外部生成の状態系列生成器でも動く** (要件書 受け入れ条件7、D-50)。
  `readout/autoregressive.py` は `reservoir` を import せず、状態更新器を
  `StateUpdater` プロトコルで受ける

図5枚 (`fig_onestep` / `fig_freerun_attractor` / `fig_valid_time` /
`fig_stability_map` / `fig_freerun_stats`) と各指標の定義・閾値感度・実行時間は
`docs/design.md` §12。

## 実験05: センサー時系列の異常検知 (実データ)

```bash
# 05 のデータセット (MGAB, CC0-1.0) を data/ へ取得し SHA256 で照合する
make data-05

# 05 の成果物 (CSV5枚 + 図5枚 + meta.json) を results/05_anomaly_detection/ に
# 再生成する (実測 wall_time_s = 231.4 秒 / 予算 900 秒)
make figures-05
```

`make figures-05` は `python main.py --experiment 05` と同じ経路で、`data-05` に
依存する (既定のデータ源は実データ **MGAB** の系列 1〜3、先頭 60,000 点)。
5-A (検知性能の比較) / 5-B (閾値のトレードオフ) / 5-C (プロトコル感度) /
5-D (リザバーサイズ) を1回の実行でそろえる。

- **主指標は AUPRC で、point-adjust を一切通していない** (D-54 / D-55)。
  一様乱数スコアの AUPRC は異常率に張り付く (実測 **0.0567** vs 異常率
  **0.0555**) ので、この一致自体が「指標を水増ししていないこと」の証拠になる

| 系統 | AUPRC (平均±s.d.) | 一様乱数対照比 | 対照と区別できるか (片側符号検定) |
|---|---|---|---|
| **ESN 残差** | **0.1602 ± 0.0130** | **2.83x** | **あり (15/15, p=3.1e-05)** |
| 遅延線 残差 | 0.0570 ± 0.0118 | 1.01x | なし (9/15, p=0.304) |
| 直前値 残差 | 0.0559 ± 0.0106 | 0.99x | なし (6/15, p=0.849) |
| 移動統計 | 0.0560 ± 0.0111 | 0.99x | なし (7/15, p=0.696) |
| 一様乱数 (対照) | 0.0567 ± 0.0120 | 1.00x | なし (0/15, p=1.000) |
| 入力ノルム (対照) | 0.0558 ± 0.0113 | 0.98x | なし (7/15, p=0.696) |

- **MGAB で対照から離れるのはリザバー残差だけ** (3系列 x 5レプリケート = 15 対)。
  遅延線・直前値・移動統計・入力ノルムは**一様乱数と区別がつかない** ——
  「対照を置かずに 0.057 を報告していたら、検知できていることになっていた」
- **運用閾値は較正区間のスコア分位点だけで決め、テスト区間のラベルを見ない**
  (D-56)。テスト側で閾値を最適化した参考値との差 `f1_test_optimal - f1_calibrated`
  は 90 行の平均 **0.0761** (最大 0.1182 / 負の行は 0)
- **プロトコル (正規化 x 入力窓 x スコア平滑化 = 27 格子点) を振ると順位は動く**
  (**27 点中 23 点**で変動、逆転した系統対は延べ **62 組**)。**ただし逆転 62 組の
  うち「両方が対照と区別できる」組は 0 組**で、**ESN の順位は 27 点すべてで
  1 位のまま**である (D-78)。順位だけを報告すると「プロトコルに敏感で信頼
  できない」という**逆の結論**になる —— `fig_protocol_sensitivity.png` は
  印のある系統を太い実線、無い系統を細い破線で描き、右パネルに
  「延べ 62 組 / 両方に印 0 組」を並べる
- **リザバーサイズ N を削ると性能は落ちる** (5-D)。基準 N=200 の AUPRC 0.1602 に
  対し N=100 で 0.1392 (0.869 倍)、N=50 で 0.585 倍、N=25 で 0.469 倍 ——
  90% を初めて割るのは **N=100** (`n_units_at_90pct`、格子の端ではない)。
  全 24 行が `n_train=14800` で学習量は揃っている (D-80)
- **UCR (250系列のアーカイブ) も同じ経路で回せる**が、系列ごとに `train_end` が
  違うため 5-D の前提 (学習量が揃っていること) を満たさず `ValueError` になる。
  8系列での実測 (ESN が乱数より弱い系列が 2 本ある) は
  `docs/checkpoint-05b-t3.md`

図5枚 (`fig_pr_curves` / `fig_score_timeline` / `fig_threshold_tradeoff` /
`fig_protocol_sensitivity` / `fig_size_vs_performance`) と既定値の出どころ・
実行時間の内訳は `docs/design.md` §13。データの取得方法とライセンスは次節。

## データセットのライセンスと取得手順

実験05 (センサー時系列の異常検知) は外部の公開データセットを使う。
**データ本体はこのリポジトリに一切含めない** (D-58)。置いてあるのは取得スクリプトと
ファイル名・SHA256 のマニフェスト (`src/rc_basics_lab/datasets/manifests/*.csv`) だけで、
`data/` は `.gitignore` 済みである。

```bash
# MGAB (既定) を data/05_anomaly/mgab/ へ取得し SHA256 で照合する
make data-05

# UCR も取りたいとき (ZIP 184 MB。採用サブセット8系列だけを展開する)
uv run python -m rc_basics_lab.datasets --dataset all
```

取得は HTTPS のみ・リダイレクト3回まで・サイズ上限 200 MB・タイムアウトつきで行い、
**SHA256 が一致しないファイルは例外にしてキャッシュにも残さない**。
ZIP は member 名を検査してから必要な8ファイルだけを展開する
(`zipfile.ZipFile.extractall` は使わない)。

取得処理 (`datasets/fetch.py`) は `os.O_DIRECTORY`/`os.O_NOFOLLOW`/`dir_fd=` に
依存するため **POSIX (macOS/Linux) 専用であり Windows では動作しない**。

| データセット | ライセンス | 出典 | 本リポジトリでの扱い |
|---|---|---|---|
| MGAB (Mackey-Glass Anomaly Benchmark) | `CC0-1.0` | <https://github.com/MarkusThill/MGAB> / DOI <https://doi.org/10.5281/zenodo.3760086> | 取得スクリプトと SHA256 のみ。データ本体は同梱しない |
| UCR Time Series Anomaly Archive (2021) | `未指定 (再配布可否不明・データ本体は同梱しない)` | <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/> | 取得スクリプトと SHA256 のみ。**再配布はしない** |

- **MGAB** は CC0-1.0 (パブリックドメイン相当)。引用のお願いとして
  Thill, Konen, Bäck, *MGAB: The Mackey-Glass Anomaly Benchmark* (2020),
  Zenodo, doi:10.5281/zenodo.3760086 を挙げる。
- **UCR** は公式ページにライセンス表記が一切なく、引用のお願いだけがある。
  したがって**再配布可否が法的に不明**であり、本リポジトリはデータ本体を
  同梱しない。利用者が自分でダウンロードする経路だけを提供する。
  Figshare のミラーは寄託者が原著者でなくサイズも異なるため根拠にしない。

ライセンス文字列はマニフェスト CSV の先頭 (`# license:`) と
`datasets/{mgab,ucr}.py` の `LICENSE` 定数、そしてこの表の3箇所にあり、
`tests/test_datasets_anomaly.py::test_readme_license_matches_the_manifests` が
一致を機械検査する。

## リポジトリ構成

```
src/rc_basics_lab/
├── seeds.py / metrics.py / meta.py / types.py   # 土台
├── config/          # 設定 dataclass 群 (実験サイクル単位で分割、公開経路は
│                   #   `rc_basics_lab.config` の1本のまま)
│                   #   _common: ローダ / experiment01 / esp02 / capacity03
│                   #   / chaos04 / anomaly05
├── tasks/           # 課題層 (MG / 遅延パリティ / NARMA10 / Lorenz / 異常検知)
│                   #   anomaly: 系列の器・共通前処理・合成源 (I/O を持たない)
├── datasets/        # 外部データセットの取得・SHA256 照合・読み取り
│                   #   **I/O を持つ唯一のパッケージ** (D-59)。依存は datasets -> tasks
├── diagnostics/     # 状態系列 X だけを見る診断層 (reservoir に依存しない)
│                   #   PCA / ESP 判定 / 条件付き Lyapunov / 実効時定数
│                   #   / MC / IPC / 最大 Lyapunov 指数
├── reservoir/       # ESN (step / run / x0 / state_noise を公開)
├── readout/         # 設計行列 (3手法の差はここだけ) とリッジ回帰
│                   #   autoregressive: 自由走行 (reservoir に依存しない)
├── experiment/      # 分割・ランナー・PCA 比較・書き出し・1コマンド経路
│                   #   02: esp / washout / threshold と esp_pipeline
│                   #   03: capacity (MC/IPC) と capacity_pipeline
│                   #   04: attractor (自走の評価) / freerun (4-A・4-B)
│                   #       / stability (4-C・4-D) / freerun_pipeline
└── plotting/        # スタイル (CJK フォント探索) と図
experiments/01_what_is_rc/{config.yaml,run.py}         # 実験1の設定と CLI
experiments/02_esp_and_dynamics/{config.yaml,run_02.py}  # 実験2の設定と CLI
experiments/03_capacity/{config.yaml,run_03.py}          # 実験3の設定と CLI
experiments/04_chaotic_freerun/{config.yaml,run_04.py}   # 実験4の設定と CLI
results/             # 生成物 (コミット対象。再実行で上書きされる)
results/02_esp_and_dynamics/  # 実験2の生成物
results/03_capacity/  # 実験3の生成物
results/04_chaotic_freerun/   # 実験4の生成物
docs/design.md       # 数値の根拠と実測結果
docs/plans/          # 仕様書 (タスク分解と受け入れ基準)
tests/               # pytest
```

## 開発

検証コマンドの単一の真実は `Makefile`:

```bash
make ci           # lock-check + lint + fmt-check + type + test (CI と同じ)
make test         # uv run pytest -q
make figures-01   # 実験01 の results/ を再生成
make figures-02   # 実験02 の results/02_esp_and_dynamics/ を再生成
make threshold-02 # 実験02 の閾値感度 CSV だけを再生成
make figures-03   # 実験03 の results/03_capacity/ を再生成
make saturation-03 # 実験03 の系列長掃引 CSV だけを再生成 (予算外・手動)
make figures-04   # 実験04 の results/04_chaotic_freerun/ を再生成
```

- Python 3.12+ / 依存は **numpy・scipy・matplotlib・pyyaml のみ**
- `mypy --strict`（`disallow_any_explicit`）と ruff（`print()` 禁止）を緩めない
- 設定 YAML の**未知キーは即エラー**（`ConfigError`）。タイプミスが黙って無視されると
  「設定したのに効いていない」実験になるため（D-09）
- 全設定パラメータについて「値を変えると出力が変わる」ことを
  `tests/test_config_wiring.py` が網羅的に検査する。パラメータを足したら
  同ファイルに1行足すまでテストは赤のまま

## ライセンス

Apache License 2.0 — `LICENSE` を参照。Copyright 2026 Takumi Kotooka.
