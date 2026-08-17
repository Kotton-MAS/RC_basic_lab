# rc-basics-lab

リザバー計算（Reservoir Computing）の基礎を、**同一 API・同一分割・同一探索予算**で
比較しながら確かめるための実験ラボ。連載記事「RC 基礎編」01〜05 の実装基盤であり、
本リポジトリは**サイクル01（記事01: リザバー計算とは何か）**の範囲を実装している。

- 3ベースライン（**線形 / 遅延線 / ESN**）を `FeatureSpec` の差だけで切り替える
- 誤差指標は **NRMSE = RMSE / std(y_true)**（`NRMSE = 1` が「平均予測と同等」）
- 診断層（PCA など）は `X` だけを入力に取り、**ESN に一切依存しない**
  （他のリザバー実装へそのまま移植できる）
- 意図的な設計判断は `.claude/decisions.yaml` に **guard_test 付き**で記録している

## 3コマンドで再現する

```bash
# 1. 依存をロックどおりに入れる
uv sync --locked

# 2. テスト (200 件 / 約 3 秒)
uv run pytest -q

# 3. 実験を回して results/ を再生成する (約 1 秒)
uv run python experiments/01_what_is_rc/run.py --config experiments/01_what_is_rc/config.yaml
```

3 番目は `make figures-01` でも同じ（`python main.py --experiment 01` も同じ経路）。
`results/` に次の4点が出る:

| ファイル | 内容 |
|---|---|
| `results/comparison.csv` | 2課題 × 3手法 × 5レプリケート = 30行の長形式の結果 |
| `results/fig_comparison.png` | 課題別の NRMSE（点+誤差棒、`NRMSE=1` の基準線つき） |
| `results/fig_state_space.png` | 入力空間とリザバー状態空間の PCA |
| `results/meta.json` | commit / 時刻 / ライブラリ版 / 設定全体 / 実測 wall time / PCA の要約 |

図は 200 dpi（retina 相当）。日本語フォント（Hiragino Sans / Noto Sans CJK JP /
IPAexGothic / Yu Gothic）が見つかる環境では日本語ラベル、見つからない環境では
**ラベル文字列ごと英語**に切り替わる（豆腐文字を出さないため。D-10）。
新しい依存は追加していない。

## 何が分かるか（実測値）

テスト区間・5レプリケートの NRMSE 平均 ± 標準偏差:

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

## リポジトリ構成

```
src/rc_basics_lab/
├── config.py / seeds.py / metrics.py / meta.py / types.py   # 土台
├── diagnostics/     # 状態系列 X だけを見る診断層 (reservoir に依存しない)
├── reservoir/       # ESN (step / run / x0 / state_noise を公開)
├── readout/         # 設計行列 (3手法の差はここだけ) とリッジ回帰
├── experiment/      # 分割・ランナー・PCA 比較・書き出し・1コマンド経路
└── plotting/        # スタイル (CJK フォント探索) と図
experiments/01_what_is_rc/{config.yaml,run.py}   # 実験1の設定と CLI
results/             # 生成物 (コミット対象。再実行で上書きされる)
docs/design.md       # 数値の根拠と実測結果
docs/plans/          # 仕様書 (タスク分解と受け入れ基準)
tests/               # pytest
```

## 開発

検証コマンドの単一の真実は `Makefile`:

```bash
make ci          # lock-check + lint + fmt-check + type + test (CI と同じ)
make test        # uv run pytest -q
make figures-01  # results/ を再生成
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
