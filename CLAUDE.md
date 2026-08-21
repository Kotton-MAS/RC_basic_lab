# CLAUDE.md — rc-basics-lab

## プロジェクト概要

リザバーコンピューティング (RC) の**基礎編の連載記事のための実験リポジトリ**。
numpy / scipy による数値実験を行い、記事に載せる図と CSV を再現可能な形で生成する。

このリポジトリの成果物は `results/` の CSV と PNG であり、**それが読者に届く唯一のもの**である。
コードの正しさは成果物の正しさのためにある。

## プロジェクト構成

```
/
├── main.py                 # 全実験の入口 (EXPERIMENTS 辞書に1行ずつ登録)
├── Makefile                # 検証コマンドの単一の真実 (make ci / figures-0N / data-05)
├── experiments/0N_*/       # 実験ごとの run スクリプトと config.yaml (argparse の薄層)
├── src/rc_basics_lab/
│   ├── tasks/              # 課題 (データ生成)。純関数。readout も reservoir も I/O も知らない
│   ├── datasets/           # 外部データの取得・キャッシュ・SHA256。**I/O を持つ唯一の場所**
│   ├── reservoir/          # ESN 本体
│   ├── readout/            # 特徴設計 (FeatureSpec) とリッジ回帰
│   ├── diagnostics/        # ESP / 容量 / IPC / リアプノフ などの計量
│   ├── experiment/         # 合成層。実験の骨格と成果物の書き出し
│   ├── plotting/           # 作図層
│   ├── config/             # 実験ごとの設定 dataclass + load_config_as
│   └── metrics*.py         # 評価指標
├── results/                # **成果物**。指紋を tests/artifact_manifest.csv にコミット済み
├── tests/                  # pytest
├── docs/                   # 要件 / 仕様 (plans/) / ADR (adr/) / サーベイ / 振り返り
└── .claude/
    ├── decisions.yaml      # 意図的な設計判断 (guard_test 必須)。**散文の正本はここ**
    ├── agents/             # プロジェクト固有の reviewer (reviewer-deletion)
    └── settings.json       # プロジェクト固有の設定のみ。ガード類はキット側
```

フック・スラッシュコマンド・スキル・汎用 agent は user スコープの claude-pdca-kit が提供する。
**プロジェクト側にコピーを置かない** (キットの改良が上書きで打ち消される)。

## コマンド一覧

```bash
uv sync                     # 依存の同期
uv add --group dev <pkg>    # dev 依存の追加 (実行時依存は増やさない。下記の規約を参照)

make ci                     # lock-check + ruff + ruff format --check + mypy + pytest
uv run pytest -q            # テスト
uv run pytest -k "name"     # 単一テスト
uv run mypy .               # 型チェック (strict)

make figures-0N             # 実験 N の成果物を results/ に再生成する
make data-05                # 外部データセットを data/ に取得する (SHA256 照合つき)
make artifacts-manifest     # 成果物の指紋を書き直す (**意図して変えたときだけ**)
```

## アーキテクチャ原則

- **層の向きは一方向**: `experiment` → `diagnostics` / `readout` / `reservoir` / `tasks`。
  `datasets` → `tasks`。逆流させない
- **`tasks/` と `metrics*.py` は純関数層**。ネットワークもファイル I/O も持たない (D-59)
- **外部 I/O は `datasets/` だけ**。取得・キャッシュ・SHA256 照合・ライセンス表記はここに閉じる
- **`experiment` から `plotting` を module-level import しない** (D-53)。作図は関数内 import
- **設定は実験ごとに独立した dataclass** (D-13)。既存の `ExperimentConfig` に足さない
- **前処理・分割の係数を作れる場所は1つに閉じる** (D-41 / D-57)。
  手法ごと・区間ごとに再推定する経路を構造上書けなくする

## コード規約

- Python 3.12+、型アノテーション必須
- **`Any` 禁止** (`object` か `Protocol` を使う)
- 関数は単一責任、50行超は分割を検討
- **1モジュール 600 行を上限とする** (D-63)。超えたら割る。**上限のほうを緩めない**
- **実行時依存は増やさない** (現在 matplotlib / numpy / pyyaml / scipy の4つ)。
  テストのオラクルに要るだけなら dev グループへ (D-62)
- **`__init__.py` はモジュール名だけを再エクスポートする** (D-75)。ただし `config` は除く。
  シンボルを並べると、次にモジュールを足す人が2箇所へ写経する義務だけが増える
- **docstring に書くのは「何を返すか・前提・送出する例外」**。
  「なぜ / 以前は / 代案 / 実測の経緯」は `.claude/decisions.yaml` か `docs/adr/` に置き、
  コードには `(D-37)` のような ID 参照だけ残す。
  **散文が3箇所 (decisions.yaml / design.md / docstring) にあると3箇所が独立にドリフトする**
- `# type: ignore` / `# noqa` を理由なく使わない。**同じ ignore を複数箇所にコピーしない**
  (1箇所に集約できるはず。実例: `plotting/style.py`)

### 実験を1本足すときに触る場所

`experiments/0N_*/` (run スクリプトと config.yaml) / `config/0N.py` (設定 dataclass) /
`experiment/` (骨格。**既存の共通ヘルパを使う。write_rows_csv / to_summary を書き直さない**) /
`plotting/figures_*.py` (**`style.py` の `new_figure` / `save_png` / `rc_context_for` を使う**) /
`main.py` の `EXPERIMENTS` / `Makefile` の `figures-0N` / `README.md` / `docs/design.md`。

## テストルール

- 新しいコードには必ずテストを書く。ファイル名は `test_{モジュール名}.py`
- **実装後は `uv run pytest` を実行し全テストパスを確認してから完了とする**
- **設定の全葉に「値を変えたら出力が変わる」テストを付ける** (D-13)。
  効かないフィールドは設定ではなく飾りである
- **ガードは「いま実行して確かめる」形で書かない。** 環境がそれを覆い隠して空振りする
  (実例: 処理系のバージョンを実行時に確かめるガードが、venv に覆われて素通りした)。
  **データで判定する形にし、変異注入で赤くなることを確認する**

## 成果物 (`results/`) の扱い

- **成果物のバイト不変が、リファクタリングの唯一の合否判定である** (D-74)。
  `uv run pytest tests/test_artifact_invariance.py` が判定器
- **指紋を「緑にするため」に書き直さない。** 意図して成果物を変えたときだけ
  `make artifacts-manifest` を実行し、**なぜ変わったかをコミットメッセージに書く**
- 指紋は「編集されていないこと」を測るが「再生成しても同じか」は測らない。
  作図・書き出しを触ったら**一時ディレクトリに再生成して比較する**
  (`--out <tmp>`)。**PNG はバイト一致する。CSV は `wall_time_s` 列だけ変わる**

## Git / PR ワークフロー

- ブランチ: `feat/xxx`, `fix/xxx`, `refactor/xxx`。**作業前に必ず作業ブランチを切る**
  (キットの `auto-commit.sh` は `main` では動かず、subagent の成果が溜まり続ける)
- コミットは SubagentStop フックが自動で行う (wip コミット。**PR 前に squash する**)
- **`git diff` は自動コミットにより空になり得る。**
  差分は `git diff $(cat .claude/tmp/base-ref)` で見る
- PR 手順: 実装+テスト → `make ci` → 差分確認 → push → `/pr`

## Claude への作業指示

- 複数ファイルにまたがる変更は planner を通してから実装する
- 既存コードパターンに合わせる。**不明点は推測せず調査する**
- 成果物の保存先: planner 仕様書 → `docs/plans/`、ADR → `docs/adr/`、資料 → `docs/`
- reviewer の findings は `.claude/schemas/findings.schema.json` に準拠させる。
  **`evidence` は必須。実測でないものを `measured` と書かない**
- **「普通はこうするが、今回は理由があって別の選択をする」判断は
  `.claude/decisions.yaml` に guard_test 付きで記録する。**
  **1決定 = 1約束 = 1 guard_test** (`check_decisions.py` は単一 node id しか解決しない)
- **subagent の報告の数値を検証せずに転記しない。** 1コマンドで裏が取れるものは取る
- 運用でうまくいかなかったことは、その場で1行記録する:
  `~/.claude/hooks/record-incident.sh note manual "<課題を1文で>"`

## やってはいけないこと

- `print()` をプロダクションコードに残さない (ruff T20)
- テストなしで機能追加しない
- 公開 API の型シグネチャを無断変更しない
- **`results/01..04/` の既存成果物を、理由の説明なしに変えない**
- **`.claude/settings.json` の `env` で処理系を固定しない** (D-73)。
  マシン固有の解決は gitignore された `settings.local.json` の `PATH` で行う
- **`decisions.yaml` の `guard_test` に載っているテストを消さない。**
  決定を守っている唯一の機械であり、消すと決定が散文に戻る

## 学習メモ

実際に起きたことだけを書く。一般論は書かない。

- **フックが動く処理系がリポジトリのコードを解析できない状態が8サイクル残っていた。**
  `PYENV_VERSION: 3.10.0` の固定が回避策のまま恒久設定になり、PEP 695 の型エイリアスを
  SyntaxError でしか読めなかった。**回避策を設定に残すときは、それが何を壊すかまで書く** (D-73)
- **「いま実行して確かめる」ガードは環境に覆い隠される。**
  上を防ぐガードを最初その形で書いたら、`uv run` 配下では venv の 3.12 が
  pyenv shims を覆い隠し、固定を戻しても緑のまま通った。**データで判定する**
- **reviewer の数値主張は外れることがある。** 存在しない関数を指して「100k点で26.7秒」と
  報告された件は、実測 0.428 秒で **62倍ずれ**ていた。**数を主張されたら1コマンドで裏を取る**
- **成果物を巻き戻すとコードと乖離する。** `results/` を過去の状態へ revert した際に
  コード側のフィールドと食い違い、「編集していないこと」しか測っていなかったため
  誰も気づけなかった。**巻き戻したら再生成して一致を確認する**
- **削減量の見積りは実測より大きく出る。** 共通化は集約先のコストを伴うため
  (呼び出し側 −64 行に対し共通ヘルパ +54 行、純減 −10 行)。
  **過大な見積りは「言われたほど減らない」と判断されて途中でやめる原因になる**
