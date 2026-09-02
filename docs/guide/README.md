# 汎用基盤の手引き

このディレクトリは**どの実験からでも使える部分**の手引きです。連載固有のもの
（記事の要件・図の設計方針・レビュー記録）は `docs/series/` と `docs/plans/` に
あります。

| 手引き | 何を書いてあるか |
|---|---|
| [リザバーを足す.md](リザバーを足す.md) | モデルを 1 つ足す（3 ファイル）+ 掃引軸 |
| [課題を足す.md](課題を足す.md) | 課題を 1 本足す（union に 2 行 + `case` を 2 つ） |
| [診断を足す.md](診断を足す.md) | 診断を 1 本足す。`X` を取る族と `W` を取る族 |
| [実験を足す.md](実験を足す.md) | 実験を 1 本足す（カタログに 1 エントリ） |
| [条件を変えて試す.md](条件を変えて試す.md) | `--preset` / `--set` で手元で振る |
| [交差検証.md](交差検証.md) | 時系列の交差検証（既定では使わない） |
| [モジュール地図.md](モジュール地図.md) | 自動生成。手で書かない |
| [型と名前の対応表.md](型と名前の対応表.md) | Protocol と判別子つき union の一覧 |
| [切り出す.md](切り出す.md) | 汎用基盤を別リポジトリへ出す段取り（手順は CI が検証） |

## 汎用と連載の境界（D-126）

```
汎用  tasks/ reservoir/ readout/ diagnostics/ datasets/
      metrics*.py seeds.py split.py types.py cli.py overrides.py
      experiment/{rows_csv,diagnostics_rows,capacity_bounds}.py

連載  experiment/ の各実験パイプライン plotting/ config/ meta.py
      experiments/ results/ docs/series/ docs/plans/
```

**汎用側は連載側を 1 行も import しません。**
`tests/test_core_series_boundary.py` が機械的に守っています（分類漏れも赤に
なるので、新しいモジュールが黙って検査の外に出ることはありません）。

この形なので、汎用側だけを別リポジトリへ切り出す日が来たら `git` の操作だけで
済みます。パッケージ名 `rc_basics_lab` は**そのとき**改名すれば十分です。
