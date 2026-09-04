# docs/ の歩き方

このディレクトリは6つの役割に分かれている。**探すときはまずこの表を見る。**

| 置き場 | 役割 | 誰が読むか |
|---|---|---|
| `docs/design.md` | **実測値の正本**。各実験の設計判断と、感度表・閾値表などの数値記録 | 実装者。数値の出どころを辿るとき |
| `docs/review-findings-*.md` | reviewer が出した findings の記録。`F-xx-y-zzz` の ID はここに実在する | fixer / reviewer。ID を引くとき |
| `docs/adr/` | architect の設計判断記録 (ADR) | 実装者。「なぜこの構造か」を辿るとき |
| `docs/plans/` | planner の仕様書 (サイクル 01〜05) と、生きている方針書。**消化した指示書は畳んで `docs/plans/README.md` の索引に1行だけ残す** | 実装者。着手前に読む |
| `docs/series/` | **記事(連載)側の文書**。企画・要件・図の方針・サーベイ・整合レビュー | 執筆者。記事を書く/直すとき |
| `docs/process/` | 開発プロセスそのものの記録。エージェント運用の振り返り・削減候補・申し送り | 運用改善のとき。実装中は読まなくてよい |

## 動かしてはいけないファイル

次の3系統は**テストがパスを固定している**ので、移動・改名するとテストが落ちる。

- `docs/design.md` — `tests/test_design_doc.py` と `tests/test_config_package_layout.py` が
  `ROOT / "docs" / "design.md"` で開き、§9.2 / §9.6 / §11.2 / §11.5 の表を実測 CSV・`meta.json` と機械照合する
- `docs/review-findings-{01,02,03,03b}.md` — `tests/test_finding_id_references_resolve.py` が
  `docs/review-findings-{suffix}.md` を組み立てて、コード中の finding ID が実在することを検証する
- `docs/process/削減候補-05.md` — `tests/test_cycle_hygiene.py` の `DELETION_REVIEW_DOCS` が
  パスを定数で持ち、サイクルごとに削除レビューの記録が在ることを要求する

移動したい場合は、テスト側の定数を同じコミットで直すこと。

## docs/series/ の中身

| ファイル | 何が書いてあるか |
|---|---|
| `連載構成案_RC基礎編.md` | 記事 01〜05 の構成・分量目安・図の枚数目安・公開順。**連載の憲法** |
| `要件_rc-basics-0N.md` | 記事 N に対応する実験の要件・受け入れ条件・決定済み事項 |
| `図の設計方針_RC基礎編.md` | 全記事の図に通す規約 (FIG-1〜)。`tests/test_figure_policy.py` が機械照合する |
| `rc-basics-survey.md` | 引用すべき先行研究と、図に落とすべき定量参照点 |
| `survey_異常検知データセット_05.md` | 記事05 の執筆前サーベイ (データセットと評価指標) |
| `記事整合レビュー_2026-08-30.md` | 記事の記述と `results/` の突き合わせ結果。**未解決の指摘はここが最新** |

## docs/process/ の中身

| ファイル | 何が書いてあるか |
|---|---|
| `agent-operations-retrospective.md` | サイクル 01〜04 のエージェント運用の振り返り |
| `agent-system-review-from-artifacts.md` | 成果物から逆算したエージェント構成のレビュー |
| `agent-behaviour-fixes.md` | 上記の指摘 B-1〜B-6 への是正 |
| `リファクタリング方針.md` | モジュール分割・共通層の切り方の方針 |
| `削減候補-05.md` / `削減候補-figure-polish-3.md` | 「削れるか」reviewer の実行結果。**`tests/test_cycle_hygiene.py` が要求する** |
| `checkpoint-05b-t3.md` | 05b T3 の測定チェックポイント。`tests/test_anomaly_dataset_source.py` が「なぜ UCR を回さなかったか」の根拠として引く |

## 畳んだ文書

一時的な指示書・レビューは、**消化した時点で本体を消して索引に1行だけ残す**
(`docs/plans/README.md`)。残し続けると同じ判断が指示書と決定とコードの3箇所に
散り、独立にドリフトする。中身は `git log --follow -- <パス>` で読める。

2026-09-04 に 10 本 (2,650 行) を畳んだ。あわせて `docs/モジュール地図.md` を
消した —— D-126 で `docs/guide/` へ移した際の**移動漏れ**で、機械が読んでいる
のは `docs/guide/モジュール地図.md` のほうだけだった。

## 書き分けの原則 (CLAUDE.md より)

- 「なぜそうしたか」はコードの docstring に書かない。`.claude/decisions.yaml` (guard_test 付き) /
  `docs/adr/` / `docs/review-findings-*.md` のいずれかに書き、コードには `(D-37)` `(F-03-1-013)` のような
  **ID 参照だけ**残す
- 新しい文書を足す前に、上の6つの置き場のどれに属するかを決める。どれにも属さないなら、
  それは文書ではなく `.claude/tmp/` の作業メモである
