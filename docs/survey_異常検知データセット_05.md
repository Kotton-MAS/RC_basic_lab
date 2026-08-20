# 執筆前サーベイ: 時系列異常検知のデータセットと評価指標 (rc-basics-05)

*調査日: 2026-08-20 / 対象記事: `rc-basics-05`「実データで使ってみる — センサー時系列の異常検知」*
*位置づけ: 要件書 `docs/要件_rc-basics-05.md` の未確定事項1〜4に決着をつけるための一次情報調査*

表記: **[一次]** = 原典 (論文本文・公式ページ・実データ) で確認 / **[二次]** = 二次情報のみ / **未確認**

---

## 0. 要約 (結論だけ読む人向け)

1. 時系列異常検知の**定番ベンチマーク (Yahoo S5 / NAB / NASA SMAP・MSL / SMD) は、原著者自身が
   「放棄すべき」と書いている**。理由は自明性・非現実的な異常密度・ラベル誤り・run-to-failure bias の4点 [一次]
2. **point-adjust (PA) 評価は、一様乱数スコアが当時のSOTAを上回る**。5データセット中4つで実測されている [一次]
3. したがって本実験は **(a) 批判に応えて作られた UCR Anomaly Archive と、(b) ラベル誤りが構成上ありえない
   MGAB** を使い、**(c) AUPRC (average precision) を主指標、PA は「過大評価の実演」としてのみ併記**する
4. **一様乱数スコアを対照に常置する**。これは「PA を使っていないことの証拠」として機能する

---

## 1. 定番ベンチマークへの批判

### 1.1 Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed…"

出典: <https://arxiv.org/abs/2009.13807> / IEEE TKDE 35(3):2421-2429 (拡張アブストラクトは ICDE 2022)。
**PDF本文を取得して確認 [一次]**。

批判対象は Yahoo S5 / Numenta NAB / NASA SMAP・MSL / SMD (論文中の表記は OMNI) の4つ。

**欠陥1: Triviality (自明性)**

論文の定義: 「標準ライブラリの MATLAB コード1行で解けるなら、その異常検知問題は自明である」
(`kmeans` のような高水準関数の呼び出しは禁止。`mean`/`max`/`std`/`diff` 等の基本演算のみ)

| データセット | 実測 |
|---|---|
| Yahoo S5 | **367系列中 316系列 (86.1%) がワンライナーで解ける**。内訳 A1 65.7% / A2 97.0% / A3 98.0% / A4 77.0% |
| Yahoo S5 | さらに **367系列中193系列 (過半数) は `abs(diff(TS)) > b` という定数閾値1個だけで解ける** |
| NASA | 「約半数のケースで異常は桁違いの値の差として現れる。これは自明を通り越している」。「やや挑戦的なのは10%程度」 |
| SMD | 「28問のうち大多数がワンライナーで解ける」 |
| NAB | `art_increase_spike_density` は `movstd(AISD,5)>10` で解決 |

86.1% という数字は、論文いわく「この dataset を扱った既発表論文の性能と遜色ない」水準である。

**欠陥2: 非現実的な異常密度**

- NASA の D-2, M-1, M-2 は**テストデータの半分以上が連続した異常区間**。十数個が全長の1/3以上を異常としている
- SMD の machine-2-5 は短い区間に21個の異常
- 著者の主張: 「1つのテスト系列に含まれる異常の理想的な数は**ちょうど1個**」

**欠陥3: ラベル誤り**

- Yahoo A1-Real32: 同一の定数区間内で、A地点は TP、B地点は FP と判定される
- Yahoo A1-Real13 と A1-Real15 は実質重複
- **NAB の NYC Taxi**: 公式ラベルは5件だが、マラソンの異常は実は**同日の夏時間調整が原因**。
  独立記念日・労働者の日・MLKデー・抗議デモなど「**少なくとも7件が同等にラベル付けに値する**」。
  「TP ゼロ・FP多数と報告されたアルゴリズムが、実際には正しい事象を見つけていた、ということが起こりうる」
- NASA MSL G-1: 唯一ラベルされた異常と**視覚的に同一の挙動が他に2箇所あるがラベルなし**

**欠陥4: Run-to-Failure Bias**

Yahoo A1 の異常位置の分布は明らかに末尾に集中しており、
「**最後の点を異常とラベルするだけの素朴なアルゴリズムが高確率で正解する**」。

**総括 (4.1節)**: 「コミュニティは Yahoo, Numenta, NASA, OMNI のベンチマークを**放棄すべき**。
これらは修復不能なほど欠陥がある。これらのみで評価・比較した既存論文は割り引いて解釈すべき」

### 1.2 補強文献

- **Quo Vadis, Unsupervised Time Series Anomaly Detection?** (ICML 2024) <https://arxiv.org/abs/2405.02678>
  SOTAモデルは実質「線形写像を学習しているだけ」。モデル設計競争より benchmarking practice の改善を優先すべき
- **Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate** (MiLeTS@KDD 2026)
  <https://arxiv.org/abs/2606.02670>
  8ベンチマーク中6つで、ラベル付き異常区間の少なくとも半数が全時点の89〜100%で**単変量的に逸脱**している。
  channel-dependent モデルに channel-independent 比の測定可能な優位はない。
  → **本実験が単変量に絞る判断を積極的に正当化する材料**

### 1.3 point-adjust の過大評価 — Kim et al., AAAI 2022

出典: <https://arxiv.org/abs/2109.05257> / <https://ojs.aaai.org/index.php/AAAI/article/view/20680>
実装: <https://github.com/tuslkkk/tadpak> **[一次]**

**PA の定義**: 連続した異常区間の中で1点でも異常と検知されれば、その区間全体が正しく予測されたとみなす。

**理論**: 一様乱数スコア U(0,1) の下で recall は `R = 1 - (1/γ)·δ'(t_e - t_s)`。
すなわち**異常区間長が長いほど、閾値によらず recall → 1** に近づく。

**一様乱数スコアの F1_PA vs 当時のSOTA [一次]**:

| Dataset | ランダムスコア F1_PA | 報告されているSOTA |
|---|---|---|
| SWaT | **0.969** | GDN 0.935 |
| WADI | **0.965** | GDN 0.855 |
| MSL | **0.931** | OmniAnomaly 0.899 |
| SMAP | **0.961** | MSCRED 0.942 |
| SMD | 0.804 | OmniAnomaly 0.944 |

→ **5データセット中4つで、乱数が報告済みSOTAを上回る**。

**PA を外すと**: 未学習 (ランダム初期化) の1層LSTM Encoder-Decoder や、入力ノルムそのものが、
学習済みSOTAとほぼ同等の素の F1 を出す (SWaT: 入力ノルム 0.781 / 未学習 0.789 / GDN 0.81)。

**提案**: PA%K (区間内の検知割合が K% を超えたときだけ adjust。K=0 で PA、K=100 で素のF1) と、
K を掃引した AUC。加えて「**未学習モデルをベースラインとし、これを超えないモデルは有効性を再検討すべき**」。

---

## 2. データセット候補の比較

| Dataset | 入手 | ライセンス | 再配布 | 規模 | 変数 | ラベル粒度 |
|---|---|---|---|---|---|---|
| **UCR Anomaly Archive** | 直リンクZIP (HTTP 200 実測) | **未指定** (公式に表記なし) | **不明** | 250系列 / ZIP 175 MiB / 展開 347 MB | **単変量** | **区間 (ファイル名に埋め込み)** |
| **MGAB** | GitHub 直 clone | **CC0-1.0** (LICENSE 実物確認) | **完全に可** | 10系列 × 10万点 / 約29 MB | **単変量** | **区間** (`is_anomaly` + `is_ignored` 列) |
| GutenTAG | GitHub (生成器) | MIT | 可 | 任意生成 | 単/多変量 | 点＋区間 |
| SKAB | GitHub 直 | GPL-3.0 | 可 (GPL条件) | 38 CSV | 多変量 8ch | 点＋changepoint |
| Exathlon | GitHub `data/raw` | データ CC BY-NC-SA 4.0 | **非商用のみ・継承** | 93 trace / 24.6 GB [二次] | 多変量 2,283ch | 区間 |
| SMD | GitHub 直 | MIT (データ専用LICENSE) | 可 | 28台 × 38ch | 多変量 | 点 |
| NAB | GitHub 直 | MIT (AGPL→MITに変更済) | 可 | 58 CSV / 120 windows | 単変量 | 窓 |
| SMAP / MSL | Kaggle CLI 経由 (S3直リンクは削除済) | **未確認** | 未確認 | 82ch | 単変量 | 区間 |
| Yahoo S5 | HuggingFace gated (旧URLはDNS消滅) | 独自 | **禁止** | 371 files | 単変量 | 点 |
| SWaT / WADI | **申請フォーム必須** | 個別表記なし | **規約で明示的に禁止** | 51ch / 123ch | 多変量 | 点 |
| TSB-UAD / TSB-AD | 直リンクZIP | curation のみ Apache-2.0 | **不可** (SWaT/Yahoo混入) | 620 / 450系列 | 単/多変量 | 点 |

**注意: 2026年時点で論文記載URLの多くが死んでいる** — KDD Cup 2021 (証明書失効)、
Yahoo Webscope (DNS消滅)、iTrust 旧URL (リダイレクト)、telemanom の S3 リンク (README から削除)。

### 2.1 UCR Anomaly Archive — ZIP を直接読んだ実測値 [一次]

| 項目 | 実測値 |
|---|---|
| 公式URL | `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip` |
| 生存確認 | HTTP 200 / `application/zip` / Last-Modified: 2021-10-19 |
| ZIPサイズ | 184,066,400 bytes (約175 MiB) |
| 直リンクか | **完全な直リンク。申請・登録・Cookie 不要** |
| ファイル数 | `FilesAreInHere/UCR_Anomaly_FullData/` 配下に**ちょうど250個の .txt** |
| 命名規則 | **250個すべてが `NNN_UCR_Anomaly_{名前}_{学習区間終端}_{異常開始}_{異常終了}.txt` に完全準拠** (非準拠0件) |
| 例 | `001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt` (先頭35000点が学習用、異常は 52000〜52620) |
| 1ファイルサイズ | 最小 120 KB / **中央値 541 KB** / 最大 16.2 MB |
| 系列長 | 中央値で約3.8万点、最大で約116万点 (1行1値のASCII) |
| ラベル | **1系列につき異常は必ず1個。区間ラベル。別ラベルファイルは無い** |

**「200 vs 250」の食い違い**: Wu のサポートページは SIGKDD 2021 コンペを「200 datasets」と記述するが、
**配布ZIPの実体は250ファイル**である (上記実測)。TimeEval の `KDD-TSAD` コレクションも250と記載。**250が正**。

**ライセンスが唯一の弱点**:

- 公式ページ <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/> には
  **ライセンス表記が一切なく、引用のお願いのみ [一次]** → 法的には「ライセンス未指定」。再配布可否は**不明**
- Figshare ミラー (DOI 10.6084/m9.figshare.26410744) は **CC BY 4.0** を宣言しているが、
  **寄託者は Keogh ではなく第三者**であり、かつ **99 MB と公式ZIP (184 MB) でサイズが異なる**。
  中身が同一である保証がないため、ライセンスの根拠としては**採用しない**
- TimeEval 前処理版は「MIT License applies, where not otherwise stated」と明記 [一次]。
  `aeon` の `load_anomaly_detection(("KDD-TSAD", ...))` で系列単位のプログラム的取得が可能

→ **対処: データ本体をリポジトリに含めない。取得スクリプトとファイル名リストのみをコミットする。**

### 2.2 MGAB (Mackey-Glass Anomaly Benchmark) [一次]

- 入手: <https://github.com/MarkusThill/MGAB> (clone で完結) / DOI <https://doi.org/10.5281/zenodo.3760086>
- **ライセンス: CC0-1.0** (LICENSE ファイル実物で確認) → **再配布に一切の制約がない**
- 構成: 10系列 × 100,000点、各系列に異常10個。CSV 4列 `,value,is_anomaly,is_ignored` (1ファイル約2.9 MB)
- **ラベルが「明確」の最上級**: 異常は「値と微分が一致する2点を見つけてセグメントを除去し縫合する」
  手続きで**人工的に挿入**されたもの。**ラベル誤りが構成上ありえない**。
  一方カオス的挙動のため**人間の目には正常と区別が難しい** = 自明ではない
- `is_ignored` 列があり、評価から除外すべき遷移領域が定義済み。
  **「点調整なしの厳密評価」を設計するのに都合が良い**
- 本連載との接続: 基底系列が Mackey-Glass であり、既に `src/rc_basics_lab/tasks/mackey_glass.py` がある
- 弱点: 完全に合成。Wu & Keogh の「1系列1異常」原則には反する (10個/系列、約4%の異常率)

---

## 3. 評価指標

| 指標 | 出典 | 要点 | 追加依存 |
|---|---|---|---|
| **AUPRC / Average Precision** | 古典。TSB-AD・VUS 論文が推奨 | 閾値非依存。稀少事象で ROC-AUC より情報量が高い。**PA を一切かけないので乱数スコア問題が原理的に起きない** | 不要 (自前実装可) |
| VUS-PR | Paparrizos et al., PVLDB 15(11) / VLDBJ 34(3) <https://arxiv.org/abs/2502.13318> | ラベル周囲にバッファ長 ℓ を付けた Range-AUC を、ℓ を掃引した曲面の体積として定義 | `pip install vus` |
| PATE | Ghorbani et al., **KDD 2024** <https://arxiv.org/abs/2405.12096> | 異常区間前後のバッファ内で近接度重み付けした PR 曲線の AUC。早期/遅延検知を明示評価 | `pip install PATE` |
| Range-based P/R | Tatbul et al., NeurIPS 2018 <https://arxiv.org/abs/1803.03639> | 区間の部分重なりを existence/size/position/cardinality に分解 | `prts` |
| Affiliation P/R | Huet et al., KDD 2022 <https://arxiv.org/abs/2206.13167> | 最近傍で帰属付けし時間距離を確率へ変換。パラメータフリー | PyPI 未公開 |
| PA%K | Kim et al., AAAI 2022 | 区間内の K% 以上を検知したときだけ adjust | `tadpak` |

### 3.1 AUPRC 実装上の必須注意 [一次]

scikit-learn 公式ドキュメント原文
(<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>):

> "This implementation is not interpolated and is different from computing the area under the
> precision-recall curve with the trapezoidal rule, which uses linear interpolation and
> **can be too optimistic**."

- ✅ **average precision** (階段和 `Σ (R_n − R_{n−1}) · P_n`) を使う
- ❌ **台形則で PR 曲線下の面積を取ってはいけない**。線形補間の分だけ楽観バイアスが乗る
- TSB-AD の公式実装も `average_precision_score` を採用

### 3.2 「PA-F1 を主指標にするな」はコンセンサス、「代わりに何を使うか」は部分的合意

否定側の一次文献 (反対意見は発見できず): Kim et al. AAAI 2022 / TSB-AD NeurIPS 2024 /
PATE KDD 2024 / IBM <https://arxiv.org/abs/2409.13053> ("over-estimated detector performance")。

代替の推奨は分岐している:

- **VUS-PR 推し**: TSB-AD (NeurIPS 2024 D&B) が abstract で明示 —
  「we identify the most reliable and accurate measure, namely, VUS-PR」。1,070系列 × 40アルゴリズム
- **「単一指標は存在しない」派**: Sørbø & Ruocco, DMKD 38(3) 2024 <https://arxiv.org/abs/2303.01272>
  は20指標の taxonomy を作り「タスク依存で選べ」。
  Wagner et al. 2025 <https://arxiv.org/abs/2510.17562> は37指標を形式的性質で検証し
  「**どれも全性質を満たさない**」
- **収束点は「PR系であること」**

---

## 4. 本実験での決定

| 未確定事項 (要件書) | 決定 | 根拠 |
|---|---|---|
| 1. データセット選定 | **MGAB (既定・CI用) + UCR Anomaly Archive のサブセット (実データ軸)** | MGAB は CC0 でラベル誤りが構成上ありえない。UCR は批判文献への直接の回答であり、単変量・1系列1異常・区間ラベル |
| 2. 異常スコア構成 | **v0.1 は予測残差のみ** | 要件書の仮に従う。リザバー状態ベース (MD-RS 系) は余力があれば |
| 3. 点調整の扱い | **厳密評価を既定。PA-F1 と一様乱数スコアの PA-F1 を併記し、過大評価の度合いを実演する** | Kim et al. の再現。記事の主要な結論の1つになる |
| 4. 閾値決定方法 | **較正区間の分位点 (固定誤報率) を既定。テスト側 F1 最大化は「上限の参考値」として別枠報告** | 要件書 設計判断3 / 実験5-B の主題そのもの |

### 対照に常置するもの

Kim et al. に倣い、以下を4系統と並べて常に報告する:

1. **一様乱数スコア** — AUPRC では異常率付近に張り付く。**PA を使っていないことの証拠として機能する**
2. **入力ノルム (前処理後の |x| そのもの)** — 「学習していないもの」がどこまで届くかの下限

### データ再配布の方針

- **UCR: データ本体をリポジトリに含めない。** 取得スクリプト + ファイル名リスト + SHA256 のみコミットする
- **MGAB: CC0 のため制約なし。** ただしリポジトリ肥大を避け、取得スクリプト方式に統一する
- 取得先・ライセンス・引用要求は README に明記する (受け入れ条件6)

---

## 5. 先行研究 (ESN による異常検知)

- **MD-RS** (リザバー状態のマハラノビス距離) — npj Artificial Intelligence (2026)
  <https://www.nature.com/articles/s44387-026-00090-6>。
  **UCR Anomaly Archive 全体で評価し、主指標に PATE を使用**。正常データのみで学習する semi-supervised 設定、
  オンライン学習・推論で計算コストが低い。**本実験の構成に最も近い先行研究**
  (本文はログイン壁のため全文**未確認**)
- 同グループ, Federated Learning with Reservoir State Analysis <https://arxiv.org/abs/2502.05679> —
  SMD/SMAP/PSM 上で **AUC-ROC / AUC-PR / VUS-PR / PATE の4つの閾値非依存指標**を使用 (point-adjust なし)

---

## 6. 未確認項目 (記事執筆時に再確認するもの)

- Figshare ミラー (99 MB) と公式ZIP (184 MB) の内容同一性 — サイズが異なるため要検証
- UCR アーカイブの再配布可否 — 公式に表記がなく法的に不明
- SMAP/MSL データ本体のライセンス (コードは Apache-2.0)
- Yahoo S5 の HuggingFace 版ライセンス全文 (gate の内側)
- MD-RS 論文本文 (ログイン壁)
- PAdf (<https://arxiv.org/abs/2305.09691>) の公式実装リポジトリ
