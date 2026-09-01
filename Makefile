.PHONY: module-map sync test cov golden golden-update lint fmt fmt-check type lock-check ci data-05 threshold-02 saturation-03 symmetry-03 panels washout-02-unpadded pre-commit clean help

help:
	@echo "Available targets:"
	@echo "  sync         - Install dependencies (uv sync --locked)"
	@echo "  test         - Run tests (uv run pytest -q)"
	@echo "  cov          - Run tests with coverage"
	@echo "  lint         - Run ruff check"
	@echo "  fmt          - Run ruff format (modifies files)"
	@echo "  fmt-check    - Check formatting without modifying"
	@echo "  type         - Run mypy"
	@echo "  ci           - Full CI check: lint + fmt-check + type + test"
	@echo "  golden       - Byte-invariance of REGENERATED artifacts (01-04, seconds)"
	@echo "  golden-update - Re-record the golden baseline (tests/golden/manifest.json)"
	@echo ""
	@echo "  手元で条件を振るときは make ではなく run スクリプトを直接:"
	@echo "    uv run python experiments/03_capacity/run_03.py --preset quick"
	@echo "    uv run python experiments/03_capacity/run_03.py --set mc_sweep.n_units=50"
	@echo "  出力は scratch/ に出る (results/ は成果物なので触らない)。docs/guide/条件を変えて試す.md"
	@echo ""
	@echo "  figures-01   - Regenerate results/ for experiment 01 (CSV + 2 figures + meta)"
	@echo "  figures-02   - Regenerate results/ for experiment 02 (2 CSV + 4 figures + meta)"
	@echo "  figures-03   - Regenerate results/ for experiment 03 (4 CSV + 6 figures + meta)"
	@echo "  figures-04   - Regenerate results/ for experiment 04 (5 CSV + 5 figures + meta)"
	@echo "  figures-05   - Regenerate results/ for experiment 05 (5 CSV + 5 figures + meta; needs data-05)"
	@echo "  data-05      - Download + verify (SHA256) the experiment 05 datasets into data/"
	@echo "  threshold-02 - Regenerate the ESP threshold sensitivity CSV (design.md 9)"
	@echo "  saturation-03 - Regenerate the sequence-length sweep CSV (manual, ~30 min)"
	@echo "  symmetry-03  - Regenerate the drive-symmetry sweep CSV (manual, D-116)"
	@echo "  module-map   - Regenerate docs/guide/モジュール地図.md from the code"
	@echo "  panels       - Measure panels per figure (manual, ~20 min; FIG-15)"
	@echo "  washout-02-unpadded - Regenerate the pad_series=False washout CSV (design.md 9.6)"
	@echo "  pre-commit   - Run pre-commit on all files"
	@echo "  clean        - Remove caches and build artifacts"

sync:
	uv sync --locked

test:
	uv run pytest -q

cov:
	uv run pytest --cov

# 成果物 (results/) の指紋を書き直す。**意図して成果物を変えたときだけ**実行する。
# リファクタリングの合否判定は「成果物が1バイトも変わらないこと」なので (D-74)、
# 何も考えずにこれを叩くと判定そのものが無効になる。
artifacts-manifest:
	uv run python tests/_artifact_manifest.py

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

type:
	uv run mypy .

# 再生成した成果物のバイト不変検査。リファクタリングの合否判定はこれで行う。
# test_artifact_invariance (D-74) が「コミット済み results/ が変わっていないか」
# を見るのに対し、こちらは「同じ設定から作り直した成果物が変わっていないか」を
# 縮小設定 (tests/golden/configs/) で数秒で見る。実験05 は外部データセットが要る
# ため対象外 (D-60: pytest はネットワークに触れない)。test に含まれるので ci でも走る。
golden:
	uv run pytest tests/test_golden.py -q

# 基準値を取り直す。**成果物を意図的に変えたときだけ**実行する
# (振る舞い不変のはずのリファクタでこれを打つと、検査そのものが無意味になる)。
golden-update:
	uv run python tests/golden_support.py --update

lock-check:
	uv lock --check

# Stop hook (verify-ci.sh) と GitHub Actions が両方これを呼ぶ。
# ここを単一の真実とすることで、ローカルと CI の検証ロジック乖離を防ぐ。
ci: lock-check lint fmt-check type test

# 実験 0N の成果物を results/ へ再生成する (**パターンルール1本**。D-125)
#
# 何をどこへ書くかは Makefile ではなく
# src/rc_basics_lab/experiment/catalog.py の CATALOG が宣言する。
# かつては実験ごとにターゲットを手書きし、同じ事実が main.py の EXPERIMENTS と
# run_0N.py にも書いてあった (3箇所)。成果物の一覧は各 spec の artifacts。
figures-%:
	uv run python main.py --experiment $* --results

# 実験05 は外部データセットの取得が要る
figures-05: data-05

# 実験05 の外部データセットを data/05_anomaly/ へ取得し SHA256 で照合する
# (D-58: データ本体はリポジトリに含めない。マニフェストは
# src/rc_basics_lab/datasets/manifest.py)
data-05:
	uv run python -m rc_basics_lab.datasets --dataset mgab

# 補助実験 (記事の本体成果物ではない。variant で選ぶ)
threshold-02:
	uv run python main.py --experiment 02 --variant threshold --results

washout-02-unpadded:
	uv run python main.py --experiment 02 --variant washout-unpadded --results

saturation-03:
	uv run python main.py --experiment 03 --variant length --results

symmetry-03:
	uv run python main.py --experiment 03 --variant symmetry --results

# 各図のパネル数 (軸の本数) を実測する (FIG-15)。Figure.savefig を捕まえて
# len(figure.axes) を数え、results/ には触れず一時ディレクトリへ生成する。
# **ci には入れない** —— 本番設定の全実験を回すので約20分かかり、「CI は実験を
# 回さない」という分担が壊れる。docs/series/図の設計方針_RC基礎編.md FIG-15 の
# 表を更新するときに手で回す。
# 地図はコードから起こす。手書きの表は写経を忘れた時点で嘘になる
# (tests/test_module_map.py が生成し直した結果と照合する)。
module-map:
	uv run python scripts/module_map.py

panels:
	uv run python scripts/count_panels.py

# 補償なし (pad_series=False) の washout 感度 CSV を再生成する (docs/design.md
# §9.6 の対比表の一次資料)。本番 config.yaml は編集しない —— このスクリプト
# 自体が dataclasses.replace で pad_series=False を上書きするため、D-19 の
# 既定 (pad_series=True) を config.yaml から動かす必要はない。
washout-02-unpadded:
	uv run python -c "\
	import dataclasses; \
	from pathlib import Path; \
	from rc_basics_lab.config import Esp02Config, load_config_as; \
	from rc_basics_lab.experiment.washout import run_washout_sweep; \
	from rc_basics_lab.experiment.esp_pipeline import write_washout_csv; \
	config = load_config_as(Path('experiments/02_esp_and_dynamics/config.yaml'), Esp02Config); \
	unpadded = dataclasses.replace(config, washout=dataclasses.replace(config.washout, pad_series=False)); \
	write_washout_csv(run_washout_sweep(unpadded), Path('results/02_esp_and_dynamics/washout_sensitivity_unpadded.csv'))"

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true