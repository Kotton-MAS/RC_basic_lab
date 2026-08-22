.PHONY: sync test cov lint fmt fmt-check type lock-check ci figures-01 figures-02 figures-03 figures-04 figures-05 data-05 threshold-02 saturation-03 washout-02-unpadded pre-commit clean help

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
	@echo "  figures-01   - Regenerate results/ for experiment 01 (CSV + 2 figures + meta)"
	@echo "  figures-02   - Regenerate results/ for experiment 02 (2 CSV + 4 figures + meta)"
	@echo "  figures-03   - Regenerate results/ for experiment 03 (4 CSV + 6 figures + meta)"
	@echo "  figures-04   - Regenerate results/ for experiment 04 (5 CSV + 5 figures + meta)"
	@echo "  figures-05   - Regenerate results/ for experiment 05 (5 CSV + 5 figures + meta; needs data-05)"
	@echo "  data-05      - Download + verify (SHA256) the experiment 05 datasets into data/"
	@echo "  threshold-02 - Regenerate the ESP threshold sensitivity CSV (design.md 9)"
	@echo "  saturation-03 - Regenerate the sequence-length sweep CSV (manual, ~30 min)"
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

lock-check:
	uv lock --check

# Stop hook (verify-ci.sh) と GitHub Actions が両方これを呼ぶ。
# ここを単一の真実とすることで、ローカルと CI の検証ロジック乖離を防ぐ。
ci: lock-check lint fmt-check type test

# 実験01の成果物 (comparison.csv / comparison_summary.csv / fig_comparison.png /
# fig_state_space.png / meta.json) を results/ に再生成する。ci の構成には
# 入れない (CI は実験を回さない)。
figures-01:
	uv run python experiments/01_what_is_rc/run.py --config experiments/01_what_is_rc/config.yaml --out results

# 実験02の成果物 (esp_diagnostics.csv / washout_sensitivity.csv / fig_esp_decay.png /
# fig_leak_timescale.png / fig_esp_map.png / fig_washout_sensitivity.png /
# meta.json) を results/02_esp_and_dynamics/ に再生成する。
# 01 と出力先を分けるのは meta.json / results 直下のファイル名が衝突するため。
figures-02:
	uv run python experiments/02_esp_and_dynamics/run_02.py --config experiments/02_esp_and_dynamics/config.yaml --out results/02_esp_and_dynamics

# 実験03の成果物 (capacity.csv / capacity_profile.csv / narma10.csv /
# narma10_taps.csv / fig_mc_sweep.png / fig_ipc_profile.png /
# fig_memory_nonlinearity.png / fig_ipc_conservation.png /
# fig_narma10_control.png / fig_narma10_taps.png / meta.json) を
# results/03_capacity/ に再生成する。成果物の一覧の単一の真実は
# experiment/capacity_pipeline.py の CAPACITY_ARTIFACTS (テストが突き合わせる)。
# 系列長掃引 (capacity_length.csv) は含めない —— T=1e6 まで回すので単独で
# 900 秒予算を食い潰す。saturation-03 で明示的に再生成する
# (threshold-02 と figures-02 の関係と同じ規律)。
figures-03:
	uv run python experiments/03_capacity/run_03.py --config experiments/03_capacity/config.yaml --out results/03_capacity

# 実験04 の成果物 (onestep.csv / freerun.csv / freerun_profile.csv /
# stability.csv / capacity.csv / fig_onestep.png / fig_freerun_attractor.png /
# fig_valid_time.png / fig_stability_map.png / fig_freerun_stats.png /
# meta.json) を results/04_chaotic_freerun/ に再生成する。成果物の一覧の単一の
# 真実は experiment/freerun_pipeline.py の FREERUN_ARTIFACTS (テストが突き
# 合わせる)。4-A だけを回す onestep-04 は置かない —— 同じ meta.json を
# 区間の一部だけの内訳で上書きしてしまうため。
figures-04:
	uv run python experiments/04_chaotic_freerun/run_04.py --config experiments/04_chaotic_freerun/config.yaml --out results/04_chaotic_freerun

# 実験05 の成果物 (anomaly.csv / anomaly_threshold.csv / anomaly_timeline.csv /
# anomaly_protocol.csv / anomaly_size.csv / fig_pr_curves.png /
# fig_score_timeline.png / fig_threshold_tradeoff.png /
# fig_protocol_sensitivity.png / fig_size_vs_performance.png / meta.json) を
# results/05_anomaly_detection/ に再生成する。成果物の一覧の単一の真実は
# experiment/anomaly_pipeline.py の ANOMALY_ARTIFACTS (テストが突き合わせる)。
# **data-05 に依存する** —— 既定の設定 (config.yaml) は実データ源 MGAB で、
# キャッシュが無ければ ValueError になる (記事のタイトルが「実データで使って
# みる」なので、主図が合成データでは食い違う)。2回目以降はキャッシュを使うので
# オフラインで回る。
figures-05: data-05
	uv run python experiments/05_anomaly_detection/run_05.py --config experiments/05_anomaly_detection/config.yaml --out results/05_anomaly_detection

# 実験05 の外部データセットを data/05_anomaly/ へ取得し SHA256 で照合する
# (D-58: データ本体はリポジトリに含めない。マニフェストは
# src/rc_basics_lab/datasets/manifests/*.csv)。既定は MGAB (CC0-1.0) のみ。
# UCR (ライセンス未指定・ZIP 184 MB) まで要るときは --dataset all を使う。
# ci の構成には入れない —— pytest はネットワークに一切触れない (D-60)。
data-05:
	uv run python -m rc_basics_lab.datasets --dataset mgab

# ESP 判定の閾値感度 (esp_threshold_sensitivity.csv) を再生成する。
# abs_tol 3点 x window 3点で 2-C の格子を判定し直すので figures-02 とは
# 分けてある (docs/design.md §9 の感度表の一次資料)。
threshold-02:
	uv run python experiments/02_esp_and_dynamics/run_02.py --config experiments/02_esp_and_dynamics/config.yaml --out results/02_esp_and_dynamics --threshold-sweep

# 系列長 T の掃引 (capacity_length.csv) を再生成する。T in {1e5, 2e5, 5e5, 1e6}
# を回すので figures-03 (予算 900 秒) には含めない (予算外・手動、< 1800 秒)。
# 「容量が足りないのか T が足りないのか」を分けるための補助実験。
saturation-03:
	uv run python experiments/03_capacity/run_03.py --config experiments/03_capacity/config.yaml --out results/03_capacity --length-sweep

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