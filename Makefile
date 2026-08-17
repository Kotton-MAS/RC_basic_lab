.PHONY: sync test cov lint fmt fmt-check type lock-check ci figures-01 figures-02 threshold-02 pre-commit clean help

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
	@echo "  threshold-02 - Regenerate the ESP threshold sensitivity CSV (design.md 9)"
	@echo "  pre-commit   - Run pre-commit on all files"
	@echo "  clean        - Remove caches and build artifacts"

sync:
	uv sync --locked

test:
	uv run pytest -q

cov:
	uv run pytest --cov

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

# ESP 判定の閾値感度 (esp_threshold_sensitivity.csv) を再生成する。
# abs_tol 3点 x window 3点で 2-C の格子を判定し直すので figures-02 とは
# 分けてある (docs/design.md §9 の感度表の一次資料)。
threshold-02:
	uv run python experiments/02_esp_and_dynamics/run_02.py --config experiments/02_esp_and_dynamics/config.yaml --out results/02_esp_and_dynamics --threshold-sweep

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true