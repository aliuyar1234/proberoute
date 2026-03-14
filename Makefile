PYTHON ?= python
CONFIG_SMOKE := configs/smoke_local_tiny.yaml
OUTPUTS_ROOT := outputs
RUN_DIR := $(OUTPUTS_ROOT)/runs/SMOKE_LOCAL_TINY/local-toy-gpt/seed_1337

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv\Scripts\python
else
VENV_PYTHON := .venv/bin/python
endif

.PHONY: env smoke prepare-data probes screen collect select-finalist final ablations eval paper test

env:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(VENV_PYTHON) -m pip install --upgrade -r requirements-torch-cu128.txt
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	$(VENV_PYTHON) -c "import platform; print(f'python={platform.python_version()}')"
	$(VENV_PYTHON) -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')"

smoke:
	$(PYTHON) -m src.cli.prepare_data --config $(CONFIG_SMOKE)
	$(PYTHON) -m src.cli.train_probes --config $(CONFIG_SMOKE)
	$(PYTHON) -m src.cli.train_mtp --config $(CONFIG_SMOKE)
	$(PYTHON) -m src.cli.evaluate --run-dir $(RUN_DIR)

prepare-data:
	$(PYTHON) -m src.cli.prepare_data --config configs/probe_1b.yaml

probes:
	$(PYTHON) -m src.cli.train_probes --config configs/probe_410m.yaml
	$(PYTHON) -m src.cli.train_probes --config configs/probe_1b.yaml

screen:
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_last_linear_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_last_mlp_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_dense_whs_random_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_dense_whs_probe_init_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_sparse_probe_init_1b.yaml
	$(PYTHON) -m src.cli.collect_results --outputs-root $(OUTPUTS_ROOT)

collect:
	$(PYTHON) -m src.cli.collect_results --outputs-root $(OUTPUTS_ROOT)

select-finalist:
	$(PYTHON) -m src.cli.select_finalist --outputs-root $(OUTPUTS_ROOT) --emit-config configs/generated/final_best_baseline_1b.yaml

final:
	$(PYTHON) -m src.cli.train_mtp --config configs/generated/final_best_baseline_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/final_sparse_probe_init_1b.yaml
	$(PYTHON) -m src.cli.collect_results --outputs-root $(OUTPUTS_ROOT)

ablations:
	$(PYTHON) -m src.cli.train_mtp --config configs/screen_sparse_random_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/ablation_sparse_warmup_1b.yaml
	$(PYTHON) -m src.cli.train_mtp --config configs/ablation_sparse_deephead_1b.yaml
	$(PYTHON) -m src.cli.collect_results --outputs-root $(OUTPUTS_ROOT)

eval:
	$(PYTHON) -m src.cli.build_paper_assets --outputs-root $(OUTPUTS_ROOT)

paper:
	$(PYTHON) -m src.cli.write_paper --outputs-root $(OUTPUTS_ROOT) --template PAPER_TEMPLATES/PAPER_TEMPLATE.md

test:
	$(PYTHON) -m pytest tests/unit tests/integration
