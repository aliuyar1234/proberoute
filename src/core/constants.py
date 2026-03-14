from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SCHEMAS_ROOT = REPO_ROOT / "schemas"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
LOGS_ROOT = REPO_ROOT / "logs"

CANONICAL_HIDDEN_NORM = "stateless_layer_norm"
LAYER_MIX_MODES = {"last_layer", "dense_whs", "sparse_topm", "probe_only"}
ROUTER_INIT_MODES = {"none", "random", "probe_zscore_top5"}
RUN_STAGES = {"smoke", "probe", "screen", "final", "ablation", "confirm", "optional", "template"}
RUN_PRIORITIES = {"must", "should", "optional", "none"}

RESERVED_TOKENS = ("<pad>", "<eos>", "<unk>")
SMOKE_TOKENIZER_NAME = "local_toy_whitespace"
SMOKE_BACKBONE_NAME = "local_toy_gpt"


SCHEMA_PATHS = {
    "dataset_manifest": SCHEMAS_ROOT / "dataset_manifest.schema.json",
    "run_manifest": SCHEMAS_ROOT / "run_manifest.schema.json",
    "future_metrics": SCHEMAS_ROOT / "future_metrics.schema.json",
    "acceptance_metrics": SCHEMAS_ROOT / "acceptance_metrics.schema.json",
    "router_metrics": SCHEMAS_ROOT / "router_metrics.schema.json",
    "finalist_selection": SCHEMAS_ROOT / "finalist_selection.schema.json",
    "paper_asset_manifest": SCHEMAS_ROOT / "paper_asset_manifest.schema.json",
}

