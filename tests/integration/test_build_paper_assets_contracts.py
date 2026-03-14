from __future__ import annotations

import json

from tests.helpers import make_fake_run, run_python_module


def test_build_paper_assets_emits_manifest_with_figures_tables_and_appendix(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    make_fake_run(output_root, exp_id="FINAL_A", stage="final", model_id="model-a", with_probe=True, with_png=True)

    run_python_module("src.cli.build_paper_assets", "--outputs-root", str(output_root))

    manifest_path = output_root / "paper_assets" / "paper_asset_manifest.json"
    claim_matrix = output_root / "paper_assets" / "appendix" / "claim_evidence_matrix.md"
    assert manifest_path.exists()
    assert claim_matrix.exists()
    assert "pending" not in claim_matrix.read_text(encoding="utf-8")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["figures"], "expected at least one figure entry from the fake PNG"
    assert payload["tables"], "expected exported table entries"
    assert payload["appendix"], "expected appendix entries"
    table_names = {entry["name"] for entry in payload["tables"]}
    assert "table_main_results.csv" in table_names
    assert any(entry["path"].endswith(".png") for entry in payload["figures"])
