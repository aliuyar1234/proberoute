from __future__ import annotations


def compute_router_metrics(model, *, exp_id: str, top_m: int) -> dict:
    weights, diagnostics = model.router.mixture()
    return {
        "exp_id": exp_id,
        "layer_mix_mode": model.layer_mix_mode,
        "horizons": model.horizons,
        "top_m": top_m,
        "entropy_by_horizon": diagnostics["entropy_by_horizon"],
        "selected_layers_by_horizon": diagnostics["selected_layers_by_horizon"],
        "average_weights_by_horizon": diagnostics["average_weights_by_horizon"],
        "overlap_with_probe_topm_by_horizon": None,
    }

