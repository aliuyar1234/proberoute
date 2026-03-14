from __future__ import annotations

import inspect

import torch

from tests.helpers import call_with_known_kwargs, first_existing_attr


def _instantiate_router(layer_mix_mode: str, router_init_mode: str = "random"):
    from src.models.router import LayerRouter

    return call_with_known_kwargs(
        LayerRouter,
        num_layers=6,
        horizons=[1, 2, 3, 4],
        layer_mix_mode=layer_mix_mode,
        router_init_mode=router_init_mode,
        top_m=2,
    )


def _weights_from_router(router) -> torch.Tensor:
    accessor = first_existing_attr(router, ["get_weights", "normalized_weights", "weights", "forward"])
    if callable(accessor):
        signature = inspect.signature(accessor)
        payload = accessor() if len(signature.parameters) == 0 else accessor(None)
    else:
        payload = accessor
    if isinstance(payload, dict):
        for key in ["weights", "router_weights", "normalized_weights"]:
            if key in payload:
                return payload[key]
    return payload


def test_sparse_router_keeps_exactly_top_m_nonzero_weights() -> None:
    router = _instantiate_router("sparse_topm")
    weights = _weights_from_router(router)
    assert tuple(weights.shape) == (4, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.equal((weights > 0).sum(dim=-1), torch.full((4,), 2))


def test_dense_router_uses_all_layers() -> None:
    router = _instantiate_router("dense_whs")
    weights = _weights_from_router(router)
    assert tuple(weights.shape) == (4, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.equal((weights > 0).sum(dim=-1), torch.full((4,), 6))


def test_last_layer_router_selects_only_the_final_layer() -> None:
    router = _instantiate_router("last_layer", router_init_mode="none")
    weights = _weights_from_router(router)
    assert tuple(weights.shape) == (4, 6)
    assert torch.equal((weights > 0).sum(dim=-1), torch.ones(4, dtype=torch.long))
    assert torch.allclose(weights[:, -1], torch.ones(4), atol=1e-6)
