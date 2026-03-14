from __future__ import annotations

import numpy as np


def percentile_bootstrap(values: list[float], *, samples: int, seed: int) -> dict[str, float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    boots = []
    for _ in range(samples):
        draw = rng.choice(array, size=len(array), replace=True)
        boots.append(float(draw.mean()))
    return {
        "lower": float(np.percentile(boots, 2.5)),
        "mean": float(np.mean(boots)),
        "upper": float(np.percentile(boots, 97.5)),
    }

