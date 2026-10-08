from __future__ import annotations

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')

import json

import numpy as np

from gru import NumpyGRU


def main() -> None:
    rng = np.random.default_rng(113)
    model = NumpyGRU(4, 5, 6, 9, 20260924)
    seq = rng.normal(size=(3, 4, 4))
    context = rng.normal(size=(3, 5))
    truth = rng.normal(size=(3, 9))
    _, analytical = model.loss_and_grad(seq, context, truth)
    h = 1e-6
    errors = {}
    for name, param in model.p.items():
        coords = list(np.ndindex(param.shape))
        chosen = coords[: min(5, len(coords))]
        worst = 0.0
        for ix in chosen:
            original = param[ix]
            param[ix] = original + h
            plus = model.loss_and_grad(seq, context, truth)[0]
            param[ix] = original - h
            minus = model.loss_and_grad(seq, context, truth)[0]
            param[ix] = original
            numeric = (plus - minus) / (2 * h)
            scale = max(1e-8, abs(numeric), abs(analytical[name][ix]))
            worst = max(worst, abs(numeric - analytical[name][ix]) / scale)
        errors[name] = worst
    result = {"max_relative_error": float(max(errors.values())), "by_parameter": errors,
              "pass": bool(max(errors.values()) < 1e-5)}
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
