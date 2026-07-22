"""Loader for the Julia custom loss functions in models/losses/.

Each .jl file is a full PySR `loss_function_expression` with hyperparameters
left as __UPPERCASE__ placeholder tokens (e.g. __NU_T__). load_loss() fills
them from keyword arguments: nu_t=3.0 fills __NU_T__.
"""
import re
from pathlib import Path

LOSS_DIR = Path(__file__).resolve().parent / "losses"

_PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")


def load_loss(name: str, **params) -> str:
    src = (LOSS_DIR / f"{name}.jl").read_text()
    for key, val in params.items():
        token = f"__{key.upper()}__"
        if token not in src:
            raise KeyError(f"{name}.jl has no placeholder {token}")
        src = src.replace(token, str(val))
    leftover = sorted(set(_PLACEHOLDER.findall(src)))
    if leftover:
        raise KeyError(f"{name}.jl: unfilled placeholders {leftover}")
    return src
