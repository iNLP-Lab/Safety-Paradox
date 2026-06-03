"""
vLLM's import_triton_kernels() prefers site-packages `triton_kernels` over
`vllm.third_party.triton_kernels`. Some environments ship an older/partial
tree (no SparseMatrix), which breaks Gemma 4 MoE. If we detect that, alias the
vendor copy before `import vllm` loads fused_moe.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys


def apply() -> None:
    if importlib.util.find_spec("triton_kernels") is None:
        return
    try:
        import triton_kernels.tensor as tkt  # noqa: PLC0415
    except Exception:
        return
    if getattr(tkt, "SparseMatrix", None) is not None:
        return
    for name in list(sys.modules):
        if name == "triton_kernels" or name.startswith("triton_kernels."):
            del sys.modules[name]

    tk = importlib.import_module("vllm.third_party.triton_kernels")
    sys.modules["triton_kernels"] = tk


apply()
