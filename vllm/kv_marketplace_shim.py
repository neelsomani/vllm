# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin loader for kv-marketplace integration."""

from typing import Any


def load_plugin() -> Any:
    """Load the kv-marketplace plugin if available.
    
    Returns:
        The kv-marketplace adapter module if available, None otherwise.
    """
    try:
        from kv_marketplace.adapter import vllm as kvm
        return kvm
    except Exception:
        return None

