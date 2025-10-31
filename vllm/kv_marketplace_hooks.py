# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks and helpers for kv-marketplace integration."""

import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from vllm.kv_marketplace_shim import load_plugin

if TYPE_CHECKING:
    from vllm.v1.request import Request
    from vllm.config import VllmConfig

logger = None


def _get_logger():
    global logger
    if logger is None:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
    return logger


def _device_id_from_ctx(engine_ctx: Any) -> int:
    """Extract device ID from engine context."""
    # Prefer whatever your worker/executor exposes; fallback to torch
    dev = getattr(engine_ctx, "device", None)
    if isinstance(dev, int):
        return dev
    if hasattr(engine_ctx, "device_id"):
        return engine_ctx.device_id
    try:
        return torch.cuda.current_device()
    except Exception:
        return 0


def _compat_from_ctx(engine_ctx: Any):
    """Build KVCompat-compatible dict from engine context."""
    cfg = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "config", None)
    mc = getattr(cfg, "model_config", None) if cfg else None
    cc = getattr(cfg, "cache_config", None) if cfg else None
    tok = getattr(cfg, "tokenizer", None) or getattr(cfg, "tokenizer_config", None)

    return {
        "model_params": {
            "n_layers": getattr(mc, "num_hidden_layers", None) or getattr(mc, "n_layers", None),
            "hidden_size": getattr(mc, "hidden_size", None),
            "n_kv_heads": getattr(mc, "num_key_value_heads", None) or getattr(mc, "n_kv_heads", None),
            "head_dim": getattr(mc, "head_dim", None),
            "rope_base": getattr(mc, "rope_theta", None) or getattr(mc, "rope_base", None),
            "rope_scaling": getattr(mc, "rope_scaling", None),
            "alibi": getattr(mc, "alibi", False),
        },
        "tokenizer_config": {
            "name_or_path": getattr(tok, "name_or_path", None) if tok else None,
            "normalizer": getattr(tok, "normalizer", None) if tok else None,
            "vocab_size": getattr(tok, "vocab_size", None) or getattr(mc, "vocab_size", None) if mc else None,
        },
        "kv_layout": {
            "page_size": getattr(cc, "block_size", None) or getattr(cc, "page_size", None),
            "dtype": str(getattr(cc, "cache_dtype", None) or getattr(cc, "dtype", None)),
            "layout": "paged",
        },
    }


def _layout_from_ctx(engine_ctx: Any):
    """Extract KV layout from engine context."""
    cfg = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "config", None)
    mc = getattr(cfg, "model_config", None) if cfg else None
    cc = getattr(cfg, "cache_config", None) if cfg else None
    return {
        "n_layers": getattr(mc, "num_hidden_layers", None) or getattr(mc, "n_layers", 0) if mc else 0,
        "n_kv_heads": getattr(mc, "num_key_value_heads", None) or getattr(mc, "n_kv_heads", 0) if mc else 0,
        "head_dim": getattr(mc, "head_dim", 0) if mc else 0,
        "page_size": getattr(cc, "block_size", 0) or getattr(cc, "page_size", 0) if cc else 0,
        "strides": {},
    }


def _get_prefill_stream_ptr(device_id: int) -> int:
    """Get CUDA stream pointer for prefill."""
    try:
        s = torch.cuda.current_stream(device=device_id)
        # Try to get the stream pointer
        if hasattr(s, "cuda_stream"):
            return int(s.cuda_stream)
        elif hasattr(s, "_cudastream"):
            return int(s._cudastream)
        # Fallback: return 0 for default stream
        return 0
    except Exception:
        return 0


def _allocator_closure(engine_ctx: Any, device_id: int):
    """Create allocator closure for prefix allocation."""
    # You need a way to allocate/mark prefix pages as materialized.
    # Expose a small helper in your fork if needed; here we mock an interface.
    alloc = getattr(engine_ctx, "kv_allocator", None)
    kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
    
    def alloc_prefix(length: int):
        # Your allocator should return per-layer pointers or page handles
        k_ptrs, v_ptrs = [], []
        if alloc is not None:
            # Example interface; replace with your actual calls:
            # pages = alloc.allocate_prefix(length)
            # k_ptrs = [p.k_ptr for p in pages_per_layer]
            # v_ptrs = [p.v_ptr for p in pages_per_layer]
            pass
        elif kv_cache_manager is not None:
            # Try to use kv_cache_manager to allocate blocks
            # This is a placeholder - actual implementation depends on vLLM internals
            pass
        return {"k_ptrs": k_ptrs, "v_ptrs": v_ptrs, "length": length}
    return alloc_prefix


def make_import_ctx(
    req: "Request",
    engine_ctx: Any,  # EngineCore or similar
) -> Optional[dict[str, Any]]:
    """Create import context for before_prefill hook.
    
    Args:
        req: The request being processed
        engine_ctx: Engine context (may contain model config, flags, etc.)
        
    Returns:
        Context dict for before_prefill, or None if context cannot be built
    """
    plugin = load_plugin()
    if not plugin:
        return None
    
    try:
        flags = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "flags", None)
        if not flags:
            return None
        
        device_id = _device_id_from_ctx(engine_ctx)
        tokens = getattr(req, "prompt_token_ids", None) or getattr(req, "prompt", []) or []
        compat = _compat_from_ctx(engine_ctx)
        layout = _layout_from_ctx(engine_ctx)
        alloc_prefix = _allocator_closure(engine_ctx, device_id)
        stream = _get_prefill_stream_ptr(device_id)
        
        return {
            "device_id": device_id,
            "compat": compat,
            "tokens": tokens,
            "alloc_prefix": alloc_prefix,
            "layout": layout,
            "stream": stream,
        }
    except Exception as e:
        _get_logger().warning(
            f"Failed to build import context for kv-marketplace: {e}"
        )
        return None


def make_export_ctx(
    req: "Request",
    engine_ctx: Any,  # EngineCore or similar
) -> Optional[dict[str, Any]]:
    """Create export context for after_prefill hook.
    
    Args:
        req: The request that just completed prefill
        engine_ctx: Engine context
        
    Returns:
        Context dict for after_prefill, or None if context cannot be built
    """
    plugin = load_plugin()
    if not plugin:
        return None
    
    try:
        flags = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "flags", None)
        if not flags:
            return None
        
        device_id = _device_id_from_ctx(engine_ctx)
        tokens = getattr(req, "prompt_token_ids", None) or []
        compat = _compat_from_ctx(engine_ctx)
        layout = _layout_from_ctx(engine_ctx)
        
        # Pull the pages/pointers the allocator just filled during prefill
        # Replace this with your real exposure (e.g., engine_ctx.kv_allocator.get_prefix_handles(req))
        kv_pages = {"k_ptrs": [], "v_ptrs": [], "length": len(tokens)}
        
        # Try to get actual KV cache pointers if available
        kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
        if kv_cache_manager and hasattr(kv_cache_manager, "get_prefill_pages"):
            kv_pages = kv_cache_manager.get_prefill_pages(req)
        
        length = kv_pages.get("length", len(tokens))
        
        return {
            "device_id": device_id,
            "compat": compat,
            "tokens": tokens,
            "kv_pages": kv_pages,
            "layout": layout,
            "length": length,
        }
    except Exception as e:
        _get_logger().warning(
            f"Failed to build export context for kv-marketplace: {e}"
        )
        return None


def _maybe_import_prefix(
    req: "Request",
    engine_ctx: Any,
) -> Optional[Tuple[int, Any]]:
    """Hook called before prefill to attempt KV cache import.
    
    Args:
        req: The request being processed
        engine_ctx: Engine context
        
    Returns:
        Tuple of (lcp_len, dst_alloc) if import succeeds, None otherwise
    """
    plugin = load_plugin()
    if not plugin:
        return None
    
    # Check if kv-marketplace is enabled
    flags = getattr(engine_ctx, "vllm_config", None)
    if flags is None:
        flags = getattr(engine_ctx, "flags", None)
    
    if not flags or not getattr(flags, "kv_marketplace", False):
        return None
    
    ctx_dict = make_import_ctx(req, engine_ctx)
    if ctx_dict is None:
        return None
    
    try:
        # Call the plugin's before_prefill function
        result = plugin.before_prefill(ctx_dict)
        
        if result is not None:
            lcp_len, dst_alloc = result
            
            # Update request to skip prefix tokens
            if hasattr(req, "seq_pos"):
                req.seq_pos = lcp_len
            if hasattr(req, "prompt_token_ids") and req.prompt_token_ids:
                req.prompt_token_ids = req.prompt_token_ids[lcp_len:]
            
            # Inform allocator that prefix pages are materialized
            kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
            if kv_cache_manager and hasattr(kv_cache_manager, "materialize_prefix"):
                kv_cache_manager.materialize_prefix(req, dst_alloc, lcp_len)
            
            _get_logger().info(
                f"kv-marketplace: Imported prefix of length {lcp_len} for request {req.request_id if hasattr(req, 'request_id') else 'unknown'}"
            )
            return result
    except Exception as e:
        _get_logger().warning(
            f"kv-marketplace before_prefill failed: {e}"
        )
    
    return None


def _export_prefix(
    req: "Request",
    engine_ctx: Any,
) -> None:
    """Hook called after prefill to export KV cache.
    
    Args:
        req: The request that just completed prefill
        engine_ctx: Engine context
    """
    plugin = load_plugin()
    if not plugin:
        return
    
    flags = getattr(engine_ctx, "vllm_config", None)
    if flags is None:
        flags = getattr(engine_ctx, "flags", None)
    
    if not flags or not getattr(flags, "kv_marketplace", False):
        return
    
    ctx_dict = make_export_ctx(req, engine_ctx)
    if ctx_dict is None:
        return
    
    try:
        plugin.after_prefill(ctx_dict)
        _get_logger().info(
            f"kv-marketplace: Exported prefix of length {ctx_dict['length']} for request {req.request_id if hasattr(req, 'request_id') else 'unknown'}"
        )
    except Exception as e:
        _get_logger().warning(
            f"kv-marketplace after_prefill failed: {e}"
        )
