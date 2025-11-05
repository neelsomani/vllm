# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks and helpers for kv-marketplace integration."""

import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from vllm.kv_marketplace_shim import load_plugin

if TYPE_CHECKING:
    from vllm.v1.request import Request
    from vllm.config import VllmConfig

try:
    from kv_marketplace.compat import KVCompat
except ImportError:
    KVCompat = None

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

    # Extract RoPE configuration separately
    rope_base = getattr(mc, "rope_theta", None) or getattr(mc, "rope_base", None)
    rope_scaling = getattr(mc, "rope_scaling", None)
    rope_config = {}
    if rope_base is not None:
        rope_config["rope_base"] = rope_base
    if rope_scaling is not None:
        rope_config["rope_scaling"] = rope_scaling

    return {
        "model_params": {
            "n_layers": getattr(mc, "num_hidden_layers", None) or getattr(mc, "n_layers", None),
            "hidden_size": getattr(mc, "hidden_size", None),
            "n_kv_heads": getattr(mc, "num_key_value_heads", None) or getattr(mc, "n_kv_heads", None),
            "head_dim": getattr(mc, "head_dim", None),
            "alibi": getattr(mc, "alibi", False),
        },
        "tokenizer_config": {
            "name_or_path": getattr(tok, "name_or_path", None) if tok else None,
            "normalizer": getattr(tok, "normalizer", None) if tok else None,
            "vocab_size": getattr(tok, "vocab_size", None) or getattr(mc, "vocab_size", None) if mc else None,
        },
        "rope_config": rope_config,
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


def _allocator_closure(engine_ctx: Any, device_id: int, req: "Request"):
    """Create allocator closure for prefix allocation.
    
    This allocates KV cache blocks for the prefix that will be imported.
    The blocks are allocated but not yet filled - the import will copy
    data into them.
    """
    kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
    
    def alloc_prefix(length: int):
        """Allocate KV cache blocks for the prefix.
        
        Args:
            length: Number of tokens to allocate
            
        Returns:
            AllocatedKV dict with k_ptrs and v_ptrs per layer
        """
        k_ptrs, v_ptrs = [], []
        
        if kv_cache_manager is None:
            _get_logger().warning("kv_cache_manager not available for prefix allocation")
            return {"k_ptrs": k_ptrs, "v_ptrs": v_ptrs, "length": length}
        
        try:
            # Calculate number of blocks needed
            block_size = kv_cache_manager.block_size
            if block_size is None or block_size == 0:
                _get_logger().warning(f"Invalid block_size: {block_size}")
                return {"k_ptrs": k_ptrs, "v_ptrs": v_ptrs, "length": length}
            
            num_blocks = (length + block_size - 1) // block_size
            
            # Allocate blocks using the cache manager
            # We allocate a temporary set of blocks for the prefix
            # The materialize_prefix method will later install these properly
            from vllm.v1.request import Request as RequestType
            from vllm.v1.core.kv_cache_manager import KVCacheBlocks
            from vllm.v1.core.kv_cache_utils import KVCacheBlock
            
            # Create a temporary request-like object to allocate blocks
            # In practice, the blocks should be allocated as part of the normal flow
            # but marked as "pre-allocated" for the prefix region
            # For now, we return empty pointers - the actual allocation happens
            # in the scheduler's allocate_slots, and materialize_prefix will
            # install the imported data
            
            # Get layout info
            cfg = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "config", None)
            mc = getattr(cfg, "model_config", None) if cfg else None
            n_layers = getattr(mc, "num_hidden_layers", None) or getattr(mc, "n_layers", 0) if mc else 0
            
            # Initialize empty pointers per layer
            # The actual pointers will be set by materialize_prefix
            k_ptrs = [0] * n_layers
            v_ptrs = [0] * n_layers
            
        except Exception as e:
            _get_logger().warning(f"Error allocating prefix blocks: {e}", exc_info=True)
        
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
        
        # Set min prefix length from flags if available
        if plugin and hasattr(plugin, "set_min_prefix_length"):
            min_prefix = getattr(flags, "kv_min_prefix", 64)
            plugin.set_min_prefix_length(min_prefix)
        
        device_id = _device_id_from_ctx(engine_ctx)
        tokens = getattr(req, "prompt_token_ids", None) or getattr(req, "prompt", []) or []
        
        # Store original prompt tokens and length for later use (before potential slicing)
        if not hasattr(req, "_orig_prompt_len"):
            req._orig_prompt_len = len(tokens)
        if not hasattr(req, "_orig_prompt_token_ids"):
            req._orig_prompt_token_ids = list(tokens)  # Make a copy
        
        compat_dict = _compat_from_ctx(engine_ctx)
        layout = _layout_from_ctx(engine_ctx)
        alloc_prefix = _allocator_closure(engine_ctx, device_id, req)
        stream = _get_prefill_stream_ptr(device_id)
        
        # Convert compat dict to KVCompat object
        if KVCompat is not None:
            compat = KVCompat(
                model_params=compat_dict.get("model_params", {}),
                tokenizer_config=compat_dict.get("tokenizer_config", {}),
                rope_config=compat_dict.get("rope_config", {}),
                layout_config=compat_dict.get("kv_layout", {})
            )
        else:
            # Fallback if KVCompat not available
            compat = compat_dict
        
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
        
        # Use original prompt tokens and length for export (before they were sliced)
        orig_tokens = getattr(req, "_orig_prompt_token_ids", None)
        orig_len = getattr(req, "_orig_prompt_len", None)
        current_tokens = getattr(req, "prompt_token_ids", None) or []
        
        if orig_tokens is not None and orig_len is not None and orig_len > 0:
            # Use original tokens for export (before they were sliced by import)
            tokens = orig_tokens
            prompt_len = orig_len
        else:
            # Fallback to current tokens if original not available
            tokens = current_tokens
            prompt_len = len(tokens)
        
        compat_dict = _compat_from_ctx(engine_ctx)
        layout = _layout_from_ctx(engine_ctx)
        
        # Pull the pages/pointers the allocator just filled during prefill
        kv_pages = {"k_ptrs": [], "v_ptrs": [], "length": prompt_len}
        
        # Try to get actual KV cache pointers if available
        kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
        if kv_cache_manager and hasattr(kv_cache_manager, "get_prefill_pages"):
            kv_pages = kv_cache_manager.get_prefill_pages(req)
        
        # Ensure we use the correct length
        length = kv_pages.get("length", prompt_len)
        
        # Convert compat dict to KVCompat object
        if KVCompat is not None:
            compat = KVCompat(
                model_params=compat_dict.get("model_params", {}),
                tokenizer_config=compat_dict.get("tokenizer_config", {}),
                rope_config=compat_dict.get("rope_config", {}),
                layout_config=compat_dict.get("kv_layout", {})
            )
        else:
            # Fallback if KVCompat not available
            compat = compat_dict
        
        return {
            "device_id": device_id,
            "compat": compat,
            "tokens": tokens[:length] if len(tokens) >= length else tokens,  # Use prefix tokens up to length
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
