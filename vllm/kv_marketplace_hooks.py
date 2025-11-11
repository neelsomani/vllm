# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks and helpers for kv-marketplace integration."""

import time
import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from vllm.kv_marketplace_shim import load_plugin

print(f"kv-mkt hooks module path: {__file__}", flush=True)

if TYPE_CHECKING:
    from vllm.v1.request import Request
    from vllm.config import VllmConfig

try:
    from kv_marketplace.compat import KVCompat
except ImportError:
    KVCompat = None

logger = None
_PRINTED_HEAD_INFO = False


def _get_logger():
    global logger
    if logger is None:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
    return logger


def _dbg(msg: str, *args):
    text = msg % args if args else msg
    print(f"[kv-mkt dbg] {text}")


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
    if cfg is None:
        raise RuntimeError("[kv-mkt] engine_ctx has no vllm_config/config")
    
    mc = getattr(cfg, "model_config", None)
    cc = getattr(cfg, "cache_config", None)
    tok = getattr(cfg, "tokenizer", None) or getattr(cfg, "tokenizer_config", None)
    
    if mc is None:
        raise RuntimeError("[kv-mkt] model_config is None")
    if cc is None:
        raise RuntimeError("[kv-mkt] cache_config is None")

    # Extract RoPE configuration separately
    rope_base = getattr(mc, "rope_theta", None) or getattr(mc, "rope_base", None)
    rope_scaling = getattr(mc, "rope_scaling", None)
    rope_config = {}
    if rope_base is not None:
        rope_config["rope_base"] = rope_base
    if rope_scaling is not None:
        rope_config["rope_scaling"] = rope_scaling

    # Use getters - access hf_text_config for total values
    hf_text_config = getattr(mc, "hf_text_config", None)
    if hf_text_config is None:
        raise RuntimeError("[kv-mkt] model_config has no hf_text_config")
    
    n_layers = getattr(hf_text_config, "num_hidden_layers", None) or getattr(hf_text_config, "n_layers", None)
    if n_layers is None:
        raise RuntimeError("[kv-mkt] Failed to extract n_layers from hf_text_config")
    
    hidden_size = mc.get_hidden_size()
    if hidden_size is None or hidden_size == 0:
        raise RuntimeError(f"[kv-mkt] get_hidden_size() returned invalid value: {hidden_size}")
    
    n_kv_heads = mc.get_total_num_kv_heads()
    if n_kv_heads is None or n_kv_heads == 0:
        raise RuntimeError(f"[kv-mkt] get_total_num_kv_heads() returned invalid value: {n_kv_heads}")
    
    head_dim = mc.get_head_size()
    if head_dim is None or head_dim == 0:
        n_heads = getattr(hf_text_config, "num_attention_heads", None) or getattr(hf_text_config, "n_heads", None)
        if n_heads is None or n_heads == 0:
            raise RuntimeError("[kv-mkt] Failed to extract n_heads for head_dim calculation")
        head_dim = hidden_size // n_heads
        if head_dim == 0:
            raise RuntimeError(f"[kv-mkt] Computed head_dim is 0: hidden_size={hidden_size}, n_heads={n_heads}")
    
    vocab_size = None
    if tok and getattr(tok, "vocab_size", None):
        vocab_size = tok.vocab_size
    else:
        try:
            vocab_size = mc.get_vocab_size()
        except Exception:
            pass

    page_size = getattr(cc, "block_size", None) or getattr(cc, "page_size", None)
    if page_size is None or page_size == 0:
        raise RuntimeError("[kv-mkt] Failed to extract page_size from cache_config")
    
    dtype = getattr(cc, "cache_dtype", None) or getattr(cc, "dtype", None)
    if dtype is None:
        raise RuntimeError("[kv-mkt] Failed to extract dtype from cache_config")
    dtype = str(dtype)

    return {
        "model_params": {
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "alibi": getattr(mc, "alibi", False),
        },
        "tokenizer_config": {
            "name_or_path": getattr(tok, "name_or_path", None) if tok else None,
            "normalizer": getattr(tok, "normalizer", None) if tok else None,
            "vocab_size": vocab_size,
        },
        "rope_config": rope_config,
        "kv_layout": {
            "page_size": page_size,
            "dtype": dtype,
            "layout": "paged",
        },
    }


def _layout_from_ctx(engine_ctx: Any):
    """Extract KV layout from engine context.
    
    Uses vLLM ModelConfig getters to extract layout information.
    """
    cfg = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "config", None)
    if cfg is None:
        raise RuntimeError(f"[kv-mkt] engine_ctx has no vllm_config/config: {type(engine_ctx)}")
    
    mc = getattr(cfg, "model_config", None)
    cc = getattr(cfg, "cache_config", None)
    
    if mc is None:
        raise RuntimeError(f"[kv-mkt] engine_ctx.config.model_config is None (cfg={cfg})")
    if cc is None:
        raise RuntimeError(f"[kv-mkt] engine_ctx.config.cache_config is None (cfg={cfg})")

    # Use vLLM accessors. These are the source of truth.
    hf_text_config = getattr(mc, "hf_text_config", None)
    if hf_text_config is None:
        raise RuntimeError("[kv-mkt] model_config has no hf_text_config")
    
    try:
        # Get total values from hf_text_config (not per-GPU values)
        n_layers = getattr(hf_text_config, "num_hidden_layers", None) or getattr(hf_text_config, "n_layers", None)
        if n_layers is None or n_layers == 0:
            raise RuntimeError("[kv-mkt] Failed to extract n_layers from hf_text_config")
        
        n_heads = getattr(hf_text_config, "num_attention_heads", None) or getattr(hf_text_config, "n_heads", None)
        if n_heads is None or n_heads == 0:
            raise RuntimeError("[kv-mkt] Failed to extract n_heads from hf_text_config")
        
        # Use getter for total KV heads
        n_kv_heads = mc.get_total_num_kv_heads()
        if n_kv_heads is None or n_kv_heads == 0:
            # Fallback to hf_text_config attributes
            n_kv_heads = (getattr(hf_text_config, "num_key_value_heads", None) or
                         getattr(hf_text_config, "num_kv_heads", None) or
                         getattr(hf_text_config, "n_head_kv", None) or
                         n_heads)
        if n_kv_heads is None or n_kv_heads == 0:
            raise RuntimeError("[kv-mkt] Failed to extract n_kv_heads")
        
        # Use getter for head_dim
        head_dim = mc.get_head_size()
    except Exception as e:
        raise RuntimeError(f"[kv-mkt] failed to read ModelConfig via getters: {e}")

    # Head dim fallback if getter not implemented by a backend
    if head_dim is None or head_dim == 0:
        hidden_size = mc.get_hidden_size()
        if hidden_size is None or hidden_size == 0:
            raise RuntimeError(f"[kv-mkt] get_hidden_size() returned invalid value: {hidden_size}")
        if n_heads is None or n_heads == 0:
            raise RuntimeError(f"[kv-mkt] n_heads is invalid: {n_heads}")
        head_dim = hidden_size // n_heads
        if head_dim == 0:
            raise RuntimeError(f"[kv-mkt] Computed head_dim is 0: hidden_size={hidden_size}, n_heads={n_heads}")

    # Page size from cache_config, with fallback to kv_cache_manager.block_size
    page_size = getattr(cc, "block_size", None) or getattr(cc, "page_size", None)
    if page_size is None or page_size == 0:
        kvm = None
        if hasattr(engine_ctx, "scheduler"):
            kvm = getattr(engine_ctx.scheduler, "kv_cache_manager", None)
        if not kvm:
            kvm = getattr(engine_ctx, "kv_cache_manager", None)
        page_size = getattr(kvm, "block_size", None)
        if page_size is None or page_size == 0:
            raise RuntimeError("[kv-mkt] Failed to extract page_size from cache_config or kv_cache_manager")

    # Validate all values are valid integers > 0
    for k, v in {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "page_size": page_size,
    }.items():
        if not isinstance(v, int) or v <= 0:
            raise RuntimeError(f"[kv-mkt] invalid {k}={v}")

    return {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
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
    # Mirror the export lookup: try scheduler first, then direct access
    kv_cache_manager = None
    if hasattr(engine_ctx, "scheduler"):
        kv_cache_manager = getattr(engine_ctx.scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
    engine_core = getattr(engine_ctx, "engine_core", None)
    model_executor = getattr(engine_core, "model_executor", None) if engine_core else None
    
    def alloc_prefix(length: int):
        """Allocate KV cache blocks for the prefix.
        
        Args:
            length: Number of tokens to allocate
            
        Returns:
            AllocatedKV dict with k_ptrs and v_ptrs per layer
        """
        if kv_cache_manager is None:
            raise RuntimeError("kv_cache_manager not available; cannot allocate prefix")
        
        try:
            return kv_cache_manager.reserve_prefix(
                req, length, model_executor=model_executor
            )
        except Exception as exc:
            _get_logger().warning(
                f"[kv-mkt] reserve_prefix failed (length={length}): {exc}"
            )
            if hasattr(kv_cache_manager, "release_reserved_prefix"):
                kv_cache_manager.release_reserved_prefix(req)
            return {"k_ptrs": [], "v_ptrs": [], "length": 0}
    
    return alloc_prefix


def _get_cached_engine_ctx_data(engine_ctx: Any, device_id: int):
    """Cache heavy, request-invariant data (compat/layout/stream) on engine_ctx."""
    cache = getattr(engine_ctx, "_kv_mkt_cached_ctx", None)
    if cache is None:
        cache = {
            "compat_obj": None,
            "layout": None,
            "stream_ptrs": {},
        }
        setattr(engine_ctx, "_kv_mkt_cached_ctx", cache)

    if cache.get("compat_obj") is None:
        compat_dict = _compat_from_ctx(engine_ctx)
        cache["compat_dict"] = compat_dict
        if KVCompat is not None:
            cache["compat_obj"] = KVCompat(
                model_params=compat_dict.get("model_params", {}),
                tokenizer_config=compat_dict.get("tokenizer_config", {}),
                rope_config=compat_dict.get("rope_config", {}),
                layout_config=compat_dict.get("kv_layout", {}),
            )
        else:
            cache["compat_obj"] = compat_dict

    if cache.get("layout") is None:
        cache["layout"] = _layout_from_ctx(engine_ctx)

    stream_ptrs = cache.setdefault("stream_ptrs", {})
    if device_id not in stream_ptrs:
        stream_ptrs[device_id] = _get_prefill_stream_ptr(device_id)

    return cache["compat_obj"], cache["layout"], stream_ptrs[device_id]


def warm_kv_marketplace_ctx(engine_ctx: Any) -> None:
    """Eagerly populate cached ctx data so the first request avoids the cost."""
    plugin = load_plugin()
    if not plugin:
        return

    try:
        device_id = _device_id_from_ctx(engine_ctx)
        _get_cached_engine_ctx_data(engine_ctx, device_id)
    except Exception as exc:
        _get_logger().debug(
            "kv-marketplace warm_kv_marketplace_ctx skipped: %s", exc
        )


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
    
    print(
        f"kv-mkt make_import_ctx: entered req={getattr(req, 'request_id', 'unknown')}",
        flush=True,
    )
    try:
        flags = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "flags", None)
        if not flags:
            _dbg("make_import_ctx: missing flags for req=%s", getattr(req, "request_id", "unknown"))
            return None
        _dbg(
            "make_import_ctx [NEW VERSION LOADED]: req=%s kv_min_prefix=%s",
            getattr(req, "request_id", "unknown"),
            getattr(flags, "kv_min_prefix", None),
        )
        
        # Set min prefix length from flags if available
        if plugin and hasattr(plugin, "set_min_prefix_length"):
            min_prefix = getattr(flags, "kv_min_prefix", 64)
            plugin.set_min_prefix_length(min_prefix)
            _dbg(f"Set min prefix to {min_prefix}")
        
        device_id = _device_id_from_ctx(engine_ctx)
        tokens = getattr(req, "prompt_token_ids", None) or getattr(req, "prompt", []) or []
        _dbg(f"Tokens: {tokens}")
        # Store original prompt tokens and length for later use (before potential slicing)
        if not hasattr(req, "_orig_prompt_len"):
            req._orig_prompt_len = len(tokens)
        if not hasattr(req, "_orig_prompt_token_ids"):
            req._orig_prompt_token_ids = list(tokens)  # Make a copy
        
        compat, layout, stream = _get_cached_engine_ctx_data(engine_ctx, device_id)
        alloc_prefix = _allocator_closure(engine_ctx, device_id, req)
        
        return {
            "device_id": device_id,
            "compat": compat,
            "tokens": tokens,
            "alloc_prefix": alloc_prefix,
            "layout": layout,
            "stream": stream,
        }
    except Exception as e:
        _dbg(
            "make_import_ctx: exception for req=%s -> %s",
            getattr(req, "request_id", "unknown"),
            e,
        )
        traceback.print_exc()
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
    print("kv-mkt make_export_ctx: entered", flush=True)
    plugin = load_plugin()
    if not plugin:
        print("kv-mkt make_export_ctx: plugin missing", flush=True)
        return None
    
    try:
        flags = getattr(engine_ctx, "vllm_config", None) or getattr(engine_ctx, "flags", None)
        req_id = getattr(req, "request_id", "unknown")
        if not flags:
            _dbg("make_export_ctx: missing flags for req=%s", req_id)
            print(f"kv-mkt make_export_ctx: missing flags req={req_id}", flush=True)
            return None
        print(f"kv-mkt make_export_ctx: start req={req_id}", flush=True)
        _dbg(
            "make_export_ctx: req=%s kv_min_prefix=%s",
            req_id,
            getattr(flags, "kv_min_prefix", None),
        )
        
        device_id = _device_id_from_ctx(engine_ctx)
        
        print(f"kv-mkt make_export_ctx: determining tokens req={req_id}", flush=True)
        print(f"kv-mkt make_export_ctx: determining tokens req={req_id}", flush=True)
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
        _dbg(
            "make_export_ctx: req=%s prompt_len=%d orig_len=%s token_buf=%d",
            req_id,
            prompt_len,
            orig_len,
            len(tokens),
        )
        print(
            f"kv-mkt make_export_ctx: after tokens req={req_id} prompt_len={prompt_len} orig_len={orig_len}",
            flush=True,
        )
        
        print(f"kv-mkt make_export_ctx: fetching compat/layout req={req_id}", flush=True)
        compat, layout, _ = _get_cached_engine_ctx_data(engine_ctx, device_id)
        print(f"kv-mkt make_export_ctx: have compat/layout req={req_id}", flush=True)

        # Pull the pages/pointers the allocator just filled during prefill
        kv_pages = {"k_ptrs": [], "v_ptrs": [], "length": prompt_len}
        print(f"kv-mkt make_export_ctx: initial kv_pages empty req={req_id}", flush=True)
        
        # Try to get actual KV cache pointers if available
        # kv_cache_manager is in scheduler, not directly in EngineCore
        kv_cache_manager = None
        if hasattr(engine_ctx, "scheduler"):
            kv_cache_manager = getattr(engine_ctx.scheduler, "kv_cache_manager", None)
        elif hasattr(engine_ctx, "kv_cache_manager"):
            # Fallback: try direct access (in case it's a scheduler)
            kv_cache_manager = engine_ctx.kv_cache_manager
        print(
            f"kv-mkt make_export_ctx: kv_cache_manager={'yes' if kv_cache_manager else 'no'} req={req_id}",
            flush=True,
        )
        
        if kv_cache_manager and hasattr(kv_cache_manager, "get_prefill_pages"):
            # Pass engine_ctx (EngineCore) so we can access model_executor for KV cache tensors
            kv_pages = kv_cache_manager.get_prefill_pages(req, engine_ctx=engine_ctx)
            print(
                f"kv-mkt make_export_ctx: got prefill pages req={req_id} k_ptrs={len(kv_pages.get('k_ptrs', []) or [])} v_ptrs={len(kv_pages.get('v_ptrs', []) or [])}",
                flush=True,
            )
        else:
            _dbg(
                "make_export_ctx: kv_cache_manager missing or lacks get_prefill_pages for req=%s",
                req_id,
            )
            print(
                f"kv-mkt make_export_ctx: missing get_prefill_pages req={req_id}",
                flush=True,
            )

        # Ensure we use the correct length
        length = kv_pages.get("length", prompt_len)

        _dbg(
            "make_export_ctx: req=%s export_len=%d k_ptrs=%d v_ptrs=%d page_ranges=%s",
            req_id,
            length,
            len(kv_pages.get("k_ptrs", []) or []),
            len(kv_pages.get("v_ptrs", []) or []),
            bool(kv_pages.get("page_ranges")),
        )
        print(
            f"kv-mkt make_export_ctx: final ctx req={req_id} length={length} k_ptrs={len(kv_pages.get('k_ptrs', []) or [])} v_ptrs={len(kv_pages.get('v_ptrs', []) or [])}",
            flush=True,
        )

        return {
            "device_id": device_id,
            "compat": compat,
            "tokens": tokens[:length] if len(tokens) >= length else tokens,  # Use prefix tokens up to length
            "kv_pages": kv_pages,
            "layout": layout,
            "length": length,
        }
    except Exception as e:
        _dbg(
            "make_export_ctx: exception for req=%s -> %s",
            getattr(req, "request_id", "unknown"),
            e,
        )
        print(
            f"kv-mkt make_export_ctx: exception req={getattr(req, 'request_id', 'unknown')} -> {e}",
            flush=True,
        )
        traceback.print_exc()
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
    
    print(
        f"kv-mkt _maybe_import_prefix: start req={getattr(req, 'request_id', 'unknown')}",
        flush=True,
    )
    
    # Check if kv-marketplace is enabled
    flags = getattr(engine_ctx, "vllm_config", None)
    if flags is None:
        flags = getattr(engine_ctx, "flags", None)

    if not flags or not getattr(flags, "kv_marketplace", False):
        return None

    min_prefix = getattr(flags, "kv_min_prefix", 64)

    kv_cache_manager = None
    if hasattr(engine_ctx, "scheduler"):
        kv_cache_manager = getattr(engine_ctx.scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)

    if (
        kv_cache_manager
        and hasattr(kv_cache_manager, "peek_prefix_cache_hit_length")
        and kv_cache_manager.peek_prefix_cache_hit_length(req) >= min_prefix
    ):
        _get_logger().debug(
            "kv-marketplace: skipping import for request %s; "
            "local prefix cache already has >=%d tokens",
            req.request_id,
            min_prefix,
        )
        return None

    timings: dict[str, float] = {}
    hook_start = time.perf_counter()
    
    ctx_start = time.perf_counter()
    ctx_dict = make_import_ctx(req, engine_ctx)
    timings["make_ctx_ms"] = (time.perf_counter() - ctx_start) * 1000.0
    if ctx_dict is None:
        timings["total_ms"] = (time.perf_counter() - hook_start) * 1000.0
        _get_logger().info(
            "kv-marketplace hook timings: make_ctx=%.2f ms plugin=0.00 ms materialize=0.00 ms "
            "release=0.00 ms total=%.2f ms (skip ctx)",
            timings["make_ctx_ms"],
            timings["total_ms"],
        )
        return None
    tokens = ctx_dict.get("tokens") or getattr(req, "prompt_token_ids", []) or []
    
    try:
        plugin_start = time.perf_counter()
        # Call the plugin's before_prefill function
        result = plugin.before_prefill(ctx_dict)
        timings["plugin_ms"] = (time.perf_counter() - plugin_start) * 1000.0

        if result is not None:
            lcp_len, dst_alloc = result
            print(
                f"kv-mkt _maybe_import_prefix: HIT req={getattr(req, 'request_id', 'unknown')} lcp_len={lcp_len}",
                flush=True,
            )
            global _PRINTED_HEAD_INFO
            if not _PRINTED_HEAD_INFO:
                model_cfg = getattr(flags, "model_config", None)
                if model_cfg is not None:
                    try:
                        total_q = model_cfg.get_total_num_attention_heads()
                    except Exception:
                        total_q = getattr(
                            getattr(model_cfg, "hf_config", None),
                            "num_attention_heads",
                            None,
                        )
                    try:
                        total_kv = model_cfg.get_total_num_kv_heads()
                    except Exception:
                        total_kv = getattr(
                            getattr(model_cfg, "hf_config", None),
                            "num_key_value_heads",
                            total_q,
                        )
                    if total_q and total_kv:
                        ratio = total_q // max(total_kv, 1)
                        print(
                            f"[HEADS] n_q_heads={total_q} n_kv_heads={total_kv} ratio={ratio}",
                            flush=True,
                        )
                        _PRINTED_HEAD_INFO = True
            if lcp_len > 0:
                setattr(req, "_kv_mkt_imported", True)
                setattr(req, "_kv_mkt_imported_len", lcp_len)
                orig_len = getattr(req, "_orig_prompt_len", len(tokens))
                setattr(req, "_kv_mkt_imported_full", lcp_len >= orig_len)
                setattr(req, "_kv_mkt_debug_probe", True)
                setattr(req, "_kv_mkt_debug_lcp_len", lcp_len)
            
            # Update request counters so scheduler/runner know prefix is satisfied.
            if hasattr(req, "seq_pos"):
                req.seq_pos = lcp_len
            if hasattr(req, "num_cached_tokens"):
                req.num_cached_tokens = lcp_len
            if hasattr(req, "num_computed_tokens"):
                req.num_computed_tokens = lcp_len
            
            page_ranges = dst_alloc.get("page_ranges") if isinstance(dst_alloc, dict) else None
            if page_ranges:
                flat_ranges = page_ranges
                if isinstance(page_ranges, list) and page_ranges and isinstance(page_ranges[0], list):
                    flat_ranges = page_ranges[0]
                preview = flat_ranges[:10]
                print(
                    f"[PAGES] dst first 10: {preview} total={len(flat_ranges)}",
                    flush=True,
                )
            
            # Inform allocator that prefix pages are materialized
            # Mirror the export lookup: try scheduler first, then direct access
            materialize_ms = 0.0
            if kv_cache_manager and hasattr(kv_cache_manager, "materialize_prefix"):
                mat_start = time.perf_counter()
                kv_cache_manager.materialize_prefix(req, dst_alloc, lcp_len)
                debug_blocks = []
                if hasattr(kv_cache_manager, "debug_request_blocks"):
                    try:
                        debug_blocks = kv_cache_manager.debug_request_blocks(
                            req.request_id, limit=10
                        )
                    except Exception:
                        debug_blocks = []
                if debug_blocks:
                    print(
                        f"[PAGES] allocator blocks first 10: {debug_blocks}",
                        flush=True,
                    )
                materialize_ms = (time.perf_counter() - mat_start) * 1000.0
            timings["materialize_ms"] = materialize_ms
            
            timings["total_ms"] = (time.perf_counter() - hook_start) * 1000.0
            _get_logger().info(
                "kv-marketplace hook timings: make_ctx=%.2f ms plugin=%.2f ms materialize=%.2f ms "
                "release=0.00 ms total=%.2f ms (hit)",
                timings["make_ctx_ms"],
                timings.get("plugin_ms", 0.0),
                timings.get("materialize_ms", 0.0),
                timings["total_ms"],
            )
            return result
        # Import failed after reserving blocks; release them for future use.
        if kv_cache_manager and hasattr(kv_cache_manager, "release_reserved_prefix"):
            rel_start = time.perf_counter()
            kv_cache_manager.release_reserved_prefix(req)
            timings["release_ms"] = (time.perf_counter() - rel_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - hook_start) * 1000.0
        _get_logger().info(
            "kv-marketplace hook timings: make_ctx=%.2f ms plugin=%.2f ms materialize=0.00 ms "
            "release=%.2f ms total=%.2f ms (miss)",
            timings["make_ctx_ms"],
            timings.get("plugin_ms", 0.0),
            timings.get("release_ms", 0.0),
            timings["total_ms"],
        )
        print(
            f"kv-mkt _maybe_import_prefix: MISS req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
    except Exception as e:
        _get_logger().warning(
            f"kv-marketplace before_prefill failed: {e}"
        )
        kv_cache_manager = None
        if hasattr(engine_ctx, "scheduler"):
            kv_cache_manager = getattr(engine_ctx.scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            kv_cache_manager = getattr(engine_ctx, "kv_cache_manager", None)
        if kv_cache_manager and hasattr(kv_cache_manager, "release_reserved_prefix"):
            rel_start = time.perf_counter()
            kv_cache_manager.release_reserved_prefix(req)
            timings["release_ms"] = (time.perf_counter() - rel_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - hook_start) * 1000.0
        _get_logger().info(
            "kv-marketplace hook timings: make_ctx=%.2f ms plugin=%.2f ms materialize=0.00 ms "
            "release=%.2f ms total=%.2f ms (error)",
            timings["make_ctx_ms"],
            timings.get("plugin_ms", 0.0),
            timings.get("release_ms", 0.0),
            timings["total_ms"],
        )
    return None


def _export_prefix(
    req: "Request",
    engine_ctx: Any,
) -> None:
    """Hook called after prefill to export KV cache.
    
    Args:
        req: The request that just completed prefill
        engine_ctx: Engine context (scheduler or EngineCore)
    """
    print(
        f"kv-mkt _export_prefix: invoked req={getattr(req, 'request_id', 'unknown')}",
        flush=True,
    )

    plugin = load_plugin()
    if not plugin:
        _dbg("kv-marketplace: PLUGIN NOT LOADED")
        print(
            f"kv-mkt _export_prefix: plugin unavailable req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
        return
    
    flags = getattr(engine_ctx, "vllm_config", None)
    if flags is None:
        flags = getattr(engine_ctx, "flags", None)
    
    if not flags or not getattr(flags, "kv_marketplace", False):
        _dbg("KV_MARKETPLACE FLAG IS FALSE, EXITING EXPORT")
        print(
            f"kv-mkt _export_prefix: disabled via flag req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
        return
    
    if getattr(req, "_kv_mkt_imported", False) and getattr(
        req, "_kv_mkt_imported_full", False
    ):
        setattr(req, "_kv_mkt_imported", False)
        setattr(req, "_kv_mkt_imported_full", False)
        _get_logger().debug(
            "kv-marketplace: skipping export for imported request %s",
            req.request_id,
        )
        print(
            f"kv-mkt _export_prefix: skipping imported req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
        return

    ctx_dict = make_export_ctx(req, engine_ctx)
    if ctx_dict is None:
        _dbg("_export_prefix: ctx_dict is None for req=%s", getattr(req, "request_id", "unknown"))
        print(
            f"kv-mkt _export_prefix: make_export_ctx returned None req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
        return
    
    try:
        _dbg(
            "_export_prefix: calling after_prefill req=%s len=%s k_ptrs=%d",
            getattr(req, "request_id", "unknown"),
            ctx_dict.get("length"),
            len(ctx_dict.get("kv_pages", {}).get("k_ptrs", []) or []),
        )
        plugin.after_prefill(ctx_dict)
        print(
            f"kv-mkt _export_prefix: after_prefill success req={getattr(req, 'request_id', 'unknown')}",
            flush=True,
        )
        _dbg(
            "_export_prefix: after_prefill completed req=%s",
            getattr(req, "request_id", "unknown"),
        )
        if getattr(req, "_kv_mkt_imported", False):
            setattr(req, "_kv_mkt_imported", False)
            setattr(req, "_kv_mkt_imported_full", False)
    except Exception as e:
        _get_logger().warning(
            f"kv-marketplace after_prefill failed: {e}", exc_info=True
        )
        _dbg(
            "_export_prefix: after_prefill raised for req=%s -> %s",
            getattr(req, "request_id", "unknown"),
            e,
        )
        traceback.print_exc()
