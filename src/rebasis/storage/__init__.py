"""Durability and space management.

The layer contract puts ``storage`` near the bottom: it depends only on
``errors``, ``types`` and ``observability``.

The governing rule: **rebasis does not take ownership of user data.** ``probe``
and ``fit`` are read-only; the only write path is ``migrate``, and even that only
upserts — it never deletes.

``embedding_cache`` is the one thing here that ``probe`` writes, and it does not
weaken that rule: it writes vectors rebasis computed itself, into ``.rebasis/``,
under the retention ``gc`` already applies to that directory. The index is still
never touched.
"""

from __future__ import annotations

from rebasis.storage.atomic import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_stream,
    atomic_write_text,
    cleanup_stale_temp_files,
    free_bytes,
    require_free_space,
    rotate_backup,
)
from rebasis.storage.embedding_cache import (
    EMBEDDINGS_DIR,
    CachedEmbedder,
    CacheStats,
    EmbeddingCache,
    cache_enabled,
    cache_file_for,
    default_embedding_cache_dir,
    embedding_key,
    open_cached_embedder,
)
from rebasis.storage.gc import GCCandidate, GCPlan, apply_gc, plan_gc
from rebasis.storage.integrity import (
    BLOCK_SIZE,
    hash_array,
    hash_blocks,
    hash_bytes,
    hash_file,
    hash_state_dict,
    verify_blocks,
    verify_bytes,
    verify_state_dict,
)
from rebasis.storage.layout import (
    ADAPTERS_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    SHADOW_DIR,
    STATE_DIR_NAME,
)
from rebasis.storage.locks import lock_info, state_lock
from rebasis.storage.shadow import ShadowManifest, ShadowSegment, ShadowStore

__all__ = [
    "ADAPTERS_DIR",
    "BLOCK_SIZE",
    "CACHE_DIR",
    "EMBEDDINGS_DIR",
    "REPORTS_DIR",
    "SHADOW_DIR",
    "STATE_DIR_NAME",
    "CacheStats",
    "CachedEmbedder",
    "EmbeddingCache",
    "GCCandidate",
    "GCPlan",
    "ShadowManifest",
    "ShadowSegment",
    "ShadowStore",
    "apply_gc",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_stream",
    "atomic_write_text",
    "cache_enabled",
    "cache_file_for",
    "cleanup_stale_temp_files",
    "default_embedding_cache_dir",
    "embedding_key",
    "free_bytes",
    "hash_array",
    "hash_blocks",
    "hash_bytes",
    "hash_file",
    "hash_state_dict",
    "lock_info",
    "open_cached_embedder",
    "plan_gc",
    "require_free_space",
    "rotate_backup",
    "state_lock",
    "verify_blocks",
    "verify_bytes",
    "verify_state_dict",
]
