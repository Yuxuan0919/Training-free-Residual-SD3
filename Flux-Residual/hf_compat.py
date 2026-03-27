"""Compatibility shims for local HF package mismatches.

This workspace currently uses:
- transformers 4.39.0
- peft 0.17.0

Newer PEFT expects `transformers.EncoderDecoderCache` and
`transformers.HybridCache`, but these symbols do not exist in
transformers 4.39. Provide minimal fallbacks so diffusers can import.
"""

from __future__ import annotations


def apply_transformers_peft_compat() -> None:
    import transformers
    from transformers.cache_utils import Cache, DynamicCache
    import transformers.cache_utils as cache_utils

    if not hasattr(transformers, "EncoderDecoderCache"):

        class EncoderDecoderCache(Cache):
            def __init__(self, self_attention_cache=None, cross_attention_cache=None):
                self.self_attention_cache = self_attention_cache or DynamicCache()
                self.cross_attention_cache = cross_attention_cache or DynamicCache()
                self.is_updated = {}

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                return self.self_attention_cache.update(key_states, value_states, layer_idx, cache_kwargs)

            def get_seq_length(self, layer_idx=0):
                return self.self_attention_cache.get_seq_length(layer_idx)

            def get_max_length(self):
                return self.self_attention_cache.get_max_length()

            def to_legacy_cache(self):
                return self.self_attention_cache.to_legacy_cache()

            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                self_cache = DynamicCache.from_legacy_cache(past_key_values)
                return cls(self_attention_cache=self_cache, cross_attention_cache=DynamicCache())

        transformers.EncoderDecoderCache = EncoderDecoderCache
        cache_utils.EncoderDecoderCache = EncoderDecoderCache

    if not hasattr(transformers, "HybridCache"):

        class HybridCache(DynamicCache):
            def __init__(self, *args, **kwargs):
                super().__init__()

        transformers.HybridCache = HybridCache
        cache_utils.HybridCache = HybridCache
