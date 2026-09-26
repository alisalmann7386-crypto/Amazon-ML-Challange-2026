# Dependency and transliteration notes

Pinned versions are in `requirements.txt`. AnyAscii 0.3.3 provides offline, character-based Unicode-to-ASCII transliteration under ISC; it is not contextual translation. Example output verified locally: `श्री बालाजी ट्रेडर्स` → `sri balaji tredrs`. No externally looked-up business information is involved.

Primary references:
- [AnyAscii and license](https://github.com/anyascii/anyascii)
- [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn): sparse multiplication with bounded top-N output; no dense all-pairs matrix.
- [LightGBM](https://github.com/microsoft/LightGBM): primary classifier; CPU default and optional GPU attempt with CPU fallback.
- [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz): optimized fuzzy string measures.

Legacy scripts still work with the same numerical dependencies. No transformers, pretrained embeddings, neural checkpoints, external entity APIs, translations or geocoders are used. Dependency licenses remain with their upstream projects; the trained model contains learned classical parameters, not a pretrained neural model.
