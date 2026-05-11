# ferrotorch-bert

BERT-family encoder-only model composition for ferrotorch.

This crate assembles a standard `BertModel` from ferrotorch primitives
(token / position / token-type embeddings, multi-head self-attention,
post-norm residual blocks, GELU intermediate FFN) and adds a
`SentenceTransformer` wrapper that performs mean-pooling over the
attention mask followed by optional L2 normalization for sentence
embeddings.

The first pinned checkpoint is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(22M parameters, 6 layers, 384 hidden, Apache 2.0), mirrored byte-for-byte
to `ferrotorch/all-MiniLM-L6-v2` and registered in
`ferrotorch-hub`. The `examples/text_embedding_dump.rs` binary +
`scripts/verify_text_embedding_inference.py` harness verify cosine
similarity ≥ 0.999 and max-abs diff ≤ 0.01 against the upstream
`sentence_transformers.SentenceTransformer.encode(..., normalize_embeddings=True)`
output (Phase B.1 of real-artifact-driven development).
