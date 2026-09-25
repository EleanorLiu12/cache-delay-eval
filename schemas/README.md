# Request trace format

One UTF-8 JSONL file per trace: a `trace_meta` header followed by requests.
[`request-trace-v1.schema.json`](request-trace-v1.schema.json) defines the fields;
`cache_delay_eval.trace` additionally validates ordering and session lineage.

```bash
cache-delay-validate-trace results/pattern-screen/traces/P512-negative-seed100.jsonl
```

## Current experiment

The session generator writes `block_hashes` and exact `prompt_tokens`.
`materialize.py` reconstructs the corresponding `token_ids` for GPU replay.
Both forms preserve shared leading blocks and session prefix extension.
Hash-only records need exact prompt length because their last block can be partial.

Each request has exactly one of `block_hashes`, `token_ids`, or `prompt`.
The reader also accepts plain-text prompts for schema compatibility; the current
GPU workflow uses materialized token IDs from synthetic session metadata.
Readers accept schema versions 1.0 and 1.1; writers emit 1.1.

## Timing and identity

- `arrival_time_ms`: integer offset from trace start, used for fixed-arrival replay.
- `session_id`: groups turns within one conversation.
- `parent_request_id`: earlier request in the same session.
- `user_id`: optional owner; a session cannot belong to two different users.
- `prefix_group`: shared-prefix annotation; actual reuse is determined by tokens/blocks.

The generator's `oracle_hit_rate` describes ideal historical reuse, without
finite GPU capacity or in-flight work. Live measurements record `cached_tokens`
and `prompt_tokens` separately; their ratio is the hit metric used in the
[research question](../docs/research.md).
